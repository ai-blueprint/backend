"""
蓝图执行引擎

核心模块：负责解析蓝图并按拓扑顺序执行节点计算。
"""

import torch
import inspect
from typing import Any, Callable, Dict, List, Optional

import loader  # 调用 loader.load_all()
from context import ExecutionContext  # 创建执行上下文 context
import registry  # 获取节点定义
from utils.graph import topological_sort  # 调用 topo_sort 得到执行顺序
from utils.tensor import extract_single_input
from utils.validation import coerce_type
from utils.safe import safe_call, safe_get


def run(  # 运行蓝图
    blueprint: Dict[str, Any],  # 参数：blueprint
    inputs: Dict[str, Any],  # 参数：inputs
    on_progress: Optional[Callable] = None,  # 参数：on_progress
) -> Dict[str, Any]:
    """
    运行蓝图

    参数:
        blueprint: 蓝图数据，包含nodes和edges
        inputs: 用户传入的初始数据 {node_id: {port: value}}
        on_progress: 进度回调函数 (node_id, output) -> None

    返回:
        执行结果 {"success": True} 或 {"node_id": str, "error": str}
    """
    loader.load_all_nodes()  # 调用 loader.load_all() 确保节点已加载

    context = ExecutionContext(inputs)  # 创建执行上下文 context

    nodes = blueprint.get("nodes", [])  # 解析 blueprint 得到 nodes
    edges = blueprint.get("edges", [])  # 解析 blueprint 得到 edges

    # 转换nodes列表为字典，便于查询
    nodes_data = {node["id"]: node for node in nodes if "id" in node}

    execution_order = topological_sort(nodes_data, edges)  # 调用 topo_sort 得到执行顺序

    # 执行形状推断和构建层实例
    for node_id in execution_order:
        node_data = nodes_data.get(node_id)
        if not node_data:
            continue

        node_type = _get_node_type(node_data)
        params = _extract_params(node_data)

        node_def = registry.get_function(node_type)  # 从 registry 获取节点定义
        if not node_def:
            continue

        func_factory = node_def.get("func")
        if not func_factory:
            continue

        try:
            infer, build, compute = func_factory()

            # 执行形状推断
            node_inputs = context.get_inputs(node_id, edges)
            input_shapes = _compute_input_shapes(node_inputs)

            # 构建或获取层实例
            if context.get_layer(node_id) is not None:  # 如果 context 里有缓存，直接用
                continue
            else:  # 否则
                # 获取节点参数 params
                layer = _invoke_build(
                    build, input_shapes, params
                )  # 调用定义里的 build 函数
                context.store_layer(node_id, layer)  # 存入 context
        except Exception:
            pass  # 忽略构建错误，在执行时处理

    # 遍历执行
    for node_id in execution_order:
        try:  # 尝试执行节点
            node_data = nodes_data.get(node_id)
            if not node_data:
                continue

            output = execute_node(node_id, node_data, context, edges)  # 执行单个节点
            context.store_result(node_id, output)  # 存储结果

            if on_progress:  # 如果有回调，发送进度
                on_progress(node_id, output)

        except Exception as e:  # 捕获异常
            return {"node_id": node_id, "error": str(e)}  # 返回 {node_id, error}
            # 终止执行，跳出遍历（通过return实现）

    return {"success": True}  # 返回完成信息


def execute_node(  # 执行单个节点
    node_id: str,  # 参数：node_id
    node_data: Dict[str, Any],  # 参数：node_data
    context: ExecutionContext,  # 参数：context
    edges: List[Dict[str, Any]],  # 参数：edges
) -> Dict[str, Any]:
    """
    执行单个节点

    参数:
        node_id: 节点ID
        node_data: 节点数据
        context: 执行上下文
        edges: 边列表

    返回:
        输出数据字典
    """
    node_type = _get_node_type(node_data)  # 获取节点类型
    params = _extract_params(node_data)

    node_def = registry.get_function(node_type)  # 从 registry 获取节点定义

    if not node_def:  # 如果找不到定义，抛异常
        raise Exception(f"未找到节点定义: {node_type}")

    func_factory = node_def.get("func")
    if not func_factory:
        raise Exception(f"节点 {node_type} 缺少func定义")

    infer, build, compute = func_factory()

    # 收集输入
    is_input_node = node_type == "input"
    inputs = context.get_inputs(
        node_id, edges
    )  # 收集输入：context.get_inputs(node_id, edges)

    if is_input_node and not inputs:  # 如果是输入节点且无输入
        inputs = context.get_initial_input(
            node_id
        )  # 使用 context.get_initial_input(node_id)
        # 确保inputs是字典格式
        if not isinstance(inputs, dict):
            inputs = {"out": inputs}
    # 否则（已在上面处理）

    # 执行计算
    layer = context.get_layer(node_id)  # 调用定义里的 compute 函数
    output = _run_compute(
        compute, layer, inputs, node_type, node_def
    )  # 传入 layer, inputs, params

    # 确保输出是字典格式
    output = ensure_dict_output(output, node_type, node_def)

    return output  # 返回输出


def ensure_dict_output(  # 确保字典输出
    output: Any, node_type: str, node_def: Dict[str, Any]
) -> Dict[str, Any]:
    """
    确保输出为字典格式

    参数:
        output: 原始输出
        node_type: 节点类型
        node_def: 节点定义

    返回:
        字典格式的输出
    """
    if isinstance(output, dict):  # 如果已经是字典，直接返回
        return output

    # 如果是单值，包装成 {output: value}
    out_ports = safe_get(node_def, "ports", "out", default=["out"])
    if not isinstance(output, tuple):
        return {out_ports[0]: output}

    # 如果是元组，按 outputs 端口名包装
    result = {}
    for i, value in enumerate(output):
        if i < len(out_ports):
            result[out_ports[i]] = value
    return result


# ==================== 辅助函数（支持主流程） ====================


def _get_node_type(node_info: Dict[str, Any]) -> str:
    """从节点信息中提取类型（opcode）"""
    return safe_get(node_info, "data", "nodeKey") or node_info.get("type", "")


def _extract_params(node_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    从节点信息中提取参数

    支持两种格式：
    1. 新格式: data.params = {"key": {"type": "...", "default": value}}
    2. 旧格式: params = {"key": value}
    """
    raw_params = safe_get(node_info, "data", "params", default={})

    if not raw_params:
        return node_info.get("params", {})

    # 检测是否为新格式
    first_value = next(iter(raw_params.values()), None)
    if isinstance(first_value, dict) and "default" in first_value:
        return _convert_new_format_params(raw_params)

    return raw_params


def _convert_new_format_params(raw_params: Dict[str, Dict]) -> Dict[str, Any]:
    """将新格式参数转换为简单键值对"""
    result = {}
    for key, param_obj in raw_params.items():
        default_val = param_obj.get("default")
        param_type = param_obj.get("type", "string")
        result[key] = coerce_type(default_val, param_type, default=default_val)
    return result


def _compute_input_shapes(inputs: Dict[str, Any]) -> Dict[str, List[int]]:
    """计算输入的形状信息"""
    shapes = {}
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            shapes[key] = list(value.shape)
        elif isinstance(value, (list, tuple)):
            shapes[key] = _infer_list_shape(value)
        else:
            shapes[key] = value
    return shapes


def _infer_list_shape(lst: Any) -> List[int]:
    """推断嵌套列表的形状"""
    shape = []
    current = lst
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if len(current) == 0:
            break
        current = current[0]
    return shape


def _invoke_build(
    build_func: Callable, input_shapes: Dict[str, List[int]], params: Dict[str, Any]
) -> Any:
    """根据build函数签名调用构建"""
    sig = inspect.signature(build_func)
    param_count = len(sig.parameters)

    if param_count == 2:
        return safe_call(build_func, input_shapes, params, default=None)
    else:
        return safe_call(build_func, params, default=None)


def _run_compute(
    compute_func: Callable,
    layer: Any,
    inputs: Dict[str, Any],
    node_type: str,
    node_def: Dict[str, Any],
) -> Any:
    """
    执行节点计算

    策略：
    1. 无输入节点 -> compute(None, layer)
    2. 单输入 + nn.Module -> 直接调用 layer(x)
    3. 单输入 + 其他 -> compute(x, layer)
    4. 多输入 -> compute(inputs, layer)
    """
    input_ports = safe_get(node_def, "ports", "in", default=[])

    # 无输入节点
    if len(input_ports) == 0:
        return safe_call(compute_func, None, layer, default=None)

    # 单输入节点
    if len(input_ports) == 1:
        port_name = input_ports[0]
        x = extract_single_input(inputs, port_name)

        if x is None:
            return None

        # nn.Module 直接调用
        if isinstance(layer, torch.nn.Module):
            return safe_call(layer, x, default=None)

        return safe_call(compute_func, x, layer, default=None)

    # 多输入节点
    return safe_call(compute_func, inputs, layer, default=None)
