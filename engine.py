"""
蓝图执行引擎

核心模块：负责解析蓝图并按拓扑顺序执行节点计算。

设计原则：
1. 语义化函数分离 - 每个方法只做一件事
2. 容错处理 - 异常不会中断整个执行流程
3. 松散耦合 - 通过Registry获取节点定义
"""

import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from registry import Registry
from utils.tensor import extract_single_input
from utils.graph import topological_sort, get_node_inputs
from utils.validation import normalize_params, coerce_type
from utils.safe import safe_call, safe_get


class BlueprintEngine:
    """
    蓝图执行引擎
    
    职责：
    1. 解析蓝图结构（节点、边）
    2. 计算执行顺序（拓扑排序）
    3. 按顺序执行每个节点
    4. 管理节点层的缓存
    
    使用示例：
        engine = BlueprintEngine(blueprint_data)
        results = engine.execute(initial_inputs)
    """
    
    def __init__(self, blueprint_data: Dict[str, Any]):
        """
        初始化引擎
        
        参数:
            blueprint_data: 蓝图数据，包含 nodes 和 edges
        """
        # 解析蓝图数据
        self.nodes_data = self._parse_nodes(blueprint_data)
        self.edges = blueprint_data.get('edges', [])
        
        # 初始化注册表
        self.registry = Registry()
        self.registry.load_nodes()
        
        # 缓存
        self._layer_cache: Dict[str, Any] = {}      # 节点层实例缓存
        self._funcs_cache: Dict[str, Tuple] = {}    # 节点函数缓存
    
    # ==================== 公共API ====================
    
    def execute(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步执行蓝图
        
        参数:
            initial_inputs: 初始输入数据 {node_id: {port: value}}
        
        返回:
            所有节点的执行结果 {node_id: output}
        """
        execution_order = self._compute_execution_order()
        results = {}
        
        for node_id in execution_order:
            result = self._execute_node(node_id, results, initial_inputs)
            results[node_id] = result
        
        return results
    
    async def execute_with_callback(
        self, 
        initial_inputs: Dict[str, Any],
        on_node_complete: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        异步执行蓝图，支持节点完成回调
        
        参数:
            initial_inputs: 初始输入数据
            on_node_complete: 节点完成时的回调函数 async def(node_id, output)
        
        返回:
            所有节点的执行结果
        """
        execution_order = self._compute_execution_order()
        results = {}
        
        for node_id in execution_order:
            result = self._execute_node(node_id, results, initial_inputs)
            results[node_id] = result
            
            if on_node_complete:
                await on_node_complete(node_id, result)
        
        return results
    
    # ==================== 执行流程 ====================
    
    def _execute_node(
        self, 
        node_id: str, 
        results: Dict[str, Any],
        initial_inputs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        执行单个节点
        
        执行流程：
        1. 获取节点信息
        2. 收集输入数据
        3. 获取/构建层实例
        4. 执行计算
        5. 处理特殊节点
        6. 规范化输出格式
        """
        # Step 1: 获取节点信息
        node_info = self.nodes_data.get(node_id)
        if not node_info:
            return None
        
        node_type = self._get_node_type(node_info)
        params = self._extract_params(node_info)
        
        # Step 2: 获取节点函数
        funcs = self._get_node_funcs(node_id, node_type)
        if not funcs:
            return None
        
        infer, build, compute = funcs
        
        # Step 3: 收集输入
        inputs = get_node_inputs(node_id, self.edges, results)
        
        # Step 4: 构建层
        layer = self._build_layer(node_id, build, inputs, params)
        
        # Step 5: 执行计算
        output = self._run_compute(compute, layer, inputs, node_type)
        
        # Step 6: 后处理
        output = self._handle_input_node(node_id, output, initial_inputs)
        output = self._ensure_dict_output(output, node_type)
        
        return output
    
    def _run_compute(
        self,
        compute_func: Callable,
        layer: Any,
        inputs: Dict[str, Any],
        node_type: str
    ) -> Any:
        """
        执行节点计算
        
        策略：
        1. 无输入节点 -> compute(None, layer)
        2. 单输入 + nn.Module -> 直接调用 layer(x)
        3. 单输入 + 其他 -> compute(x, layer)
        4. 多输入 -> compute(inputs, layer)
        """
        input_ports = self._get_input_ports(node_type)
        
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
    
    # ==================== 节点信息提取 ====================
    
    def _get_node_type(self, node_info: Dict[str, Any]) -> str:
        """从节点信息中提取类型（opcode）"""
        return safe_get(node_info, 'data', 'nodeKey') or node_info.get('type', '')
    
    def _extract_params(self, node_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        从节点信息中提取参数
        
        支持两种格式：
        1. 新格式: data.params = {"key": {"type": "...", "default": value}}
        2. 旧格式: params = {"key": value}
        """
        raw_params = safe_get(node_info, 'data', 'params', default={})
        
        if not raw_params:
            return node_info.get('params', {})
        
        # 检测是否为新格式
        first_value = next(iter(raw_params.values()), None)
        if isinstance(first_value, dict) and 'default' in first_value:
            return self._convert_new_format_params(raw_params)
        
        return raw_params
    
    def _convert_new_format_params(
        self, 
        raw_params: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """将新格式参数转换为简单键值对"""
        result = {}
        for key, param_obj in raw_params.items():
            default_val = param_obj.get('default')
            param_type = param_obj.get('type', 'string')
            result[key] = coerce_type(default_val, param_type, default=default_val)
        return result
    
    def _get_input_ports(self, node_type: str) -> List[str]:
        """获取节点的输入端口列表"""
        node_def = self.registry.get_function(node_type)
        if not node_def:
            return []
        return safe_get(node_def, 'ports', 'in', default=[])
    
    def _get_output_ports(self, node_type: str) -> List[str]:
        """获取节点的输出端口列表"""
        node_def = self.registry.get_function(node_type)
        if not node_def:
            return ['out']
        return safe_get(node_def, 'ports', 'out', default=['out'])
    
    # ==================== 缓存管理 ====================
    
    def _get_node_funcs(
        self, 
        node_id: str, 
        node_type: str
    ) -> Optional[Tuple[Callable, Callable, Callable]]:
        """获取节点的 (infer, build, compute) 函数三元组"""
        if node_id in self._funcs_cache:
            return self._funcs_cache[node_id]
        
        node_def = self.registry.get_function(node_type)
        if not node_def:
            return None
        
        func_factory = node_def.get('func')
        if not func_factory:
            return None
        
        funcs = safe_call(func_factory, default=None)
        if funcs:
            self._funcs_cache[node_id] = funcs
        
        return funcs
    
    def _build_layer(
        self,
        node_id: str,
        build_func: Callable,
        inputs: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Any:
        """获取或构建节点的层实例"""
        if node_id in self._layer_cache:
            return self._layer_cache[node_id]
        
        # 计算输入形状
        input_shapes = self._compute_input_shapes(inputs)
        
        # 根据 build 函数签名决定调用方式
        layer = self._invoke_build(build_func, input_shapes, params)
        
        self._layer_cache[node_id] = layer
        return layer
    
    def _invoke_build(
        self,
        build_func: Callable,
        input_shapes: Dict[str, List[int]],
        params: Dict[str, Any]
    ) -> Any:
        """根据build函数签名调用构建"""
        sig = inspect.signature(build_func)
        param_count = len(sig.parameters)
        
        if param_count == 2:
            return safe_call(build_func, input_shapes, params, default=None)
        else:
            return safe_call(build_func, params, default=None)
    
    def _compute_input_shapes(
        self, 
        inputs: Dict[str, Any]
    ) -> Dict[str, List[int]]:
        """计算输入的形状信息"""
        shapes = {}
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                shapes[key] = list(value.shape)
            elif isinstance(value, (list, tuple)):
                shapes[key] = self._infer_list_shape(value)
            else:
                shapes[key] = value
        return shapes
    
    @staticmethod
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
    
    # ==================== 输出处理 ====================
    
    def _handle_input_node(
        self,
        node_id: str,
        output: Any,
        initial_inputs: Dict[str, Any]
    ) -> Any:
        """处理输入节点：透传初始数据"""
        if output is None and node_id in initial_inputs:
            return initial_inputs[node_id]
        return output
    
    def _ensure_dict_output(
        self, 
        output: Any, 
        node_type: str
    ) -> Optional[Dict[str, Any]]:
        """确保输出为字典格式"""
        if output is None or isinstance(output, dict):
            return output
        
        out_ports = self._get_output_ports(node_type)
        return {out_ports[0]: output}
    
    # ==================== 蓝图解析 ====================
    
    def _parse_nodes(
        self, 
        blueprint_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """将节点列表转换为ID索引的字典"""
        nodes = blueprint_data.get('nodes', [])
        return {node['id']: node for node in nodes if 'id' in node}
    
    def _compute_execution_order(self) -> List[str]:
        """计算节点执行顺序（拓扑排序）"""
        return topological_sort(self.nodes_data, self.edges)
