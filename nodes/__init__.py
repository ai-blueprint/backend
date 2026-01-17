"""
节点模块

提供便捷的节点创建工具，减少重复代码。

使用示例:
    from nodes import create_activation_node, create_module_node
    
    @node(...)
    def relu_node():
        return create_activation_node(F.relu)
    
    @node(...)
    def linear_layer():
        return create_module_node(
            nn.Linear,
            build_args=lambda p: (p["in_features"], p["out_features"])
        )
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple

# 类型别名
NodeFuncs = Tuple[Callable, Callable, Callable]
ShapeDict = Dict[str, List[int]]
ParamDict = Dict[str, Any]


def create_activation_node(
    activation_func: Callable[[torch.Tensor], torch.Tensor],
    *,
    shape_unchanged: bool = True
) -> NodeFuncs:
    """
    创建激活函数节点
    
    适用于：ReLU, Sigmoid, Tanh, GELU, SiLU 等不改变形状的激活函数。
    
    参数:
        activation_func: 激活函数，如 F.relu, torch.sigmoid
        shape_unchanged: 输出形状是否与输入相同
    
    返回:
        (infer, build, compute) 三元组
    
    示例:
        @node(opcode="relu", ...)
        def relu_node():
            return create_activation_node(F.relu)
    """
    def infer(input_shapes: ShapeDict, params: ParamDict) -> Optional[ShapeDict]:
        if shape_unchanged:
            return input_shapes.get("x")
        return None
    
    def build(params: ParamDict) -> None:
        return None
    
    def compute(x: torch.Tensor, layer: Any) -> torch.Tensor:
        return activation_func(x)
    
    return infer, build, compute


def create_parameterized_activation_node(
    activation_func: Callable,
    param_name: str,
    default_value: Any = None
) -> NodeFuncs:
    """
    创建带参数的激活函数节点
    
    适用于：LeakyReLU(negative_slope), Softmax(dim) 等需要参数的激活函数。
    
    参数:
        activation_func: 激活函数
        param_name: 参数名称
        default_value: 参数默认值
    
    返回:
        (infer, build, compute) 三元组
    """
    def infer(input_shapes: ShapeDict, params: ParamDict) -> Optional[ShapeDict]:
        return input_shapes.get("x")
    
    def build(params: ParamDict) -> Any:
        return params.get(param_name, default_value)
    
    def compute(x: torch.Tensor, layer: Any) -> torch.Tensor:
        return activation_func(x, **{param_name: layer})
    
    return infer, build, compute


def create_module_node(
    module_class: type,
    *,
    build_args: Callable[[ParamDict], tuple] = None,
    build_kwargs: Callable[[ParamDict], dict] = None,
    infer_shape: Callable[[ShapeDict, ParamDict], ShapeDict] = None,
    needs_input_shapes: bool = True
) -> NodeFuncs:
    """
    创建基于 nn.Module 的节点
    
    适用于：Linear, Conv2d, BatchNorm2d, Dropout 等 PyTorch 层。
    
    参数:
        module_class: nn.Module 子类
        build_args: 从参数构建模块位置参数的函数
        build_kwargs: 从参数构建模块关键字参数的函数
        infer_shape: 形状推断函数
        needs_input_shapes: build 函数是否需要输入形状
    
    返回:
        (infer, build, compute) 三元组
    
    示例:
        @node(opcode="linear", ...)
        def linear_layer():
            return create_module_node(
                nn.Linear,
                build_args=lambda p: (p["in_features"], p["out_features"]),
                build_kwargs=lambda p: {"bias": p.get("bias", True)}
            )
    """
    def infer(input_shapes: ShapeDict, params: ParamDict) -> Optional[ShapeDict]:
        if infer_shape:
            return infer_shape(input_shapes, params)
        return {"out": input_shapes.get("x")}
    
    def build(input_shapes_or_params, params=None):
        # 支持两种签名：build(params) 和 build(input_shapes, params)
        if params is None:
            params = input_shapes_or_params
        
        args = build_args(params) if build_args else ()
        kwargs = build_kwargs(params) if build_kwargs else {}
        
        return module_class(*args, **kwargs)
    
    def compute(x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        return layer(x)
    
    return infer, build, compute


def create_binary_op_node(
    op_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    input_ports: Tuple[str, str] = ("x", "y")
) -> NodeFuncs:
    """
    创建二元运算节点
    
    适用于：加法、减法、乘法、矩阵乘法等双输入运算。
    
    参数:
        op_func: 运算函数，如 torch.add, torch.matmul
        input_ports: 输入端口名称
    
    返回:
        (infer, build, compute) 三元组
    
    示例:
        @node(opcode="add", ...)
        def add_node():
            return create_binary_op_node(torch.add)
    """
    port_x, port_y = input_ports
    
    def infer(input_shapes: ShapeDict, params: ParamDict) -> Optional[ShapeDict]:
        return input_shapes.get(port_x)
    
    def build(params: ParamDict) -> None:
        return None
    
    def compute(inputs: Dict[str, torch.Tensor], layer: Any) -> torch.Tensor:
        x = inputs.get(port_x)
        y = inputs.get(port_y)
        return op_func(x, y)
    
    return infer, build, compute


def create_loss_node(
    loss_class: type,
    *,
    input_ports: Tuple[str, str] = ("input", "target"),
    target_dtype: torch.dtype = None,
    build_kwargs: Callable[[ParamDict], dict] = None
) -> NodeFuncs:
    """
    创建损失函数节点
    
    适用于：CrossEntropy, MSE, BCE 等损失函数。
    
    参数:
        loss_class: 损失函数类
        input_ports: 输入端口名称 (预测, 目标)
        target_dtype: 目标张量的数据类型转换
        build_kwargs: 构建参数函数
    
    返回:
        (infer, build, compute) 三元组
    """
    port_pred, port_target = input_ports
    
    def infer(input_shapes: ShapeDict, params: ParamDict) -> ShapeDict:
        return {"loss": [1]}
    
    def build(params: ParamDict) -> nn.Module:
        kwargs = build_kwargs(params) if build_kwargs else {}
        return loss_class(**kwargs)
    
    def compute(inputs: Dict[str, torch.Tensor], layer: nn.Module) -> torch.Tensor:
        pred = inputs.get(port_pred)
        target = inputs.get(port_target)
        
        if target_dtype and target.dtype != target_dtype:
            target = target.to(target_dtype)
        
        return layer(pred, target)
    
    return infer, build, compute


def create_passthrough_node(output_port: str = "out") -> NodeFuncs:
    """
    创建透传节点
    
    适用于：输出节点等直接透传数据的节点。
    
    参数:
        output_port: 输出端口名称
    
    返回:
        (infer, build, compute) 三元组
    """
    def infer(input_shapes: ShapeDict, params: ParamDict) -> Optional[ShapeDict]:
        # 透传输入形状
        for key, shape in input_shapes.items():
            return {output_port: shape}
        return None
    
    def build(params: ParamDict) -> None:
        return None
    
    def compute(x: Any, layer: Any) -> Any:
        # 直接返回输入（引擎已解包为张量）
        return {output_port: x} if output_port else x
    
    return infer, build, compute


# ==================== 形状工具 ====================

def shape_unchanged(input_shapes: ShapeDict, params: ParamDict) -> ShapeDict:
    """形状不变的推断函数"""
    return {"out": input_shapes.get("x")}


def infer_linear_shape(input_shapes: ShapeDict, params: ParamDict) -> ShapeDict:
    """Linear层形状推断"""
    shape = list(input_shapes.get("x", [1, 128]))
    shape[-1] = params.get("out_features", shape[-1])
    return {"out": shape}


def infer_conv2d_shape(input_shapes: ShapeDict, params: ParamDict) -> ShapeDict:
    """Conv2d层形状推断"""
    shape = list(input_shapes.get("x", [1, 3, 32, 32]))
    out_c = params.get("out_channels", shape[1])
    k = params.get("kernel_size", 3)
    s = params.get("stride", 1)
    p = params.get("padding", 0)
    h_out = (shape[2] + 2 * p - k) // s + 1
    w_out = (shape[3] + 2 * p - k) // s + 1
    return {"out": [shape[0], out_c, h_out, w_out]}


def infer_pool2d_shape(input_shapes: ShapeDict, params: ParamDict) -> ShapeDict:
    """Pool2d层形状推断"""
    shape = list(input_shapes.get("x", [1, 64, 32, 32]))
    k = params.get("kernel_size", 2)
    s = params.get("stride", 2)
    p = params.get("padding", 0)
    h_out = (shape[2] + 2 * p - k) // s + 1
    w_out = (shape[3] + 2 * p - k) // s + 1
    return {"out": [shape[0], shape[1], h_out, w_out]}
