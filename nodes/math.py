"""
数学运算节点组

提供基础数学运算节点：加法、矩阵乘法、求和等。
"""

import torch
from typing import Any, Dict, List, Optional

from decorators import category, node
from nodes import create_binary_op_node


# ==================== 分类定义 ====================

@category(
    id="math",
    name="数学运算",
    color="#FFB6C1",
    icon="base64…"
)
def math_category():
    pass


# ==================== 节点定义 ====================

@node(
    opcode="add",
    name="加法 (+)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def add_node():
    """张量加法"""
    return create_binary_op_node(torch.add)


@node(
    opcode="matmul",
    name="矩阵乘法 (@)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def matmul_node():
    """矩阵乘法"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> List[int]:
        shape_x = input_shapes.get("x", [0, 0])
        shape_y = input_shapes.get("y", [0, 0])
        return [shape_x[0], shape_y[-1]]
    
    def build(params: Dict) -> None:
        return None
    
    def compute(inputs: Dict[str, torch.Tensor], layer: Any) -> torch.Tensor:
        return torch.matmul(inputs["x"], inputs["y"])
    
    return infer, build, compute


@node(
    opcode="sum",
    name="求和 (Σ)",
    ports={"in": ["x"], "out": ["result"]},
    params={"dim": None, "keepdim": False}
)
def sum_node():
    """张量求和"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> List[int]:
        dim = params.get("dim")
        if dim is None:
            return [1]
        shape = list(input_shapes.get("x", []))
        if not params.get("keepdim", False) and shape:
            shape.pop(dim)
        return shape
    
    def build(params: Dict) -> Dict[str, Any]:
        return {
            "dim": params.get("dim"),
            "keepdim": params.get("keepdim", False)
        }
    
    def compute(x: torch.Tensor, layer: Dict[str, Any]) -> torch.Tensor:
        dim = layer.get("dim") if layer else None
        keepdim = layer.get("keepdim", False) if layer else False
        return torch.sum(x, dim=dim, keepdim=keepdim)
    
    return infer, build, compute


@node(
    opcode="mul",
    name="乘法 (*)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def mul_node():
    """张量逐元素乘法"""
    return create_binary_op_node(torch.mul)


@node(
    opcode="sub",
    name="减法 (-)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def sub_node():
    """张量减法"""
    return create_binary_op_node(torch.sub)


@node(
    opcode="div",
    name="除法 (/)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def div_node():
    """张量除法"""
    return create_binary_op_node(torch.div)


@node(
    opcode="mean",
    name="均值 (μ)",
    ports={"in": ["x"], "out": ["result"]},
    params={"dim": None, "keepdim": False}
)
def mean_node():
    """张量均值"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> List[int]:
        dim = params.get("dim")
        if dim is None:
            return [1]
        shape = list(input_shapes.get("x", []))
        if not params.get("keepdim", False) and shape:
            shape.pop(dim)
        return shape
    
    def build(params: Dict) -> Dict[str, Any]:
        return {
            "dim": params.get("dim"),
            "keepdim": params.get("keepdim", False)
        }
    
    def compute(x: torch.Tensor, layer: Dict[str, Any]) -> torch.Tensor:
        dim = layer.get("dim") if layer else None
        keepdim = layer.get("keepdim", False) if layer else False
        return torch.mean(x, dim=dim, keepdim=keepdim)
    
    return infer, build, compute
