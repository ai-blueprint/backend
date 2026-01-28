"""
基础节点组

提供基础的输入/输出节点。
"""

from typing import Any, Dict, List, Optional

from decorators import category, node
from nodes import create_passthrough_node


# ==================== 分类定义 ====================

@category(
    id="basic",
    label="基础",
    color="#8B92E5",
    icon="base64"
)
def basic_category():
    pass


# ==================== 节点定义 ====================

@node(
    opcode="input",
    label="输入",
    outputs=["out"],
    params={"输出维度": [1, 10]}
)
def input_node():
    """
    输入节点

    这是蓝图的入口点，不执行任何计算。
    引擎会直接透传 initial_inputs 中的数据。
    """
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        return {"out": params.get("输出维度", [1, 10])}

    def build(input_shapes: Dict[str, List[int]], params: Dict) -> None:
        return None

    def compute(x: Any, layer: Any) -> None:
        # 输入节点不执行计算，由引擎透传数据
        return None

    return infer, build, compute


@node(
    opcode="output",
    label="输出",
    inputs=["in"],
    params={}
)
def output_node():
    """
    输出节点

    这是蓝图的出口点，直接透传输入数据。
    """
    return create_passthrough_node(output_port="out")


@node(
    opcode="constant",
    label="常量",
    outputs=["out"],
    params={"value": 0}
)
def constant_node():
    """常量节点"""
    import torch

    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        return {"out": [1]}

    def build(input_shapes: Dict[str, List[int]], params: Dict) -> Any:
        value = params.get("value", 0)
        return torch.tensor([value], dtype=torch.float32)

    def compute(x: Any, layer: Any) -> Any:
        return layer

    return infer, build, compute


@node(
    opcode="debug",
    label="调试输出",
    inputs=["x"],
    outputs=["out"],
    params={"label": "debug"}
)
def debug_node():
    """
    调试节点

    打印输入数据并透传。用于调试蓝图执行。
    """
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        return {"out": input_shapes.get("x")}

    def build(input_shapes: Dict[str, List[int]], params: Dict) -> str:
        return params.get("label", "debug")

    def compute(x: Any, layer: str) -> Any:
        print(f"🔍 [{layer}] shape={x.shape if hasattr(x, 'shape') else 'N/A'}, dtype={x.dtype if hasattr(x, 'dtype') else type(x)}")
        return x

    return infer, build, compute
