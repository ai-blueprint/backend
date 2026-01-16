# ================= 数学运算节点组 =================
import sys
import os

# 添加backend目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from decorators import category, node
import torch

@category(id="math", name="数学运算", color="#FFB6C1", icon="base64…")
def math():
    pass

@node(
    opcode="add",
    name="加法 (+)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def add_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return inputs["x"] + inputs["y"]

    return infer, build, compute

@node(
    opcode="sub",
    name="减法 (-)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def sub_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return inputs["x"] - inputs["y"]

    return infer, build, compute

@node(
    opcode="mul",
    name="乘法 (*)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def mul_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return inputs["x"] * inputs["y"]

    return infer, build, compute

@node(
    opcode="div",
    name="除法 (/)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def div_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return inputs["x"] / inputs["y"]

    return infer, build, compute

@node(
    opcode="pow",
    name="幂运算 (^)",
    ports={"in": ["x", "exponent"], "out": ["result"]},
    params={}
)
def pow_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return torch.pow(inputs["x"], inputs["exponent"])

    return infer, build, compute

@node(
    opcode="sqrt",
    name="平方根 (√)",
    ports={"in": ["x"], "out": ["result"]},
    params={}
)
def sqrt_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return torch.sqrt(inputs["x"])

    return infer, build, compute

@node(
    opcode="abs",
    name="绝对值 (|x|)",
    ports={"in": ["x"], "out": ["result"]},
    params={}
)
def abs_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return torch.abs(inputs["x"])

    return infer, build, compute

@node(
    opcode="neg",
    name="取负 (-x)",
    ports={"in": ["x"], "out": ["result"]},
    params={}
)
def neg_node():
    def infer(input_shapes, params):
        return input_shapes[0]

    def build(params):
        return None

    def compute(inputs, layer):
        return -inputs["x"]

    return infer, build, compute

@node(
    opcode="matmul",
    name="矩阵乘法 (@)",
    ports={"in": ["x", "y"], "out": ["result"]},
    params={}
)
def matmul_node():
    def infer(input_shapes, params):
        return [input_shapes[0][0], input_shapes[1][1]]

    def build(params):
        return None

    def compute(inputs, layer):
        return torch.matmul(inputs["x"], inputs["y"])

    return infer, build, compute

@node(
    opcode="sum",
    name="求和 (Σ)",
    ports={"in": ["x"], "out": ["result"]},
    params={"dim": None, "keepdim": False}
)
def sum_node():
    def infer(input_shapes, params):
        if params["dim"] is None:
            return [1]
        shape = input_shapes[0].copy()
        if not params["keepdim"]:
            del shape[params["dim"]]
        return shape

    def build(params):
        return None

    def compute(inputs, layer):
        return torch.sum(
            inputs["x"],
            dim=params["dim"],
            keepdim=params["keepdim"]
        )

    return infer, build, compute

@node(
    opcode="mean",
    name="平均值 (μ)",
    ports={"in": ["x"], "out": ["result"]},
    params={"dim": None, "keepdim": False}
)
def mean_node():
    def infer(input_shapes, params):
        if params["dim"] is None:
            return [1]
        shape = input_shapes[0].copy()
        if not params["keepdim"]:
            del shape[params["dim"]]
        return shape

    def build(params):
        return None

    def compute(inputs, layer):
        return torch.mean(
            inputs["x"],
            dim=params["dim"],
            keepdim=params["keepdim"]
        )

    return infer, build, compute
