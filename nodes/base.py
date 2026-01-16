# ================= 实际节点组编写示例 =================
import sys
import os

# 添加backend目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from decorators import category, node
import torch


@category(id="basic", name="基础", color="#8B92E5", icon="base64…")
def example():
    pass


@node(
    opcode="input",
    name="输入",
    ports={"out": ["out"]},
    params={"输出维度": [2,4,8]},
)
def input_node():

    # 算输出形状
    def infer(input_shapes, params):
        output_shapes = params["输出维度"]
        return {"out": output_shapes}

    # 初始化pytorch层
    def build(input_shapes,params):
        return None

    # 计算
    def compute(inputs, layer):
        return None

    return infer, build, compute


@node(
    opcode="output",
    name="输出",
    ports={"in": ["in"]},
    params={},
)
def output_node():
    # 算输出形状
    def infer(input_shapes, params):
        return None

    # 初始化pytorch层
    def build(input_shapes,params):
        return None

    # 计算
    def compute(inputs, layer):
        return None

    return infer, build, compute
