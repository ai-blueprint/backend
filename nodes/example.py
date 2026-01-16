# ================= 实际节点组编写示例 =================
import sys
import os

# 添加backend目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from decorators import category, node
import torch


@category(id="example", name="示例", color="#82CBFA", icon="base64…")
def example():
    pass


@node(
    opcode="linear",
    name="全连接层",
    ports={"in": ["x"], "out": ["y"]},
    params={"输出特征数": 128, "bias": True},
)
def linear_node():

    # 算输出形状
    def infer(input_shapes, params):
        output_shapes = input_shapes["x"]
        output_shapes[-1] = params["输出特征数"]  # 只需要修改最后一个维度
        return {"y": output_shapes}

    # 初始化pytorch层
    def build(input_shapes,params):
        in_feat = input_shapes["x"][-1]  # 从输入推断
        out_feat = params["输出特征数"]
        layer=torch.nn.Linear(in_feat, out_feat, bias=params["bias"])
        return layer

    # 计算
    def compute(inputs, layer):
        outputs = layer(inputs)
        return outputs

    return infer, build, compute


@node(
    opcode="add",
    name="+",
    ports={"in": ["x1", "x2"], "out": ["y"]},
    params={},
)
def add_node():
    # 算输出形状
    def infer(input_shapes, params):
        # 这里有多个input_shapes
        output_shapes = input_shapes.copy()
        return output_shapes

    # 初始化pytorch层
    def build(params):
        return None

    # 计算
    def compute(inputs, layer):
        outputs = inputs["x1"] + inputs["x2"]
        return outputs

    return infer, build, compute
