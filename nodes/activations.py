"""
激活函数节点组

提供常用的激活函数节点：ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, GELU, SiLU。
"""

import torch
import torch.nn.functional as F

from decorators import category, node
from nodes import create_activation_node, create_parameterized_activation_node


# ==================== 分类定义 ====================

@category(
    id="activations",
    name="激活函数",
    color="#FFB6C1"
)
def activations_category():
    pass


# ==================== 节点定义 ====================

@node(
    id="relu",
    name="ReLU 激活",
    inputs=["x"],
    outputs=["result"],
    params={}
)
def relu_node():
    """ReLU 激活函数：max(0, x)"""
    return create_activation_node(F.relu)


@node(
    id="sigmoid",
    name="Sigmoid 激活",
    inputs=["x"],
    outputs=["result"],
    params={}
)
def sigmoid_node():
    """Sigmoid 激活函数：1 / (1 + exp(-x))"""
    return create_activation_node(torch.sigmoid)


@node(
    id="tanh",
    name="Tanh 激活",
    inputs=["x"],
    outputs=["result"],
    params={}
)
def tanh_node():
    """Tanh 激活函数"""
    return create_activation_node(torch.tanh)


@node(
    id="gelu",
    name="GELU 激活",
    inputs=["x"],
    outputs=["result"],
    params={}
)
def gelu_node():
    """GELU 激活函数（Gaussian Error Linear Unit）"""
    return create_activation_node(F.gelu)


@node(
    id="silu",
    name="SiLU/Swish 激活",
    inputs=["x"],
    outputs=["result"],
    params={}
)
def silu_node():
    """SiLU/Swish 激活函数：x * sigmoid(x)"""
    return create_activation_node(F.silu)


@node(
    id="leaky_relu",
    name="LeakyReLU 激活",
    inputs=["x"],
    outputs=["result"],
    params={"negative_slope": 0.01}
)
def leaky_relu_node():
    """LeakyReLU 激活函数"""
    return create_parameterized_activation_node(
        F.leaky_relu, 
        "negative_slope", 
        default_value=0.01
    )


@node(
    id="softmax",
    name="Softmax 激活",
    inputs=["x"],
    outputs=["result"],
    params={"dim": -1}
)
def softmax_node():
    """Softmax 激活函数"""
    return create_parameterized_activation_node(
        F.softmax,
        "dim",
        default_value=-1
    )
