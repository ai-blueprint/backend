"""
神经网络层节点组

提供常用的神经网络层：Linear, Conv2d, BatchNorm, Dropout, Pool, Flatten, LSTM, Embedding。
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List

from decorators import category, node
from nodes import (
    create_module_node, 
    infer_linear_shape, 
    infer_conv2d_shape, 
    infer_pool2d_shape,
    shape_unchanged
)


# ==================== 分类定义 ====================

@category(
    id="layers",
    name="神经网络层",
    color="#82CBFA",
    icon="base64…"
)
def layers_category():
    pass


# ==================== 全连接层 ====================

@node(
    opcode="linear",
    name="全连接层 (Linear)",
    ports={"in": ["x"], "out": ["out"]},
    params={"in_features": 128, "out_features": 64, "bias": True}
)
def linear_layer():
    """全连接层"""
    return create_module_node(
        nn.Linear,
        build_args=lambda p: (p["in_features"], p["out_features"]),
        build_kwargs=lambda p: {"bias": p.get("bias", True)},
        infer_shape=infer_linear_shape
    )


# ==================== 卷积层 ====================

@node(
    opcode="conv2d",
    name="卷积层 (Conv2d)",
    ports={"in": ["x"], "out": ["out"]},
    params={
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
    }
)
def conv2d_layer():
    """2D卷积层"""
    return create_module_node(
        nn.Conv2d,
        build_kwargs=lambda p: {
            "in_channels": p["in_channels"],
            "out_channels": p["out_channels"],
            "kernel_size": p["kernel_size"],
            "stride": p["stride"],
            "padding": p["padding"]
        },
        infer_shape=infer_conv2d_shape
    )


# ==================== 归一化层 ====================

@node(
    opcode="batchnorm2d",
    name="批归一化 (BatchNorm2d)",
    ports={"in": ["x"], "out": ["out"]},
    params={"num_features": 64}
)
def batchnorm2d_layer():
    """2D批归一化层"""
    return create_module_node(
        nn.BatchNorm2d,
        build_args=lambda p: (p["num_features"],),
        infer_shape=shape_unchanged
    )


@node(
    opcode="batchnorm1d",
    name="批归一化 (BatchNorm1d)",
    ports={"in": ["x"], "out": ["out"]},
    params={"num_features": 64}
)
def batchnorm1d_layer():
    """1D批归一化层"""
    return create_module_node(
        nn.BatchNorm1d,
        build_args=lambda p: (p["num_features"],),
        infer_shape=shape_unchanged
    )


@node(
    opcode="layernorm",
    name="层归一化 (LayerNorm)",
    ports={"in": ["x"], "out": ["out"]},
    params={"normalized_shape": 64}
)
def layernorm_layer():
    """层归一化"""
    return create_module_node(
        nn.LayerNorm,
        build_args=lambda p: (p["normalized_shape"],),
        infer_shape=shape_unchanged
    )


# ==================== 正则化层 ====================

@node(
    opcode="dropout",
    name="Dropout",
    ports={"in": ["x"], "out": ["out"]},
    params={"p": 0.5}
)
def dropout_layer():
    """Dropout层"""
    return create_module_node(
        nn.Dropout,
        build_kwargs=lambda p: {"p": p.get("p", 0.5)},
        infer_shape=shape_unchanged
    )


# ==================== 池化层 ====================

@node(
    opcode="maxpool2d",
    name="最大池化 (MaxPool2d)",
    ports={"in": ["x"], "out": ["out"]},
    params={"kernel_size": 2, "stride": 2, "padding": 0}
)
def maxpool2d_layer():
    """2D最大池化层"""
    return create_module_node(
        nn.MaxPool2d,
        build_kwargs=lambda p: {
            "kernel_size": p["kernel_size"],
            "stride": p["stride"],
            "padding": p["padding"]
        },
        infer_shape=infer_pool2d_shape
    )


@node(
    opcode="avgpool2d",
    name="平均池化 (AvgPool2d)",
    ports={"in": ["x"], "out": ["out"]},
    params={"kernel_size": 2, "stride": 2, "padding": 0}
)
def avgpool2d_layer():
    """2D平均池化层"""
    return create_module_node(
        nn.AvgPool2d,
        build_kwargs=lambda p: {
            "kernel_size": p["kernel_size"],
            "stride": p["stride"],
            "padding": p["padding"]
        },
        infer_shape=infer_pool2d_shape
    )


@node(
    opcode="adaptiveavgpool2d",
    name="自适应平均池化",
    ports={"in": ["x"], "out": ["out"]},
    params={"output_size": 1}
)
def adaptive_avgpool2d_layer():
    """自适应2D平均池化层"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        shape = list(input_shapes.get("x", [1, 64, 7, 7]))
        output_size = params.get("output_size", 1)
        return {"out": [shape[0], shape[1], output_size, output_size]}
    
    def build(input_shapes, params):
        output_size = params.get("output_size", 1)
        return nn.AdaptiveAvgPool2d(output_size)
    
    def compute(x, layer):
        return layer(x)
    
    return infer, build, compute


# ==================== 形状变换层 ====================

@node(
    opcode="flatten",
    name="展平 (Flatten)",
    ports={"in": ["x"], "out": ["out"]},
    params={"start_dim": 1, "end_dim": -1}
)
def flatten_layer():
    """展平层"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        shape = list(input_shapes.get("x", [1, 64, 8, 8]))
        start = params.get("start_dim", 1)
        batch_size = shape[0]
        flat_size = 1
        for s in shape[start:]:
            flat_size *= s
        return {"out": [batch_size, flat_size]}
    
    return create_module_node(
        nn.Flatten,
        build_kwargs=lambda p: {
            "start_dim": p.get("start_dim", 1),
            "end_dim": p.get("end_dim", -1)
        },
        infer_shape=infer
    )


# ==================== 序列层 ====================

@node(
    opcode="lstm",
    name="LSTM",
    ports={"in": ["x"], "out": ["out", "hidden"]},
    params={
        "input_size": 128,
        "hidden_size": 256,
        "num_layers": 1,
        "batch_first": True
    }
)
def lstm_layer():
    """LSTM层"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        shape = list(input_shapes.get("x", [1, 10, 128]))
        hidden = params["hidden_size"]
        return {
            "out": [shape[0], shape[1], hidden],
            "hidden": [shape[0], hidden]
        }
    
    def build(input_shapes, params):
        return nn.LSTM(
            input_size=params["input_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            batch_first=params.get("batch_first", True)
        )
    
    def compute(x, layer):
        out, (hn, cn) = layer(x)
        return {"out": out, "hidden": hn[-1]}
    
    return infer, build, compute


@node(
    opcode="gru",
    name="GRU",
    ports={"in": ["x"], "out": ["out", "hidden"]},
    params={
        "input_size": 128,
        "hidden_size": 256,
        "num_layers": 1,
        "batch_first": True
    }
)
def gru_layer():
    """GRU层"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        shape = list(input_shapes.get("x", [1, 10, 128]))
        hidden = params["hidden_size"]
        return {
            "out": [shape[0], shape[1], hidden],
            "hidden": [shape[0], hidden]
        }
    
    def build(input_shapes, params):
        return nn.GRU(
            input_size=params["input_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            batch_first=params.get("batch_first", True)
        )
    
    def compute(x, layer):
        out, hn = layer(x)
        return {"out": out, "hidden": hn[-1]}
    
    return infer, build, compute


# ==================== 嵌入层 ====================

@node(
    opcode="embedding",
    name="嵌入层 (Embedding)",
    ports={"in": ["x"], "out": ["out"]},
    params={"num_embeddings": 10000, "embedding_dim": 128}
)
def embedding_layer():
    """词嵌入层"""
    def infer(input_shapes: Dict[str, List[int]], params: Dict) -> Dict[str, List[int]]:
        shape = list(input_shapes.get("x", [1, 10]))
        shape.append(params["embedding_dim"])
        return {"out": shape}
    
    def build(input_shapes, params):
        return nn.Embedding(
            num_embeddings=params["num_embeddings"],
            embedding_dim=params["embedding_dim"]
        )
    
    def compute(x, layer):
        return layer(x.long())
    
    return infer, build, compute
