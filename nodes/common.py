"""
nodes/common.py - 常用神经网络节点

集中提供池化、失活、嵌入、循环网络和 Transformer 层。所有序列节点统一接收
[batch, sequence, feature]，避免不同节点之间反复转置。
"""

import torch.nn as nn  # PyTorch 层能力，用于构建常用可训练模块

from registry import BaseNode, category, node  # 节点注册能力，将层暴露给蓝图编辑器


category(id="common", label="常用层", color="#57b8a6", icon="")  # 注册常用模型组件分类


def parseSize(rawValue):
    if isinstance(rawValue, (list, tuple)):
        return tuple(int(value) for value in rawValue)  # 列表尺寸直接转换为 PyTorch 接受的元组
    values = [int(value.strip()) for value in str(rawValue).split(",") if value.strip()]  # 字符串兼容前端逗号输入
    return values[0] if len(values) == 1 else tuple(values)  # 单维池化保留整数，多维池化使用元组


@node(
    opcode="pooling",
    label="池化",
    ports={"input": {"x": "输入"}, "output": {"out": "输出"}},
    params={
        "mode": {"label": "模式", "type": "enum", "value": "max", "options": {"max": "最大池化", "avg": "平均池化", "adaptive_max": "自适应最大池化", "adaptive_avg": "自适应平均池化"}},
        "dim": {"label": "维度", "type": "enum", "value": "2d", "options": {"1d": "1D", "2d": "2D", "3d": "3D"}},
        "kernel_size": {"label": "池化核", "type": "list", "value": "2"},
        "stride": {"label": "步幅", "type": "list", "value": "2"},
        "padding": {"label": "填充", "type": "list", "value": "0"},
        "output_size": {"label": "输出尺寸", "type": "list", "value": "1"},
    },
    description="支持最大、平均和自适应池化的统一节点",
)
class PoolingNode(BaseNode):
    # --- 构建所选池化层 ---
    def build(self):
        mode = self.params.get("mode", "max")  # 模式决定普通池化或自适应池化
        dim = self.params.get("dim", "2d")  # 维度决定输入空间轴数量
        if mode.startswith("adaptive"):
            layerMap = {"adaptive_max": {"1d": nn.AdaptiveMaxPool1d, "2d": nn.AdaptiveMaxPool2d, "3d": nn.AdaptiveMaxPool3d}, "adaptive_avg": {"1d": nn.AdaptiveAvgPool1d, "2d": nn.AdaptiveAvgPool2d, "3d": nn.AdaptiveAvgPool3d}}
            self.pool = layerMap[mode][dim](parseSize(self.params.get("output_size", "1")))  # 自适应模式只需要目标尺寸
            return

        layerMap = {"max": {"1d": nn.MaxPool1d, "2d": nn.MaxPool2d, "3d": nn.MaxPool3d}, "avg": {"1d": nn.AvgPool1d, "2d": nn.AvgPool2d, "3d": nn.AvgPool3d}}
        self.pool = layerMap[mode][dim](kernel_size=parseSize(self.params.get("kernel_size", "2")), stride=parseSize(self.params.get("stride", "2")), padding=parseSize(self.params.get("padding", "0")))  # 普通模式明确核、步幅和填充

    # --- 执行池化 ---
    def compute(self, input):
        return {"out": self.pool(input.get("x"))}  # 池化仅改变空间尺寸并保留命名端口


@node(
    opcode="dropout",
    label="随机失活",
    ports={"input": {"x": "输入"}, "output": {"out": "输出"}},
    params={"p": {"label": "失活率", "type": "float", "value": 0.5, "range": [0, 1]}},
    description="训练时随机将部分元素置零，推理时保持原值",
)
class DropoutNode(BaseNode):
    def build(self):
        self.dropout = nn.Dropout(self.params.get("p", 0.5))  # 模型 train/eval 状态自动控制随机行为

    def compute(self, input):
        return {"out": self.dropout(input.get("x"))}  # 透传形状并应用当前模型模式


@node(
    opcode="embedding",
    label="嵌入",
    ports={"input": {"indices": "索引"}, "output": {"out": "向量"}},
    params={
        "num_embeddings": {"label": "词表大小", "type": "int", "value": 1000, "range": [1, 10000000]},
        "embedding_dim": {"label": "嵌入维度", "type": "int", "value": 64, "range": [1, 65536]},
        "padding_idx": {"label": "填充索引", "type": "int", "value": -1, "range": [-1, 10000000]},
    },
    description="把整数索引映射为可学习向量",
)
class EmbeddingNode(BaseNode):
    def build(self):
        paddingIndex = self.params.get("padding_idx", -1)  # -1 表示不保留填充向量
        self.embedding = nn.Embedding(self.params.get("num_embeddings", 1000), self.params.get("embedding_dim", 64), padding_idx=None if paddingIndex < 0 else paddingIndex)  # 创建持久嵌入参数表

    def compute(self, input):
        return {"out": self.embedding(input.get("indices").long())}  # 索引统一转换为嵌入层要求的长整数


class RecurrentNode(BaseNode):
    recurrentClass = nn.RNN  # 子类覆盖具体循环结构，公共路由保持一致

    def build(self):
        self.recurrent = self.recurrentClass(input_size=self.params.get("input_size", 64), hidden_size=self.params.get("hidden_size", 64), num_layers=self.params.get("num_layers", 1), bias=self.params.get("bias", True), batch_first=True, dropout=self.params.get("dropout", 0.0) if self.params.get("num_layers", 1) > 1 else 0.0, bidirectional=self.params.get("bidirectional", False))  # 所有循环节点统一 batch_first

    def compute(self, input):
        state = input.get("state")  # 可选状态支持连续序列分段执行
        output, finalState = self.recurrent(input.get("x"), state) if state is not None else self.recurrent(input.get("x"))  # 无状态时让 PyTorch 自动创建零状态
        if isinstance(finalState, tuple):
            return {"out": output, "hidden": finalState[0], "cell": finalState[1]}  # LSTM 分别暴露隐藏状态和记忆状态
        return {"out": output, "hidden": finalState}  # RNN 和 GRU 只返回隐藏状态


recurrentParams = {
    "input_size": {"label": "输入维度", "type": "int", "value": 64, "range": [1, 65536]},
    "hidden_size": {"label": "隐藏维度", "type": "int", "value": 64, "range": [1, 65536]},
    "num_layers": {"label": "层数", "type": "int", "value": 1, "range": [1, 128]},
    "bias": {"label": "偏置", "type": "bool", "value": True},
    "dropout": {"label": "层间失活率", "type": "float", "value": 0.0, "range": [0, 1]},
    "bidirectional": {"label": "双向", "type": "bool", "value": False},
}  # 三类循环节点共享同一组稳定参数


@node(opcode="rnn", label="RNN", ports={"input": {"x": "输入", "state": "初始状态"}, "output": {"out": "序列", "hidden": "隐藏状态"}}, params=recurrentParams, description="批优先的基础循环神经网络")
class RNNNode(RecurrentNode):
    recurrentClass = nn.RNN  # 使用基础 tanh RNN 实现公共循环协议


@node(opcode="lstm", label="LSTM", ports={"input": {"x": "输入", "state": "初始状态"}, "output": {"out": "序列", "hidden": "隐藏状态", "cell": "记忆状态"}}, params=recurrentParams, description="批优先的长短期记忆网络")
class LSTMNode(RecurrentNode):
    recurrentClass = nn.LSTM  # LSTM 额外返回记忆状态端口


@node(opcode="gru", label="GRU", ports={"input": {"x": "输入", "state": "初始状态"}, "output": {"out": "序列", "hidden": "隐藏状态"}}, params=recurrentParams, description="批优先的门控循环单元")
class GRUNode(RecurrentNode):
    recurrentClass = nn.GRU  # GRU 使用单隐藏状态公共协议


transformerParams = {
    "d_model": {"label": "特征维度", "type": "int", "value": 64, "range": [1, 65536]},
    "nhead": {"label": "注意力头数", "type": "int", "value": 8, "range": [1, 256]},
    "dim_feedforward": {"label": "前馈维度", "type": "int", "value": 256, "range": [1, 262144]},
    "dropout": {"label": "失活率", "type": "float", "value": 0.1, "range": [0, 1]},
    "activation": {"label": "激活函数", "type": "enum", "value": "relu", "options": {"relu": "ReLU", "gelu": "GELU"}},
    "norm_first": {"label": "先归一化", "type": "bool", "value": False},
}  # 编码器和解码器层共享标准 Transformer 参数


@node(opcode="transformer_encoder_layer", label="Transformer编码层", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params=transformerParams, description="批优先的标准 Transformer 编码器层")
class TransformerEncoderLayerNode(BaseNode):
    def build(self):
        self.layer = nn.TransformerEncoderLayer(d_model=self.params.get("d_model", 64), nhead=self.params.get("nhead", 8), dim_feedforward=self.params.get("dim_feedforward", 256), dropout=self.params.get("dropout", 0.1), activation=self.params.get("activation", "relu"), batch_first=True, norm_first=self.params.get("norm_first", False))  # 创建可独立堆叠的编码器层

    def compute(self, input):
        return {"out": self.layer(input.get("x"))}  # 编码层保持批和序列维度


@node(opcode="transformer_decoder_layer", label="Transformer解码层", ports={"input": {"x": "目标序列", "memory": "编码记忆"}, "output": {"out": "输出"}}, params=transformerParams, description="批优先的标准 Transformer 解码器层")
class TransformerDecoderLayerNode(BaseNode):
    def build(self):
        self.layer = nn.TransformerDecoderLayer(d_model=self.params.get("d_model", 64), nhead=self.params.get("nhead", 8), dim_feedforward=self.params.get("dim_feedforward", 256), dropout=self.params.get("dropout", 0.1), activation=self.params.get("activation", "relu"), batch_first=True, norm_first=self.params.get("norm_first", False))  # 创建可独立堆叠的解码器层

    def compute(self, input):
        return {"out": self.layer(input.get("x"), input.get("memory"))}  # 目标序列通过交叉注意力读取编码记忆
