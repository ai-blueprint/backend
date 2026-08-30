"""
nodes/common.py - 常用神经网络节点

集中提供池化、失活、嵌入、循环网络和 Transformer 层。所有序列节点统一接收
[batch, sequence, feature]，避免不同节点之间反复转置。
"""

import torch  # 张量能力，用于区分整数索引并转换连续输入
import torch.nn as nn  # PyTorch 层能力，用于构建常用可训练模块

from registry import BaseNode, category, node  # 节点注册能力，将层暴露给蓝图编辑器


category(id="transform", label="变换", color="#82cbfa", icon="")  # 常用网络节点归入变换分类


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
        "dim": {"label": "维度", "type": "enum", "value": "1d", "options": {"1d": "1D", "2d": "2D", "3d": "3D"}},
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
        dim = self.params.get("dim", "1d")  # 默认沿[batch, sequence, feature]的特征轴执行一维池化
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
    params={"mode": {"label": "模式", "type": "enum", "value": "element", "options": {"element": "按元素丢弃", "channel1d": "按一维通道丢弃", "channel2d": "按二维通道丢弃", "channel3d": "按三维通道丢弃", "alpha": "Alpha丢弃", "feature_alpha": "按特征Alpha丢弃"}}, "p": {"label": "失活率", "type": "float", "value": 0.1, "range": [0, 1]}},
    description="训练时随机隐藏部分数据，可在属性中选择按元素、通道或Alpha模式；推理时保持原值",
)
class DropoutNode(BaseNode):
    def build(self):
        layerMap = {"element": nn.Dropout, "channel1d": nn.Dropout1d, "channel2d": nn.Dropout2d, "channel3d": nn.Dropout3d, "alpha": nn.AlphaDropout, "feature_alpha": nn.FeatureAlphaDropout}  # 模式统一映射到PyTorch原生失活层
        self.dropout = layerMap[self.params.get("mode", "element")](self.params.get("p", 0.1))  # 模型train/eval状态自动控制随机行为

    def compute(self, input):
        return {"out": self.dropout(input.get("x"))}  # 透传形状并应用当前模型模式


@node(
    opcode="embedding",
    label="嵌入",
    ports={"input": {"indices": "索引"}, "output": {"out": "向量"}},
    params={
        "num_embeddings": {"label": "词表大小", "type": "int", "value": 8, "range": [1, 10000000]},
        "embedding_dim": {"label": "嵌入维度", "type": "int", "value": 8, "range": [1, 65536]},
        "padding_idx": {"label": "填充索引", "type": "int", "value": -1, "range": [-1, 10000000]},
    },
    description="把整数索引映射为可学习向量",
)
class EmbeddingNode(BaseNode):
    def build(self):
        numEmbeddings = self.params.get("num_embeddings", 8)  # 词表范围同时约束连续输入映射和填充索引
        paddingIndex = self.params.get("padding_idx", -1)  # -1 表示不保留填充向量
        if paddingIndex < -1 or paddingIndex >= numEmbeddings:
            raise ValueError("padding_idx必须为-1或位于[0, num_embeddings)范围内")  # 提前给出节点参数错误，避免底层断言难以定位
        self.embedding = nn.Embedding(numEmbeddings, self.params.get("embedding_dim", 8), padding_idx=None if paddingIndex == -1 else paddingIndex)  # 创建持久嵌入参数表

    def compute(self, input):
        indices = input.get("indices")  # 读取整数类别或默认生成的[-1, 1]连续输入
        if torch.is_floating_point(indices):
            indices = ((indices.clamp(-1, 1) + 1) * (self.embedding.num_embeddings - 1) / 2).round().long()  # 均匀覆盖完整词表，避免直接取整退化到少数索引
        return {"out": self.embedding(indices)}  # 整数输入不改值，保持标准Embedding索引语义
