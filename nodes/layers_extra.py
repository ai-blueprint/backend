"""
nodes/layers_extra.py - PyTorch 扩展网络层节点

补充转置卷积、上采样、正则化、距离和完整 Transformer 堆栈。默认参数统一兼容
[batch, channels/sequence, feature]=[2, 4, 8]，拖入后无需配置即可观察传播。
"""

import torch.nn as nn  # PyTorch 网络层能力，用于构建官方组件

from registry import BaseNode, category, node  # 节点注册能力，将扩展层暴露给编辑器


category(id="transform", label="变换", color="#82cbfa", icon="")  # 扩展网络节点统一归入变换分类


def parseSize(rawValue):
    if isinstance(rawValue, (list, tuple)):
        values = [int(value) for value in rawValue]  # 结构化列表逐项转换为整数
    else:
        values = [int(value.strip()) for value in str(rawValue).strip("[]()").split(",") if value.strip()]  # 兼容前端文本列表
    if not values:
        raise ValueError("尺寸参数不能为空")  # 空尺寸无法构造空间层
    return values[0] if len(values) == 1 else tuple(values)  # 单值由 PyTorch 广播，多值保持明确维度


@node(opcode="bilinear", label="双线性层", ports={"input": {"x": "输入1", "y": "输入2"}, "output": {"out": "输出"}}, params={"in1_features": {"label": "输入1特征", "type": "int", "value": 8, "range": [1, 65536]}, "in2_features": {"label": "输入2特征", "type": "int", "value": 8, "range": [1, 65536]}, "out_features": {"label": "输出特征", "type": "int", "value": 8, "range": [1, 65536]}, "bias": {"label": "偏置", "type": "bool", "value": True}}, description="让两个输入通过可学习双线性交互")
class BilinearNode(BaseNode):
    def build(self):
        self.layer = nn.Bilinear(self.params.get("in1_features", 8), self.params.get("in2_features", 8), self.params.get("out_features", 8), bias=self.params.get("bias", True))  # 最后一维分别匹配两个输入

    def compute(self, input):
        return {"out": self.layer(input.get("x"), input.get("y"))}  # 前置批量维保持不变


@node(opcode="conv_transpose", label="转置卷积", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"dim": {"label": "维度", "type": "enum", "value": "1d", "options": {"1d": "1D", "2d": "2D", "3d": "3D"}}, "in_channels": {"label": "输入通道", "type": "int", "value": 4, "range": [1, 65536]}, "out_channels": {"label": "输出通道", "type": "int", "value": 4, "range": [1, 65536]}, "kernel_size": {"label": "卷积核", "type": "list", "value": [3]}, "stride": {"label": "步幅", "type": "list", "value": [1]}, "padding": {"label": "填充", "type": "list", "value": [1]}, "output_padding": {"label": "输出填充", "type": "list", "value": [0]}, "groups": {"label": "分组数", "type": "int", "value": 1, "range": [1, 65536]}, "bias": {"label": "偏置", "type": "bool", "value": True}}, description="可学习地放大空间尺寸，也称反卷积")
class ConvTransposeNode(BaseNode):
    def build(self):
        dim = self.params.get("dim", "1d")  # 维度决定输入所需空间轴数量
        layerClass = {"1d": nn.ConvTranspose1d, "2d": nn.ConvTranspose2d, "3d": nn.ConvTranspose3d}[dim]  # 选择官方转置卷积实现
        inChannels = self.params.get("in_channels", 4)  # 默认匹配统一输入第1维通道数
        outChannels = self.params.get("out_channels", 4)  # 默认保持通道数便于连续连线
        groups = self.params.get("groups", 1)  # 分组必须均分输入输出通道
        if inChannels % groups != 0 or outChannels % groups != 0:
            raise ValueError("转置卷积分组数必须同时整除输入和输出通道数")  # 提前解释底层分组约束
        self.layer = layerClass(inChannels, outChannels, kernel_size=parseSize(self.params.get("kernel_size", [3])), stride=parseSize(self.params.get("stride", [1])), padding=parseSize(self.params.get("padding", [1])), output_padding=parseSize(self.params.get("output_padding", [0])), groups=groups, bias=self.params.get("bias", True))  # 一次构建可学习卷积核

    def compute(self, input):
        return {"out": self.layer(input.get("x"))}  # 转置卷积按配置生成空间输出


@node(opcode="upsample", label="上采样", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"scale_factor": {"label": "缩放倍数", "type": "float", "value": 2.0, "range": [0.01, 100]}, "mode": {"label": "插值模式", "type": "enum", "value": "nearest", "options": {"nearest": "最近邻", "linear": "线性"}}, "align_corners": {"label": "对齐角点", "type": "bool", "value": False}}, description="用插值放大一维空间尺寸")
class UpsampleNode(BaseNode):
    def build(self):
        mode = self.params.get("mode", "nearest")  # 三维默认输入支持最近邻和一维线性插值
        alignCorners = self.params.get("align_corners", False) if mode == "linear" else None  # 最近邻模式不接受角点参数
        self.layer = nn.Upsample(scale_factor=self.params.get("scale_factor", 2.0), mode=mode, align_corners=alignCorners)  # 默认把长度8放大到16

    def compute(self, input):
        return {"out": self.layer(input.get("x"))}  # 保持批量和通道维


@node(opcode="channel_shuffle", label="通道洗牌", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"groups": {"label": "分组数", "type": "int", "value": 2, "range": [1, 65536]}}, description="在分组之间重新排列通道")
class ChannelShuffleNode(BaseNode):
    def build(self):
        self.layer = nn.ChannelShuffle(self.params.get("groups", 2))  # 默认把4个通道分为2组交错

    def compute(self, input):
        return {"out": self.layer(input.get("x"))}  # 只重排通道，不改变形状和值


@node(opcode="pixel_rearrange", label="像素重排", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"mode": {"label": "模式", "type": "enum", "value": "up", "options": {"up": "通道变空间", "down": "空间变通道"}}, "factor": {"label": "倍数", "type": "int", "value": 2, "range": [1, 64]}}, description="重新排列通道和空间位置，可在属性中选择上采样或下采样")
class PixelRearrangeNode(BaseNode):
    def build(self):
        layerClass = nn.PixelShuffle if self.params.get("mode", "up") == "up" else nn.PixelUnshuffle  # 模式决定通道与空间的重排方向
        self.layer = layerClass(self.params.get("factor", 2))  # 两种方向共用同一个倍数参数

    def compute(self, input):
        values = input.get("x")  # 读取图像张量或统一三维教学输入
        if values.ndim == 3:
            values = values.unsqueeze(-2) if self.params.get("mode", "up") == "up" else values.unsqueeze(1)  # 按重排方向补齐四维图像格式
        return {"out": self.layer(values)}  # 重排元素，不进行数值计算


@node(opcode="reflection_pad1d", label="反射填充1D", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"padding": {"label": "左右填充", "type": "list", "value": [1, 1]}}, description="用边缘内部的镜像值填充一维信号")
class ReflectionPad1dNode(BaseNode):
    def build(self):
        self.layer = nn.ReflectionPad1d(parseSize(self.params.get("padding", [1, 1])))  # 默认在长度轴两侧各增加一个镜像值

    def compute(self, input):
        return {"out": self.layer(input.get("x"))}  # 保持批量和通道维并扩展最后一维


@node(opcode="local_response_norm", label="局部响应归一化", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"size": {"label": "通道窗口", "type": "int", "value": 3, "range": [1, 255]}, "alpha": {"label": "缩放系数", "type": "float", "value": 0.0001, "range": [0, 1]}, "beta": {"label": "指数", "type": "float", "value": 0.75, "range": [0, 10]}, "k": {"label": "偏移", "type": "float", "value": 1.0, "range": [0.000001, 100]}}, description="相邻通道之间进行竞争性归一化")
class LocalResponseNormNode(BaseNode):
    def build(self):
        self.layer = nn.LocalResponseNorm(self.params.get("size", 3), alpha=self.params.get("alpha", 0.0001), beta=self.params.get("beta", 0.75), k=self.params.get("k", 1.0))  # 创建跨通道局部归一化

    def compute(self, input):
        return {"out": self.layer(input.get("x"))}  # 保持输入形状


@node(opcode="cosine_similarity", label="余弦相似度", ports={"input": {"x": "输入1", "y": "输入2"}, "output": {"out": "相似度"}}, params={"dim": {"label": "比较维度", "type": "int", "value": -1, "range": [-10, 10]}, "eps": {"label": "防零极小值", "type": "float", "value": 0.00000001, "range": [1e-12, 1]}}, description="比较两个向量方向，相似度位于-1到1")
class CosineSimilarityNode(BaseNode):
    def build(self):
        self.layer = nn.CosineSimilarity(dim=self.params.get("dim", -1), eps=self.params.get("eps", 1e-8))  # 默认沿最后特征维比较

    def compute(self, input):
        return {"out": self.layer(input.get("x"), input.get("y"))}  # 比较同形输入并移除比较维


@node(opcode="pairwise_distance", label="成对距离", ports={"input": {"x": "输入1", "y": "输入2"}, "output": {"out": "距离"}}, params={"p": {"label": "范数阶数", "type": "float", "value": 2.0, "range": [0.001, 100]}, "eps": {"label": "防零极小值", "type": "float", "value": 0.000001, "range": [1e-12, 1]}}, description="计算两个向量之间的p范数距离")
class PairwiseDistanceNode(BaseNode):
    def build(self):
        self.layer = nn.PairwiseDistance(p=self.params.get("p", 2.0), eps=self.params.get("eps", 1e-6))  # 默认计算最后维欧氏距离

    def compute(self, input):
        return {"out": self.layer(input.get("x"), input.get("y"))}  # 前置批量维保持不变
