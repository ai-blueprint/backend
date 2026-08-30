"""
nodes/activation_extra.py - PyTorch 扩展激活节点

补充 torch.nn 中适合直接观察张量值域变化的激活层。所有节点保持输入形状，
只有 GLU 会沿指定维度均分张量并把该维长度减半。
"""

import torch.nn as nn  # PyTorch 激活层能力，用于构建与官方语义一致的节点

from registry import BaseNode, category, node  # 节点注册能力，将扩展激活暴露给编辑器


category(id="activation", label="激活", color="#8f67df", icon="")  # 所有激活节点统一归入激活分类


class SingleInputLayerNode(BaseNode):
    """把单输入 PyTorch 层统一映射为 x -> out。"""

    layerClass = nn.Identity  # 子类覆盖实际 PyTorch 层类型

    def build(self):
        self.layer = self.createLayer()  # 构建一次模块，PReLU 等可学习参数进入模型树

    def createLayer(self):
        return self.layerClass()  # 无参数层直接使用默认构造

    def compute(self, input):
        return {"out": self.layer(input.get("x"))}  # 单一数据流保持端口协议一致


@node(opcode="relu6", label="ReLU6", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"inplace": {"label": "原地操作", "type": "bool", "value": False}}, description="把输出限制在0到6之间")
class ReLU6Node(SingleInputLayerNode):
    def createLayer(self):
        return nn.ReLU6(inplace=self.params.get("inplace", False))  # 使用官方 ReLU6 上限语义


@node(opcode="rrelu", label="随机泄漏归正", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"lower": {"label": "最小负斜率", "type": "float", "value": 0.125, "range": [0, 1]}, "upper": {"label": "最大负斜率", "type": "float", "value": 0.333, "range": [0, 1]}}, description="训练时从区间随机选择负斜率，预览时使用均值")
class RReLUNode(SingleInputLayerNode):
    def createLayer(self):
        lower = self.params.get("lower", 0.125)  # 读取随机负斜率下界
        upper = self.params.get("upper", 0.333)  # 读取随机负斜率上界
        if lower > upper:
            raise ValueError("RReLU最小负斜率不能大于最大负斜率")  # 提前解释非法区间
        return nn.RReLU(lower=lower, upper=upper)  # 模型 eval 时自动使用区间均值


@node(opcode="selu", label="SELU", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="自归一化网络使用的缩放指数激活")
class SELUNode(SingleInputLayerNode):
    layerClass = nn.SELU  # 使用官方固定缩放系数


@node(opcode="celu", label="CELU", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"alpha": {"label": "负区曲率", "type": "float", "value": 1.0, "range": [0.001, 100]}}, description="在零点连续可导的指数线性激活")
class CELUNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.CELU(alpha=self.params.get("alpha", 1.0))  # alpha 控制负值区域曲率


@node(opcode="silu", label="SiLU", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="输入乘自身Sigmoid，也称Swish")
class SiLUNode(SingleInputLayerNode):
    layerClass = nn.SiLU  # 使用平滑自门控激活


@node(opcode="mish", label="Mish", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="输入乘Softplus的Tanh，平滑且非单调")
class MishNode(SingleInputLayerNode):
    layerClass = nn.Mish  # 使用官方 Mish 实现


@node(opcode="hardsigmoid", label="硬Sigmoid", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="分段线性近似Sigmoid，输出位于0到1")
class HardsigmoidNode(SingleInputLayerNode):
    layerClass = nn.Hardsigmoid  # 使用移动端友好的分段线性近似


@node(opcode="hardswish", label="硬Swish", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="输入乘硬Sigmoid，移动网络常用")
class HardswishNode(SingleInputLayerNode):
    layerClass = nn.Hardswish  # 使用硬件友好的 Swish 近似


@node(opcode="log_sigmoid", label="对数Sigmoid", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="数值稳定地计算Sigmoid的对数")
class LogSigmoidNode(SingleInputLayerNode):
    layerClass = nn.LogSigmoid  # 避免先 Sigmoid 再 log 的精度损失


@node(opcode="prelu", label="参数化ReLU", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"initial_slope": {"label": "初始负斜率", "type": "float", "value": 0.25, "range": [0, 10]}}, description="负斜率是可学习参数的ReLU")
class PReLUNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.PReLU(num_parameters=1, init=self.params.get("initial_slope", 0.25))  # 单参数可广播到任意输入形状


@node(opcode="hardshrink", label="硬收缩", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"threshold": {"label": "阈值", "type": "float", "value": 0.5, "range": [0, 100]}}, description="把正负阈值之间的值置零")
class HardshrinkNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.Hardshrink(lambd=self.params.get("threshold", 0.5))  # 保留阈值区间外的原值


@node(opcode="softshrink", label="软收缩", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"threshold": {"label": "阈值", "type": "float", "value": 0.5, "range": [0, 100]}}, description="把阈值内置零并平移阈值外的值")
class SoftshrinkNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.Softshrink(lambd=self.params.get("threshold", 0.5))  # 生成连续的稀疏收缩结果


@node(opcode="tanhshrink", label="Tanh收缩", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="从输入中减去Tanh响应")
class TanhshrinkNode(SingleInputLayerNode):
    layerClass = nn.Tanhshrink  # 使用 x - tanh(x) 的收缩曲线


@node(opcode="threshold", label="阈值替换", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"threshold": {"label": "阈值", "type": "float", "value": 0.0, "range": [-100, 100]}, "value": {"label": "替换值", "type": "float", "value": 0.0, "range": [-100, 100]}}, description="低于阈值的元素替换为指定值")
class ThresholdNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.Threshold(self.params.get("threshold", 0.0), self.params.get("value", 0.0))  # 显式控制门限和替换值


@node(opcode="glu", label="门控线性单元", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"dim": {"label": "均分维度", "type": "int", "value": -1, "range": [-10, 10]}}, description="一半张量作为值，另一半经过Sigmoid作为门")
class GLUNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.GLU(dim=self.params.get("dim", -1))  # 默认把最后8维均分为值和门


@node(opcode="softmin", label="Softmin", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"dim": {"label": "维度", "type": "int", "value": -1, "range": [-10, 10]}}, description="较小的值获得较大概率，总和为1")
class SoftminNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.Softmin(dim=self.params.get("dim", -1))  # 沿指定维度强调较小值


@node(opcode="log_softmax", label="对数Softmax", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"dim": {"label": "维度", "type": "int", "value": -1, "range": [-10, 10]}}, description="数值稳定地输出对数概率")
class LogSoftmaxNode(SingleInputLayerNode):
    def createLayer(self):
        return nn.LogSoftmax(dim=self.params.get("dim", -1))  # 避免 Softmax 后再取对数的精度损失
