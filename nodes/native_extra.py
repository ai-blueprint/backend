"""
nodes/native_extra.py - 更多PyTorch原生节点

这里按PyTorch公开函数和模块补充节点，不包含任何具体模型结构。
所有节点仍然遵守同一套输入字典、输出字典和参数定义约定。
"""

import torch  # 导入PyTorch原生张量函数
import torch.nn as nn  # 导入PyTorch原生神经网络模块
from registry import category, node, BaseNode  # 导入统一注册能力


def registerUnary(categoryId, categoryLabel, color, opcode, label, function, description):
    category(id="math", label="运算", color="#a8c4e8", icon="")  # 批量原生函数归入统一运算分类

    @node(opcode=opcode, label=label, ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description=description)
    class UnaryNode(BaseNode):  # 使用统一单输入单输出节点定义
        def compute(self, input):  # 计算方法
            return {"out": function(input.get("x"))}  # 调用对应PyTorch原生函数

    return UnaryNode  # 返回节点类供模块完成注册


def registerBinary(categoryId, categoryLabel, color, opcode, label, function, description):
    category(id="math", label="运算", color="#a8c4e8", icon="")  # 批量原生函数归入统一运算分类

    @node(opcode=opcode, label=label, ports={"input": {"x": "输入1", "y": "输入2"}, "output": {"out": "输出"}}, description=description)
    class BinaryNode(BaseNode):  # 使用统一双输入单输出节点定义
        def compute(self, input):  # 计算方法
            return {"out": function(input.get("x"), input.get("y"))}  # 调用对应PyTorch原生函数

    return BinaryNode  # 返回节点类供模块完成注册


registerUnary("logarithm", "对数函数", "#6fa8dc", "log", "自然对数", lambda x: torch.log(x.abs() + 1e-6), "计算每个元素的自然对数")  # 注册log节点
registerUnary("logarithm", "对数函数", "#6fa8dc", "log10", "常用对数", lambda x: torch.log10(x.abs() + 1e-6), "计算每个元素的以10为底对数")  # 注册log10节点
registerUnary("logarithm", "对数函数", "#6fa8dc", "log2", "二进制对数", lambda x: torch.log2(x.abs() + 1e-6), "计算每个元素的以2为底对数")  # 注册log2节点
registerUnary("logarithm", "对数函数", "#6fa8dc", "log1p", "加一对数", lambda x: torch.log1p(x.abs()), "计算log(1+x)，适合小数值")  # 注册log1p节点
registerUnary("logarithm", "对数函数", "#6fa8dc", "expm1", "减一指数", torch.expm1, "计算每个元素的e的x次方减一")  # 注册expm1节点
registerUnary("logarithm", "对数函数", "#6fa8dc", "exp2", "二进制指数", torch.exp2, "计算每个元素的2的x次方")  # 注册exp2节点
registerUnary("logarithm", "对数函数", "#6fa8dc", "square", "平方", torch.square, "计算每个元素的平方")  # 注册square节点

registerUnary("rounding", "取整与符号", "#8e7cc3", "signbit", "负号判断", torch.signbit, "判断每个元素是否为负数")  # 注册signbit节点
registerUnary("rounding", "取整与符号", "#8e7cc3", "trunc", "截断取整", torch.trunc, "去掉每个元素的小数部分")  # 注册trunc节点

registerBinary("binary", "二元运算", "#76a5af", "maximum", "逐元素最大", torch.maximum, "逐元素选择两个输入中较大的值")  # 注册maximum节点
registerBinary("binary", "二元运算", "#76a5af", "minimum", "逐元素最小", torch.minimum, "逐元素选择两个输入中较小的值")  # 注册minimum节点
registerBinary("binary", "二元运算", "#76a5af", "remainder", "取余", torch.remainder, "计算两个输入逐元素相除后的余数")  # 注册remainder节点
registerBinary("binary", "二元运算", "#76a5af", "fmod", "浮点取余", torch.fmod, "计算两个输入逐元素相除后的浮点余数")  # 注册fmod节点
registerBinary("binary", "二元运算", "#76a5af", "hypot", "直角距离", torch.hypot, "计算两个输入作为直角边时的斜边长度")  # 注册hypot节点


category(id="shape", label="形状", color="#f1af54", icon="")  # 索引和重排归入原有形状分类


@node(opcode="roll", label="循环移位", ports={"input": {"x": "输入"}, "output": {"out": "移位结果"}}, params={"shifts": {"label": "移位步数", "type": "int", "value": 1, "range": [-65536, 65536]}, "dim": {"label": "移位维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="沿指定维度循环移动元素")
class RollNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.roll(input.get("x"), shifts=self.params.get("shifts", 1), dims=self.params.get("dim", 1))}  # 使用原生roll循环移位


@node(opcode="flip", label="反转维度", ports={"input": {"x": "输入"}, "output": {"out": "反转结果"}}, params={"dim": {"label": "反转维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="沿指定维度反转元素顺序")
class FlipNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.flip(input.get("x"), dims=[self.params.get("dim", 1)])}  # 使用原生flip反转维度


@node(opcode="repeat_interleave", label="重复元素", ports={"input": {"x": "输入"}, "output": {"out": "重复结果"}}, params={"repeats": {"label": "重复次数", "type": "int", "value": 2, "range": [1, 64]}, "dim": {"label": "重复维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="沿指定维度重复每个元素")
class RepeatInterleaveNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.repeat_interleave(input.get("x"), self.params.get("repeats", 2), dim=self.params.get("dim", 1))}  # 使用原生repeat_interleave扩展序列


category(id="transform", label="变换", color="#82cbfa", icon="")  # 池化归入原有变换分类


class PoolingNode(BaseNode):  # 统一池化节点的基础实现
    poolFunction = None  # 由具体节点指定PyTorch池化函数

    def compute(self, input):  # 计算方法
        return {"out": self.poolFunction(input.get("x"), kernel_size=self.params.get("kernel_size", 2), stride=self.params.get("stride", 2))}  # 调用统一池化函数


@node(opcode="max_pool1d", label="一维最大池化", ports={"input": {"x": "序列"}, "output": {"out": "池化结果"}}, params={"kernel_size": {"label": "窗口大小", "type": "int", "value": 2, "range": [1, 64]}, "stride": {"label": "步幅", "type": "int", "value": 2, "range": [1, 64]}}, description="沿一维序列窗口取最大值")
class MaxPool1dNode(PoolingNode):  # 继承统一池化节点
    poolFunction = staticmethod(torch.nn.functional.max_pool1d)  # 使用PyTorch原生一维最大池化


@node(opcode="avg_pool1d", label="一维平均池化", ports={"input": {"x": "序列"}, "output": {"out": "池化结果"}}, params={"kernel_size": {"label": "窗口大小", "type": "int", "value": 2, "range": [1, 64]}, "stride": {"label": "步幅", "type": "int", "value": 2, "range": [1, 64]}}, description="沿一维序列窗口取平均值")
class AvgPool1dNode(PoolingNode):  # 继承统一池化节点
    poolFunction = staticmethod(torch.nn.functional.avg_pool1d)  # 使用PyTorch原生一维平均池化


@node(opcode="adaptive_avg_pool1d", label="自适应一维平均池化", ports={"input": {"x": "序列"}, "output": {"out": "池化结果"}}, params={"output_size": {"label": "输出长度", "type": "int", "value": 4, "range": [1, 256]}}, description="把一维序列池化到指定长度")
class AdaptiveAvgPool1dNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": nn.functional.adaptive_avg_pool1d(input.get("x"), self.params.get("output_size", 4))}  # 使用PyTorch原生自适应平均池化


category(id="transform", label="变换", color="#82cbfa", icon="")  # 随机正则化归入原有变换分类


@node(opcode="dropout2d", label="二维随机失活", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"p": {"label": "失活概率", "type": "float", "value": 0.5, "range": [0, 1]}}, description="按通道随机丢弃特征，适合图像和特征图")
class Dropout2dNode(BaseNode):  # 继承BaseNode
    def build(self):  # 构建随机失活模块
        self.layer = nn.Dropout2d(self.params.get("p", 0.5))  # 创建PyTorch原生二维失活

    def compute(self, input):  # 计算方法
        return {"out": self.layer(input.get("x"))}  # 执行二维随机失活


@node(opcode="dropout3d", label="三维随机失活", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"p": {"label": "失活概率", "type": "float", "value": 0.5, "range": [0, 1]}}, description="按通道随机丢弃三维特征")
class Dropout3dNode(BaseNode):  # 继承BaseNode
    def build(self):  # 构建随机失活模块
        self.layer = nn.Dropout3d(self.params.get("p", 0.5))  # 创建PyTorch原生三维失活

    def compute(self, input):  # 计算方法
        return {"out": self.layer(input.get("x"))}  # 执行三维随机失活
