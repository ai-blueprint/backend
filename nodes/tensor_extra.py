"""
nodes/tensor_extra.py - PyTorch原生张量节点组

补充逐元素、三角函数、归约、比较、分块和张量创建节点。
每个节点只对应一个原生torch运算，不包含模型专用复合结构。
"""

import torch  # 导入torch用于原生张量运算
from registry import category, node, BaseNode  # 导入分类装饰器和节点基类


category(id="math", label="运算", color="#a8c4e8", icon="")  # 逐元素运算归入原有运算分类


@node(opcode="clamp", label="限制范围", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"min": {"label": "最小值", "type": "float", "value": -1.0}, "max": {"label": "最大值", "type": "float", "value": 1.0}}, description="把每个元素限制在指定范围内")
class ClampNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.clamp(input.get("x"), min=self.params.get("min", -1.0), max=self.params.get("max", 1.0))}  # 使用原生clamp逐元素限制


@node(opcode="sign", label="取符号", ports={"input": {"x": "输入"}, "output": {"out": "符号"}}, description="把每个元素变成负一、零或正一")
class SignNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.sign(input.get("x"))}  # 使用原生sign取元素符号


@node(opcode="floor", label="向下取整", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="每个元素向负无穷取整")
class FloorNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.floor(input.get("x"))}  # 使用原生floor逐元素取整


@node(opcode="ceil", label="向上取整", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="每个元素向正无穷取整")
class CeilNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.ceil(input.get("x"))}  # 使用原生ceil逐元素取整


@node(opcode="round", label="四舍五入", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="每个元素四舍五入到最近整数")
class RoundNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.round(input.get("x"))}  # 使用原生round逐元素取整


@node(opcode="frac", label="取小数", ports={"input": {"x": "输入"}, "output": {"out": "小数"}}, description="保留每个元素的小数部分")
class FracNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.frac(input.get("x"))}  # 使用原生frac取小数部分


@node(opcode="reciprocal", label="倒数", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="计算每个元素的倒数")
class ReciprocalNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.reciprocal(input.get("x"))}  # 使用原生reciprocal计算倒数


@node(opcode="rsqrt", label="倒数平方根", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description="计算每个正元素的倒数平方根")
class RsqrtNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.rsqrt(input.get("x").abs() + 1e-6)}  # 使用原生rsqrt并避开默认输入中的负数和零


category(id="math", label="运算", color="#a8c4e8", icon="")  # 三角函数归入原有运算分类


def createUnaryTorchNode(opcode, label, function, description):
    @node(opcode=opcode, label=label, ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, description=description)
    class UnaryTorchNode(BaseNode):  # 继承BaseNode
        def compute(self, input):  # 计算方法
            return {"out": function(input.get("x"))}  # 调用对应的原生torch函数

    return UnaryTorchNode  # 返回装饰后的节点类供注册模块保留


createUnaryTorchNode("sin", "正弦", torch.sin, "计算每个元素的正弦")  # 注册sin节点
createUnaryTorchNode("cos", "余弦", torch.cos, "计算每个元素的余弦")  # 注册cos节点
createUnaryTorchNode("tan", "正切", torch.tan, "计算每个元素的正切")  # 注册tan节点
createUnaryTorchNode("atan", "反正切", torch.atan, "计算每个元素的反正切")  # 注册atan节点
createUnaryTorchNode("sinh", "双曲正弦", torch.sinh, "计算每个元素的双曲正弦")  # 注册sinh节点
createUnaryTorchNode("cosh", "双曲余弦", torch.cosh, "计算每个元素的双曲余弦")  # 注册cosh节点
createUnaryTorchNode("erf", "误差函数", torch.erf, "计算每个元素的误差函数")  # 注册erf节点


category(id="math", label="运算", color="#a8c4e8", icon="")  # 归约统计归入原有运算分类


def createReductionNode(opcode, label, function, description):
    @node(opcode=opcode, label=label, ports={"input": {"x": "输入"}, "output": {"out": "统计值"}}, params={"dim": {"label": "统计维度", "type": "int", "value": -1, "range": [-10, 10]}, "keepdim": {"label": "保留维度", "type": "bool", "value": False}}, description=description)
    class ReductionNode(BaseNode):  # 继承BaseNode
        def compute(self, input):  # 计算方法
            return {"out": function(input.get("x"), dim=self.params.get("dim", -1), keepdim=self.params.get("keepdim", False))}  # 调用对应原生归约函数

    return ReductionNode  # 返回装饰后的节点类供注册模块保留


createReductionNode("amax", "最大值归约", torch.amax, "沿指定维度取最大值")  # 注册amax节点
createReductionNode("amin", "最小值归约", torch.amin, "沿指定维度取最小值")  # 注册amin节点
createReductionNode("prod", "乘积归约", torch.prod, "沿指定维度计算元素乘积")  # 注册prod节点
createReductionNode("var", "方差", torch.var, "沿指定维度计算方差")  # 注册var节点
createReductionNode("std", "标准差", torch.std, "沿指定维度计算标准差")  # 注册std节点


@node(opcode="argmax", label="最大值位置", ports={"input": {"x": "输入"}, "output": {"out": "位置"}}, params={"dim": {"label": "查找维度", "type": "int", "value": -1, "range": [-10, 10]}}, description="返回指定维度最大值的位置")
class ArgmaxNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.argmax(input.get("x"), dim=self.params.get("dim", -1))}  # 使用原生argmax寻找位置


category(id="math", label="运算", color="#a8c4e8", icon="")  # 比较运算归入原有运算分类


def createComparisonNode(opcode, label, function, description):
    @node(opcode=opcode, label=label, ports={"input": {"x": "输入1", "y": "输入2"}, "output": {"out": "布尔结果"}}, description=description)
    class ComparisonNode(BaseNode):  # 继承BaseNode
        def compute(self, input):  # 计算方法
            return {"out": function(input.get("x"), input.get("y"))}  # 调用对应原生比较函数

    return ComparisonNode  # 返回装饰后的节点类供注册模块保留


createComparisonNode("greater", "大于", torch.gt, "逐元素判断输入1是否大于输入2")  # 注册greater节点
createComparisonNode("greater_equal", "大于等于", torch.ge, "逐元素判断输入1是否大于等于输入2")  # 注册greater_equal节点
createComparisonNode("less", "小于", torch.lt, "逐元素判断输入1是否小于输入2")  # 注册less节点
createComparisonNode("less_equal", "小于等于", torch.le, "逐元素判断输入1是否小于等于输入2")  # 注册less_equal节点
createComparisonNode("equal", "相等", torch.eq, "逐元素判断两个输入是否相等")  # 注册equal节点


category(id="shape", label="形状", color="#f1af54", icon="")  # 分块和序列操作归入原有形状分类


@node(opcode="chunk", label="均分序列", ports={"input": {"x": "输入"}, "output": {"first": "前半段", "second": "后半段"}}, params={"chunks": {"label": "分块数量", "type": "int", "value": 2, "range": [2, 2]}, "dim": {"label": "分块维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="把序列均分成两段，适合构造前后位置分支")
class ChunkNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        first, second = torch.chunk(input.get("x"), 2, dim=self.params.get("dim", 1))  # 使用原生chunk拆成两段
        return {"first": first, "second": second}  # 返回两个命名分支


category(id="transform", label="变换", color="#82cbfa", icon="")  # 序列累计操作归入原有变换分类


@node(opcode="scan", label="序列扫描", ports={"input": {"x": "序列"}, "output": {"out": "扫描结果"}}, params={"dim": {"label": "扫描维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="沿序列累计求和，每个位置都保留到当前位置的累计结果")
class ScanNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.cumsum(input.get("x"), dim=self.params.get("dim", 1))}  # 使用原生cumsum实现通用序列扫描


@node(opcode="fold", label="序列折叠", ports={"input": {"x": "序列"}, "output": {"out": "折叠结果"}}, params={"dim": {"label": "折叠维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="沿序列累计求积，把序列折叠成一个最终结果")
class FoldNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.cumprod(input.get("x"), dim=self.params.get("dim", 1))}  # 使用原生cumprod实现通用序列折叠


@node(opcode="cumulative_max", label="累计最大值", ports={"input": {"x": "序列"}, "output": {"out": "最大值"}}, params={"dim": {"label": "累计维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="沿序列保留截至当前位置的最大值")
class CumulativeMaxNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.cummax(input.get("x"), dim=self.params.get("dim", 1)).values}  # 使用原生cummax只输出累计最大值


@node(opcode="cumulative_min", label="累计最小值", ports={"input": {"x": "序列"}, "output": {"out": "最小值"}}, params={"dim": {"label": "累计维度", "type": "int", "value": 1, "range": [-10, 10]}}, description="沿序列保留截至当前位置的最小值")
class CumulativeMinNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.cummin(input.get("x"), dim=self.params.get("dim", 1)).values}  # 使用原生cummin只输出累计最小值


category(id="base", label="基础", color="#8B92E5", icon="")  # 张量创建操作归入原有基础分类


@node(opcode="zeros_like", label="同形全零", ports={"input": {"x": "参考张量"}, "output": {"out": "全零张量"}}, description="创建一个与输入形状和类型相同的全零张量")
class ZerosLikeNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.zeros_like(input.get("x"))}  # 使用原生zeros_like创建张量


@node(opcode="ones_like", label="同形全一", ports={"input": {"x": "参考张量"}, "output": {"out": "全一张量"}}, description="创建一个与输入形状和类型相同的全一张量")
class OnesLikeNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.ones_like(input.get("x"))}  # 使用原生ones_like创建张量


@node(opcode="rand_like", label="同形随机", ports={"input": {"x": "参考张量"}, "output": {"out": "随机张量"}}, description="创建一个与输入形状和类型相同的随机张量")
class RandLikeNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        return {"out": torch.rand_like(input.get("x"))}  # 使用原生rand_like创建张量
