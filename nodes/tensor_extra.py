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


@node(opcode="rounding", label="取整", ports={"input": {"x": "输入"}, "output": {"out": "输出"}}, params={"mode": {"label": "模式", "type": "enum", "value": "round", "options": {"round": "四舍五入", "floor": "向下取整", "ceil": "向上取整", "trunc": "直接去掉小数"}}}, description="把小数变成整数，可在属性中选择四舍五入、向下、向上或直接截断")
class RoundingNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        functionMap = {"round": torch.round, "floor": torch.floor, "ceil": torch.ceil, "trunc": torch.trunc}  # 模式统一映射到PyTorch取整函数
        return {"out": functionMap[self.params.get("mode", "round")](input.get("x"))}  # 执行选中的取整方式


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


category(id="base", label="基础", color="#8B92E5", icon="")  # 张量创建操作归入原有基础分类


@node(opcode="tensor_like", label="同形创建", ports={"input": {"x": "参考张量"}, "output": {"out": "新张量"}}, params={"mode": {"label": "内容", "type": "enum", "value": "zeros", "options": {"zeros": "全部为零", "ones": "全部为一", "random": "随机数"}}}, description="创建与输入形状和类型相同的新张量，可在属性中选择全零、全一或随机")
class TensorLikeNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        functionMap = {"zeros": torch.zeros_like, "ones": torch.ones_like, "random": torch.rand_like}  # 内容模式统一映射到PyTorch创建函数
        return {"out": functionMap[self.params.get("mode", "zeros")](input.get("x"))}  # 创建同形张量
