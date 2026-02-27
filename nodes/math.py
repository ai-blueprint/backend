"""
nodes/math.py - 运算节点组

提供张量运算相关节点：加减乘除、矩阵乘法、爱因斯坦求和、插值、点积、幂运算、范数、指数、开方、求和、绝对值、相反数、均值
"""

import torch  # 导入torch用于张量操作
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册运算分类
    id="math",  # 分类唯一标识
    label="运算",  # 分类显示名称
    color="#feae8a",  # 分类颜色，红色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册add节点
    opcode="add",  # 节点操作码
    label="加法",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "", "y": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="两个张量逐元素相加",  # 节点描述
)
class AddNode(BaseNode):  # 继承BaseNode
    """
    逐元素加法节点
    用法：out = x + y，支持广播
    调用示例：
        输入 x: shape=[batch, features], y: shape=[batch, features]
        输出 out: shape=[batch, features]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入1
        y = input.get("y")  # 获取输入2
        out = torch.add(x, y)  # 逐元素加法
        return {"out": out}  # 返回输出


@node(  # 注册sub节点
    opcode="sub",  # 节点操作码
    label="减法",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "", "y": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="两个张量逐元素相减",  # 节点描述
)
class SubNode(BaseNode):  # 继承BaseNode
    """
    逐元素减法节点
    用法：out = x - y，支持广播
    调用示例：
        输入 x: shape=[batch, features], y: shape=[batch, features]
        输出 out: shape=[batch, features]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入1
        y = input.get("y")  # 获取输入2
        out = torch.sub(x, y)  # 逐元素减法
        return {"out": out}  # 返回输出


@node(  # 注册mul节点
    opcode="mul",  # 节点操作码
    label="乘法",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "", "y": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="两个张量逐元素相乘",  # 节点描述
)
class MulNode(BaseNode):  # 继承BaseNode
    """
    逐元素乘法节点
    用法：out = x * y，支持广播
    调用示例：
        输入 x: shape=[batch, features], y: shape=[batch, features]
        输出 out: shape=[batch, features]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入1
        y = input.get("y")  # 获取输入2
        out = torch.mul(x, y)  # 逐元素乘法
        return {"out": out}  # 返回输出


@node(  # 注册div节点
    opcode="div",  # 节点操作码
    label="除法",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "被除数", "y": "除数"},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="两个张量逐元素相除",  # 节点描述
)
class DivNode(BaseNode):  # 继承BaseNode
    """
    逐元素除法节点
    用法：out = x / y，支持广播
    调用示例：
        输入 x: shape=[batch, features], y: shape=[batch, features]
        输出 out: shape=[batch, features]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取被除数
        y = input.get("y")  # 获取除数
        out = torch.div(x, y)  # 逐元素除法
        return {"out": out}  # 返回输出


@node(  # 注册matmul节点
    opcode="matmul",  # 节点操作码
    label="矩阵乘",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "", "y": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="矩阵相乘，支持批量",  # 节点描述
)
class MatmulNode(BaseNode):  # 继承BaseNode
    """
    矩阵乘法节点
    用法：out = x @ y，支持批量矩阵乘法
    调用示例：
        输入 x: shape=[batch, m, k], y: shape=[batch, k, n]
        输出 out: shape=[batch, m, n]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入1
        y = input.get("y")  # 获取输入2
        out = torch.matmul(x, y)  # 矩阵乘法
        return {"out": out}  # 返回输出


@node(  # 注册bmm节点
    opcode="bmm",  # 节点操作码
    label="批量矩阵乘法",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "", "y": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="多组矩阵同时相乘",  # 节点描述
)
class BmmNode(BaseNode):  # 继承BaseNode
    """
    批量矩阵乘法节点
    用法：out = batch_matmul(x, y)，要求x和y都是3D张量
    调用示例：
        输入 x: shape=[batch, m, k], y: shape=[batch, k, n]
        输出 out: shape=[batch, m, n]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入1
        y = input.get("y")  # 获取输入2
        out = torch.bmm(x, y)  # 批量矩阵乘法
        return {"out": out}  # 返回输出


@node(  # 注册einsum节点
    opcode="einsum",  # 节点操作码
    label="爱因斯坦求和",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "", "y": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "equation": {
            "label": "公式",
            "type": "str",
            "value": "ij,jk->ik",
        },  # 爱因斯坦求和公式
    },
    description="用公式描述任意张量运算",  # 节点描述
)
class EinsumNode(BaseNode):  # 继承BaseNode
    """
    爱因斯坦求和节点
    用法：用字符串公式描述张量运算 out = einsum(equation, x, y)
    调用示例：
        参数 equation="ij,jk->ik" 表示矩阵乘法
        参数 equation="bhid,bhjd->bhij" 表示注意力分数计算
        输入 x, y: 形状由公式决定
        输出 out: 形状由公式决定
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入1
        y = input.get("y")  # 获取输入2
        equation = self.params.get("equation", "ij,jk->ik")  # 获取公式字符串
        out = torch.einsum(equation, x, y)  # 爱因斯坦求和
        return {"out": out}  # 返回输出


@node(  # 注册lerp节点
    opcode="lerp",  # 节点操作码
    label="插值",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "起点", "y": "终点", "w": "权重"},  # 三个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="在两个值之间按权重过渡",  # 节点描述
)
class LerpNode(BaseNode):  # 继承BaseNode
    """
    线性插值节点
    用法：out = x + w * (y - x)，w=0时为x，w=1时为y
    调用示例：
        输入 x: shape=[任意], y: shape=[与x相同], w: shape=[标量或与x相同]
        输出 out: shape=[与x相同]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取起点
        y = input.get("y")  # 获取终点
        w = input.get("w")  # 获取权重
        out = torch.lerp(x, y, w)  # 线性插值
        return {"out": out}  # 返回输出


@node(  # 注册dot节点
    opcode="dot",  # 节点操作码
    label="点积",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "", "y": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="两个向量逐元素乘再求和",  # 节点描述
)
class DotNode(BaseNode):  # 继承BaseNode
    """
    向量点积节点
    用法：out = sum(x * y)，要求x和y都是1D张量
    调用示例：
        输入 x: shape=[n], y: shape=[n]
        输出 out: shape=[] 标量
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入1
        y = input.get("y")  # 获取输入2
        out = torch.dot(x, y)  # 向量点积
        return {"out": out}  # 返回输出


@node(  # 注册pow节点
    opcode="pow",  # 节点操作码
    label="幂运算",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "exponent": {
            "label": "指数",
            "type": "float",
            "value": 2.0,
            "range": [-10, 10],
        },  # 幂次
    },
    description="对每个元素做n次方",  # 节点描述
)
class PowNode(BaseNode):  # 继承BaseNode
    """
    幂运算节点
    用法：out = x ^ exponent
    调用示例：
        输入 x: shape=[任意形状]
        参数 exponent=2.0 表示平方
        输出 out: shape=[与输入形状相同]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取底数
        exponent = self.params.get("exponent", 2.0)  # 获取指数
        out = torch.pow(x, exponent)  # 幂运算
        return {"out": out}  # 返回输出


@node(  # 注册norm节点
    opcode="norm",  # 节点操作码
    label="范数",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "p": {
            "label": "范数阶数",
            "type": "float",
            "value": 2.0,
            "range": [0, 10],
        },  # 范数的阶数
        "dim": {
            "label": "维度",
            "type": "int",
            "value": -1,
            "range": [-10, 10],
        },  # 沿哪个维度计算
        "keepdim": {
            "label": "保持维度",
            "type": "bool",
            "value": False,
        },  # 是否保持维度
    },
    description="计算向量的长度大小",  # 节点描述
)
class NormNode(BaseNode):  # 继承BaseNode
    """
    范数计算节点
    用法：out = ||x||_p，沿指定维度计算p范数
    调用示例：
        输入 x: shape=[batch, features]
        参数 p=2.0 表示L2范数, dim=-1 沿最后维度
        输出 out: shape=[batch] 或 [batch, 1]（keepdim=True时）
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        p = self.params.get("p", 2.0)  # 获取范数阶数
        dim = self.params.get("dim", -1)  # 获取维度
        keepdim = self.params.get("keepdim", False)  # 获取是否保持维度
        out = torch.norm(x, p=p, dim=dim, keepdim=keepdim)  # 计算范数
        return {"out": out}  # 返回输出


@node(  # 注册exp节点
    opcode="exp",  # 节点操作码
    label="自然指数",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="求e的x次方",  # 节点描述
)
class ExpNode(BaseNode):  # 继承BaseNode
    """
    自然指数运算节点
    用法：out = e^x
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入形状相同]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = torch.exp(x)  # e的x次方
        return {"out": out}  # 返回输出


@node(  # 注册sqrt节点
    opcode="sqrt",  # 节点操作码
    label="开方",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="对每个元素开平方根",  # 节点描述
)
class SqrtNode(BaseNode):  # 继承BaseNode
    """
    开平方节点
    用法：out = sqrt(x)
    调用示例：
        输入 x: shape=[任意形状]，值需非负
        输出 out: shape=[与输入形状相同]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = torch.sqrt(x)  # 开平方
        return {"out": out}  # 返回输出


@node(  # 注册sum节点
    opcode="sum",  # 节点操作码
    label="求和",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {
            "label": "维度",
            "type": "int",
            "value": -1,
            "range": [-10, 10],
        },  # 沿哪个维度求和
        "keepdim": {
            "label": "保持维度",
            "type": "bool",
            "value": False,
        },  # 是否保持维度
    },
    description="沿某个维度把元素加起来",  # 节点描述
)
class SumNode(BaseNode):  # 继承BaseNode
    """
    求和节点
    用法：out = sum(x, dim)，沿指定维度求和
    调用示例：
        输入 x: shape=[batch, seq_len, features]
        参数 dim=-1 沿最后维度求和
        输出 out: shape=[batch, seq_len] 或 [batch, seq_len, 1]（keepdim=True时）
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        dim = self.params.get("dim", -1)  # 获取维度
        keepdim = self.params.get("keepdim", False)  # 获取是否保持维度
        out = torch.sum(x, dim=dim, keepdim=keepdim)  # 沿维度求和
        return {"out": out}  # 返回输出


@node(  # 注册abs节点
    opcode="abs",  # 节点操作码
    label="绝对值",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="对每个元素取绝对值",  # 节点描述
)
class AbsNode(BaseNode):  # 继承BaseNode
    """
    绝对值节点
    用法：out = |x|，负数变正数，正数不变
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入形状相同]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = torch.abs(x)  # 逐元素取绝对值
        return {"out": out}  # 返回输出


@node(  # 注册neg节点
    opcode="neg",  # 节点操作码
    label="相反数",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={},  # 无参数
    description="对每个元素取相反数",  # 节点描述
)
class NegNode(BaseNode):  # 继承BaseNode
    """
    相反数节点
    用法：out = -x，正变负，负变正
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入形状相同]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = torch.neg(x)  # 逐元素取相反数
        return {"out": out}  # 返回输出


@node(  # 注册mean节点
    opcode="mean",  # 节点操作码
    label="均值",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {
            "label": "维度",
            "type": "int",
            "value": -1,
            "range": [-10, 10],
        },  # 沿哪个维度求均值
        "keepdim": {
            "label": "保持维度",
            "type": "bool",
            "value": False,
        },  # 是否保持维度
    },
    description="沿某个维度求平均值",  # 节点描述
)
class MeanNode(BaseNode):  # 继承BaseNode
    """
    均值节点
    用法：out = mean(x, dim)，沿指定维度求平均
    调用示例：
        输入 x: shape=[batch, seq_len, features]
        参数 dim=-1 沿最后维度求均值
        输出 out: shape=[batch, seq_len] 或 [batch, seq_len, 1]（keepdim=True时）
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        dim = self.params.get("dim", -1)  # 获取维度
        keepdim = self.params.get("keepdim", False)  # 获取是否保持维度
        out = torch.mean(x, dim=dim, keepdim=keepdim)  # 沿维度求均值
        return {"out": out}  # 返回输出
