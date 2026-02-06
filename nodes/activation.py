"""
nodes/activation.py - 激活节点组

提供常用激活函数节点：ReLU、Sigmoid、Tanh、Softmax、Softplus
"""

import torch  # 导入torch用于张量操作
import torch.nn as nn  # 导入nn模块用于构建层
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册激活分类
    id="activation",  # 分类唯一标识
    label="激活",  # 分类显示名称
    color="#a073ff",  # 分类颜色，绿色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册ReLU节点
    opcode="relu",  # 节点操作码
    label="ReLU",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "inplace": {"label": "原地操作", "type": "bool", "value": False},  # 是否原地修改
    },
    description="负数归零，正数不变",  # 节点描述
)
class ReLUNode(BaseNode):  # 继承BaseNode
    """
    ReLU激活节点
    用法：将负数变为0，正数保持不变 out = max(0, x)
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入相同]
    """

    def build(self):  # 构建层
        self.relu = nn.ReLU(  # 创建ReLU层
            inplace=self.params["inplace"]["value"],  # 是否原地操作
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.relu(x)  # ReLU激活
        return {"out": out}  # 返回输出


@node(  # 注册Sigmoid节点
    opcode="sigmoid",  # 节点操作码
    label="Sigmoid",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={},  # 无参数
    description="压缩到0~1，常用于门控开关",  # 节点描述
)
class SigmoidNode(BaseNode):  # 继承BaseNode
    """
    Sigmoid激活节点
    用法：将值压缩到0~1之间 out = 1 / (1 + exp(-x))
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入相同]，值域(0,1)
    """

    def build(self):  # 构建层
        self.sigmoid = nn.Sigmoid()  # 创建Sigmoid层

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.sigmoid(x)  # Sigmoid激活
        return {"out": out}  # 返回输出


@node(  # 注册Tanh节点
    opcode="tanh",  # 节点操作码
    label="Tanh",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={},  # 无参数
    description="压缩到-1~1，中心对称",  # 节点描述
)
class TanhNode(BaseNode):  # 继承BaseNode
    """
    Tanh激活节点
    用法：将值压缩到-1~1之间 out = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入相同]，值域(-1,1)
    """

    def build(self):  # 构建层
        self.tanh = nn.Tanh()  # 创建Tanh层

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.tanh(x)  # Tanh激活
        return {"out": out}  # 返回输出


@node(  # 注册Softmax节点
    opcode="softmax",  # 节点操作码
    label="Softmax",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {"label": "维度", "type": "int", "value": -1, "range": [-10, 10]},  # 沿哪个维度做softmax
    },
    description="变成概率分布，加起来等于1",  # 节点描述
)
class SoftmaxNode(BaseNode):  # 继承BaseNode
    """
    Softmax激活节点
    用法：沿指定维度将值转为概率分布 out_i = exp(x_i) / sum(exp(x_j))
    调用示例：
        输入 x: shape=[batch, classes]
        参数 dim=-1 表示沿最后一个维度
        输出 out: shape=[与输入相同]，沿dim维度和为1
    """

    def build(self):  # 构建层
        self.softmax = nn.Softmax(  # 创建Softmax层
            dim=self.params["dim"]["value"],  # 指定维度
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.softmax(x)  # Softmax激活
        return {"out": out}  # 返回输出


@node(  # 注册Softplus节点
    opcode="softplus",  # 节点操作码
    label="Softplus",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "beta": {"label": "平滑系数", "type": "float", "value": 1.0, "range": [0.01, 100]},  # 控制平滑程度
        "threshold": {"label": "线性阈值", "type": "float", "value": 20.0, "range": [1, 100]},  # 超过此值退化为线性
    },
    description="ReLU的平滑版，永远为正",  # 节点描述
)
class SoftplusNode(BaseNode):  # 继承BaseNode
    """
    Softplus激活节点
    用法：ReLU的平滑近似 out = (1/beta) * log(1 + exp(beta * x))
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入相同]，值域(0, +inf)
    """

    def build(self):  # 构建层
        self.softplus = nn.Softplus(  # 创建Softplus层
            beta=self.params["beta"]["value"],  # 平滑系数
            threshold=self.params["threshold"]["value"],  # 线性阈值
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.softplus(x)  # Softplus激活
        return {"out": out}  # 返回输出
