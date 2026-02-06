"""
nodes/transform.py - 变换节点组

提供线性变换相关节点：Linear全连接、Bilinear双线性交互
"""

import torch  # 导入torch用于张量操作
import torch.nn as nn  # 导入nn模块用于构建层
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册变换分类
    id="transform",  # 分类唯一标识
    label="变换",  # 分类显示名称
    color="#82cbfa",  # 分类颜色，浅蓝色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册Linear节点
    opcode="linear",  # 节点操作码
    label="全连接层",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "in_features": {"label": "输入特征数", "type": "int", "value": 64, "range": [1, 1024]},  # 输入维度
        "out_features": {"label": "输出特征数", "type": "int", "value": 64, "range": [1, 1024]},  # 输出维度
        "bias": {"label": "偏置", "type": "bool", "value": False},  # 是否使用偏置
    },
    description="输入乘权重加偏置，最基础的全连接",  # 节点描述
)
class LinearNode(BaseNode):  # 继承BaseNode
    """
    Linear全连接节点
    用法：输入张量x经过线性变换 out = x @ W^T + b
    调用示例：
        输入 x: shape=[batch, in_features]
        输出 out: shape=[batch, out_features]
    """

    def build(self):  # 构建层
        self.linear = nn.Linear(  # 创建线性层
            self.params["in_features"]["value"],  # 输入特征数
            self.params["out_features"]["value"],  # 输出特征数
            bias=self.params["bias"]["value"],  # 是否使用偏置
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.linear(x)  # 线性变换
        return {"out": out}  # 返回输出


@node(  # 注册Bilinear节点
    opcode="bilinear",  # 节点操作码
    label="Bilinear",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x1": "", "x2": ""},  # 两个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "in1_features": {"label": "输入1特征数", "type": "int", "value": 64, "range": [1, 1024]},  # 输入1维度
        "in2_features": {"label": "输入2特征数", "type": "int", "value": 64, "range": [1, 1024]},  # 输入2维度
        "out_features": {"label": "输出特征数", "type": "int", "value": 64, "range": [1, 1024]},  # 输出维度
        "bias": {"label": "偏置", "type": "bool", "value": False},  # 是否使用偏置
    },
    description="两组输入交叉相乘，建模交互关系",  # 节点描述
)
class BilinearNode(BaseNode):  # 继承BaseNode
    """
    Bilinear双线性节点
    用法：两组输入交叉相乘 out = x1^T A x2 + b
    调用示例：
        输入 x1: shape=[batch, in1_features]
        输入 x2: shape=[batch, in2_features]
        输出 out: shape=[batch, out_features]
    """

    def build(self):  # 构建层
        self.bilinear = nn.Bilinear(  # 创建双线性层
            self.params["in1_features"]["value"],  # 输入1特征数
            self.params["in2_features"]["value"],  # 输入2特征数
            self.params["out_features"]["value"],  # 输出特征数
            bias=self.params["bias"]["value"],  # 是否使用偏置
        )

    def compute(self, input):  # 计算方法
        x1 = input.get("x1")  # 获取输入1张量
        x2 = input.get("x2")  # 获取输入2张量
        out = self.bilinear(x1, x2)  # 双线性变换
        return {"out": out}  # 返回输出
