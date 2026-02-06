"""
nodes/transform.py - 变换节点组

提供线性变换相关节点：Linear全连接、Bilinear双线性交互
"""

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
        "in_features": {"label": "输入特征数", "type": "int", "value": 64},  # 输入维度
        "out_features": {"label": "输出特征数", "type": "int", "value": 64},  # 输出维度
        "bias": {"label": "偏置", "type": "bool", "value": False},  # 是否使用偏置
    },
    description="输入乘权重加偏置，最基础层",  # 节点描述
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
