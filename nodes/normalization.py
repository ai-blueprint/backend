"""
nodes/normalization.py - 归一化节点组

提供归一化相关节点：LayerNorm层归一化、GroupNorm分组归一化
"""

import torch  # 导入torch用于张量操作
import torch.nn as nn  # 导入nn模块用于构建层
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册归一化分类
    id="normalization",  # 分类唯一标识
    label="归一化",  # 分类显示名称
    color="#fdab3d",  # 分类颜色，橙色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册LayerNorm节点
    opcode="layer_norm",  # 节点操作码
    label="LayerNorm",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "normalized_shape": {"label": "归一化形状", "type": "list", "value": [64]},  # 归一化的维度形状
        "eps": {"label": "防零极小值", "type": "float", "value": 1e-5, "range": [1e-10, 1e-1]},  # 防止除零的极小值
        "elementwise_affine": {"label": "可学习缩放偏移", "type": "bool", "value": True},  # 是否使用可学习的缩放和偏移
    },
    description="单样本特征维度归一化",  # 节点描述
)
class LayerNormNode(BaseNode):  # 继承BaseNode
    """
    LayerNorm层归一化节点
    用法：对单个样本的特征维度做归一化 out = (x - mean) / sqrt(var + eps) * gamma + beta
    调用示例：
        输入 x: shape=[batch, seq_len, features]
        参数 normalized_shape=[features]
        输出 out: shape=[与输入相同]
    """

    def build(self):  # 构建层
        self.layer_norm = nn.LayerNorm(  # 创建LayerNorm层
            self.params["normalized_shape"]["value"],  # 归一化的维度形状
            eps=self.params["eps"]["value"],  # 防零极小值
            elementwise_affine=self.params["elementwise_affine"]["value"],  # 可学习缩放偏移
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.layer_norm(x)  # 层归一化
        return {"out": out}  # 返回输出


@node(  # 注册GroupNorm节点
    opcode="group_norm",  # 节点操作码
    label="GroupNorm",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "num_groups": {"label": "分组数", "type": "int", "value": 8, "range": [1, 1024]},  # 分组数量
        "num_channels": {"label": "通道数", "type": "int", "value": 64, "range": [1, 65536]},  # 通道数量
        "eps": {"label": "防零极小值", "type": "float", "value": 1e-5, "range": [1e-10, 1e-1]},  # 防止除零的极小值
        "affine": {"label": "可学习缩放偏移", "type": "bool", "value": True},  # 是否使用可学习的缩放和偏移
    },
    description="特征分组，组内归一化",  # 节点描述
)
class GroupNormNode(BaseNode):  # 继承BaseNode
    """
    GroupNorm分组归一化节点
    用法：将通道分成若干组，组内做归一化
    调用示例：
        输入 x: shape=[batch, num_channels, *]
        参数 num_groups=8, num_channels=64 表示64个通道分成8组
        输出 out: shape=[与输入相同]
    """

    def build(self):  # 构建层
        self.group_norm = nn.GroupNorm(  # 创建GroupNorm层
            self.params["num_groups"]["value"],  # 分组数
            self.params["num_channels"]["value"],  # 通道数
            eps=self.params["eps"]["value"],  # 防零极小值
            affine=self.params["affine"]["value"],  # 可学习缩放偏移
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.group_norm(x)  # 分组归一化
        return {"out": out}  # 返回输出
