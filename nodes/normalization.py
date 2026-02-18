"""
nodes/normalization.py - 归一化节点组

提供归一化相关节点：LayerNorm层归一化、GroupNorm分组归一化、BatchNorm批归一化、InstanceNorm实例归一化、RMSNorm均方根归一化
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
    label="层归一化",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "normalized_shape": {"label": "归一化形状", "type": "list", "value": [64]},  # 归一化的维度形状
        "eps": {"label": "防零极小值", "type": "float", "value": 1e-5, "range": [1e-10, 1e-1]},  # 防止除零的极小值
        "elementwise_affine": {"label": "可学习缩放偏移", "type": "bool", "value": True},  # 是否使用可学习的缩放和偏移
    },
    description="对每个样本的特征做归一化",  # 节点描述
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
    label="组归一化",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "num_groups": {"label": "分组数", "type": "int", "value": 8, "range": [1, 1024]},  # 分组数量
        "num_channels": {"label": "通道数", "type": "int", "value": 64, "range": [1, 65536]},  # 通道数量
        "eps": {"label": "防零极小值", "type": "float", "value": 1e-5, "range": [1e-10, 1e-1]},  # 防止除零的极小值
        "affine": {"label": "可学习缩放偏移", "type": "bool", "value": True},  # 是否使用可学习的缩放和偏移
    },
    description="通道分组后组内归一化",  # 节点描述
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


@node(  # 注册BatchNorm节点
    opcode="batch_norm",  # 节点操作码
    label="批归一化",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {"label": "维度", "type": "enum", "value": "2d", "options": {"1d": "1D", "2d": "2D", "3d": "3D"}},  # 维度选择
        "num_features": {"label": "特征数", "type": "int", "value": 64},  # 特征/通道数
        "eps": {"label": "防零极小值", "type": "float", "value": 1e-5},  # 防止除零的极小值
        "momentum": {"label": "动量", "type": "float", "value": 0.1, "range": [0, 1]},  # 运行均值/方差的更新动量
        "affine": {"label": "可学习缩放偏移", "type": "bool", "value": True},  # 是否使用可学习的gamma和beta
        "track_running_stats": {"label": "跟踪运行统计", "type": "bool", "value": True},  # 是否跟踪运行均值和方差
    },
    description="对每个batch的特征做归一化，CNN标配，支持1D/2D/3D",  # 节点描述
)
class BatchNormNode(BaseNode):  # 继承BaseNode
    """
    BatchNorm批归一化节点
    用法：对mini-batch内的特征做归一化 out = (x - mean) / sqrt(var + eps) * gamma + beta
    调用示例：
        1D输入 x: shape=[batch, features, length]
        2D输入 x: shape=[batch, channels, height, width]
        3D输入 x: shape=[batch, channels, depth, height, width]
    """

    def build(self):  # 构建层
        dim = self.params["dim"]["value"]  # 获取维度选择
        bnCls = {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}[dim]  # 根据维度选择对应的BatchNorm类

        self.batch_norm = bnCls(  # 创建BatchNorm层
            num_features=self.params["num_features"]["value"],  # 特征数
            eps=self.params["eps"]["value"],  # 防零极小值
            momentum=self.params["momentum"]["value"],  # 动量
            affine=self.params["affine"]["value"],  # 可学习缩放偏移
            track_running_stats=self.params["track_running_stats"]["value"],  # 跟踪运行统计
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.batch_norm(x)  # 批归一化
        return {"out": out}  # 返回输出


@node(  # 注册InstanceNorm节点
    opcode="instance_norm",  # 节点操作码
    label="实例归一化",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {"label": "维度", "type": "enum", "value": "2d", "options": {"1d": "1D", "2d": "2D", "3d": "3D"}},  # 维度选择
        "num_features": {"label": "特征数", "type": "int", "value": 64},  # 特征/通道数
        "eps": {"label": "防零极小值", "type": "float", "value": 1e-5},  # 防止除零的极小值
        "momentum": {"label": "动量", "type": "float", "value": 0.1, "range": [0, 1]},  # 运行均值/方差的更新动量
        "affine": {"label": "可学习缩放偏移", "type": "bool", "value": False},  # 默认不使用可学习参数
        "track_running_stats": {"label": "跟踪运行统计", "type": "bool", "value": False},  # 默认不跟踪
    },
    description="对每个样本的每个通道独立归一化，风格迁移常用，支持1D/2D/3D",  # 节点描述
)
class InstanceNormNode(BaseNode):  # 继承BaseNode
    """
    InstanceNorm实例归一化节点
    用法：对每个样本的每个通道独立做归一化，消除实例级别的风格信息
    调用示例：
        1D输入 x: shape=[batch, features, length]
        2D输入 x: shape=[batch, channels, height, width]
        3D输入 x: shape=[batch, channels, depth, height, width]
    """

    def build(self):  # 构建层
        dim = self.params["dim"]["value"]  # 获取维度选择
        inCls = {"1d": nn.InstanceNorm1d, "2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d}[dim]  # 根据维度选择对应的InstanceNorm类

        self.instance_norm = inCls(  # 创建InstanceNorm层
            num_features=self.params["num_features"]["value"],  # 特征数
            eps=self.params["eps"]["value"],  # 防零极小值
            momentum=self.params["momentum"]["value"],  # 动量
            affine=self.params["affine"]["value"],  # 可学习缩放偏移
            track_running_stats=self.params["track_running_stats"]["value"],  # 跟踪运行统计
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.instance_norm(x)  # 实例归一化
        return {"out": out}  # 返回输出


@node(  # 注册RMSNorm节点
    opcode="rms_norm",  # 节点操作码
    label="均方根归一化",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "normalized_shape": {"label": "归一化形状", "type": "list", "value": [64]},  # 归一化的维度形状
        "eps": {"label": "防零极小值", "type": "float", "value": 1e-5},  # 防止除零的极小值
        "elementwise_affine": {"label": "可学习缩放", "type": "bool", "value": True},  # 是否使用可学习的缩放参数
    },
    description="只用均方根归一化不减均值，LLaMA等大模型常用",  # 节点描述
)
class RMSNormNode(BaseNode):  # 继承BaseNode
    """
    RMSNorm均方根归一化节点
    用法：out = x / sqrt(mean(x^2) + eps) * gamma，比LayerNorm少减均值，计算更快
    调用示例：
        输入 x: shape=[batch, seq_len, features]
        参数 normalized_shape=[features]
        输出 out: shape=[与输入相同]
    """

    def build(self):  # 构建层
        self.rms_norm = nn.RMSNorm(  # 创建RMSNorm层
            self.params["normalized_shape"]["value"],  # 归一化的维度形状
            eps=self.params["eps"]["value"],  # 防零极小值
            elementwise_affine=self.params["elementwise_affine"]["value"],  # 可学习缩放
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.rms_norm(x)  # 均方根归一化
        return {"out": out}  # 返回输出
