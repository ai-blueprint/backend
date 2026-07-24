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
        "in_features": {
            "label": "输入特征数",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 输入维度
        "out_features": {
            "label": "输出特征数",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 输出维度
        "bias": {"label": "偏置", "type": "bool", "value": True},  # 默认使用偏置
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
            self.params.get("in_features", 8),  # 输入特征数，默认匹配最后一维8
            self.params.get("out_features", 8),  # 输出特征数，默认保持最后一维8
            bias=self.params.get("bias", True),  # 是否使用偏置
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.linear(x)  # 线性变换
        return {"out": out}  # 返回输出


@node(  # 注册Conv节点
    opcode="conv",  # 节点操作码
    label="卷积层",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": ""},  # 一个输入端口
        "output": {"out": ""},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {
            "label": "维度",
            "type": "enum",
            "value": "1d",
            "options": {"1d": "1D卷积", "2d": "2D卷积", "3d": "3D卷积"},
        },  # 卷积维度选择
        "in_channels": {
            "label": "输入通道数",
            "type": "int",
            "value": 4,
            "range": [1, 65536],
        },  # 输入通道数
        "out_channels": {
            "label": "输出通道数",
            "type": "int",
            "value": 4,
            "range": [1, 65536],
        },  # 输出通道数
        "kernel_size": {
            "label": "卷积核大小",
            "type": "list",
            "value": [3],
        },  # 卷积核尺寸，支持单值或逗号分隔
        "stride": {"label": "步幅", "type": "list", "value": [1]},  # 默认步幅为1
        "padding": {"label": "填充", "type": "list", "value": [1]},  # 默认两侧填充1
        "dilation": {"label": "膨胀", "type": "list", "value": [1]},  # 默认膨胀系数为1
        "groups": {
            "label": "分组数",
            "type": "int",
            "value": 1,
            "range": [1, 65536],
        },  # 分组卷积的组数
        "bias": {"label": "偏置", "type": "bool", "value": True},  # 是否使用偏置
        "padding_mode": {
            "label": "填充模式",
            "type": "enum",
            "value": "zeros",
            "options": {
                "zeros": "零填充",
                "reflect": "反射填充",
                "replicate": "复制填充",
                "circular": "循环填充",
            },
        },  # 填充模式
    },
    description="对输入做卷积运算，支持1D/2D/3D",  # 节点描述
)
class ConvNode(BaseNode):  # 继承BaseNode
    """
    Conv卷积节点
    用法：输入张量x经过卷积变换
    调用示例：
        1D输入 x: shape=[batch, channels, length]
        2D输入 x: shape=[batch, channels, height, width]
        3D输入 x: shape=[batch, channels, depth, height, width]
    """

    def parseList(self, raw):  # 解析卷积尺寸，兼容列表、元组和字符串
        if isinstance(raw, (list, tuple)):  # 前端可能直接传递结构化序列
            nums = [int(value) for value in raw]  # 将每一项统一转换为整数
        else:  # 字符串既可能是单值，也可能带列表括号
            text = str(raw).strip().strip("[]()")  # 去掉空白及常见序列括号
            nums = [int(part.strip()) for part in text.split(",") if part.strip()]  # 解析逗号分隔整数
        if not nums:  # 空尺寸无法构造任何卷积层
            raise ValueError("卷积尺寸参数不能为空")  # 在构建阶段给出明确配置错误
        if len(nums) == 1:  # 如果只有一个值
            return nums[0]  # 返回单个整数，PyTorch会自动广播
        return tuple(nums)  # 多维尺寸固定为PyTorch接受的元组

    def build(self):  # 构建卷积层
        dim = self.params.get("dim", "1d")  # 获取维度选择，默认执行1D卷积
        convCls = {"1d": nn.Conv1d, "2d": nn.Conv2d, "3d": nn.Conv3d}[
            dim
        ]  # 根据维度选择对应的Conv类

        in_channels = self.params.get("in_channels", 4)  # 读取输入通道数
        out_channels = self.params.get("out_channels", 4)  # 读取输出通道数
        groups = self.params.get("groups", 1)  # 读取分组数
        if in_channels % groups != 0 or out_channels % groups != 0:  # 每组必须均分输入和输出通道
            raise ValueError("卷积分组数必须同时整除输入通道数和输出通道数")  # 提前反馈可理解的配置错误

        self.conv = convCls(  # 创建卷积层
            in_channels=in_channels,  # 输入通道数
            out_channels=out_channels,  # 输出通道数
            kernel_size=self.parseList(
                self.params.get("kernel_size", [3])
            ),  # 卷积核大小
            stride=self.parseList(self.params.get("stride", [1])),  # 步幅
            padding=self.parseList(self.params.get("padding", [1])),  # 填充
            dilation=self.parseList(self.params.get("dilation", [1])),  # 膨胀
            groups=groups,  # 使用已经过整除校验的分组数
            bias=self.params.get("bias", True),  # 偏置
            padding_mode=self.params.get("padding_mode", "zeros"),  # 填充模式
        )

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = self.conv(x)  # 卷积运算
        return {"out": out}  # 返回输出
