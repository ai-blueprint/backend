"""
nodes/loss.py - 损失函数节点组

提供损失函数相关节点
"""

import torch  # 导入torch用于张量操作
import torch.nn as nn  # 导入nn模块用于构建层
import torch.nn.functional as F  # 导入F用于binary_cross_entropy等函数
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册损失函数分类
    id="loss",  # 分类唯一标识
    label="损失函数",  # 分类显示名称
    color="#e44d60",  # 分类颜色，红色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册MSELoss节点
    opcode="mse_loss",  # 节点操作码
    label="均方误差损失",  # 节点显示名称
    ports={  # 端口定义
        "input": {"input": "预测值", "target": "目标值"},  # 两个输入端口：预测值和目标值
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {"label": "聚合方式", "type": "enum", "value": "mean", "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"}},  # 损失聚合方式
    },
    description="计算预测值与目标值之间的均方误差损失",  # 节点描述
)
class MSELossNode(BaseNode):  # 继承BaseNode
    """
    MSELoss均方误差损失节点
    用法：loss = MSE(input, target) = (input - target)^2
    调用示例：
        输入 input: shape=[任意形状]
        输入 target: shape=[与input相同]
        输出 loss: shape=[根据reduction决定]
    """

    def build(self):  # 构建损失函数
        reduction = self.params["reduction"]["value"]  # 获取聚合方式
        self.mse_loss = nn.MSELoss(reduction=reduction)  # 创建MSELoss层

    def compute(self, input):  # 计算方法
        input_tensor = input.get("input")  # 获取预测值
        target = input.get("target")  # 获取目标值
        loss = self.mse_loss(input_tensor, target)  # 计算均方误差损失
        return {"loss": loss}  # 返回损失值


@node(  # 注册CrossEntropyLoss节点
    opcode="cross_entropy_loss",  # 节点操作码
    label="交叉熵损失",  # 节点显示名称
    ports={  # 端口定义
        "input": {"input": "预测logits", "target": "目标类别"},  # 两个输入端口：预测logits和目标类别
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {"label": "聚合方式", "type": "enum", "value": "mean", "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"}},  # 损失聚合方式
        "ignore_index": {"label": "忽略索引", "type": "int", "value": -100, "range": [-1, 65536]},  # 要忽略的目标索引
        "label_smoothing": {"label": "标签平滑", "type": "float", "value": 0.0, "range": [0, 1]},  # 标签平滑因子
    },
    description="多分类问题的标准损失函数，输入为logits",  # 节点描述
)
class CrossEntropyLossNode(BaseNode):  # 继承BaseNode
    """
    CrossEntropyLoss交叉熵损失节点
    用法：常用于分类任务，支持标签平滑
    调用示例：
        输入 input: shape=[batch, num_classes, ...]
        输入 target: shape=[batch, ...] 包含类别索引
        输出 loss: shape=[根据reduction决定]
    """

    def build(self):  # 构建损失函数
        reduction = self.params["reduction"]["value"]  # 获取聚合方式
        ignore_index = self.params["ignore_index"]["value"]  # 获取忽略索引
        label_smoothing = self.params["label_smoothing"]["value"]  # 获取标签平滑
        self.ce_loss = nn.CrossEntropyLoss(  # 创建交叉熵损失层
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def compute(self, input):  # 计算方法
        input_tensor = input.get("input")  # 获取预测logits
        target = input.get("target")  # 获取目标类别
        loss = self.ce_loss(input_tensor, target)  # 计算交叉熵损失
        return {"loss": loss}  # 返回损失值


@node(  # 注册L1Loss节点
    opcode="l1_loss",  # 节点操作码
    label="L1损失",  # 节点显示名称
    ports={  # 端口定义
        "input": {"input": "预测值", "target": "目标值"},  # 两个输入端口：预测值和目标值
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {"label": "聚合方式", "type": "enum", "value": "mean", "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"}},  # 损失聚合方式
    },
    description="计算预测值与目标值之间的L1绝对值误差损失",  # 节点描述
)
class L1LossNode(BaseNode):  # 继承BaseNode
    """
    L1Loss绝对值误差损失节点
    用法：loss = L1Loss(input, target) = |input - target|
    调用示例：
        输入 input: shape=[任意形状]
        输入 target: shape=[与input相同]
        输出 loss: shape=[根据reduction决定]
    """

    def build(self):  # 构建损失函数
        reduction = self.params["reduction"]["value"]  # 获取聚合方式
        self.l1_loss = nn.L1Loss(reduction=reduction)  # 创建L1Loss层

    def compute(self, input):  # 计算方法
        input_tensor = input.get("input")  # 获取预测值
        target = input.get("target")  # 获取目标值
        loss = self.l1_loss(input_tensor, target)  # 计算L1损失
        return {"loss": loss}  # 返回损失值


@node(  # 注册BCELoss节点
    opcode="bce_loss",  # 节点操作码
    label="二分类交叉熵损失",  # 节点显示名称
    ports={  # 端口定义
        "input": {"input": "预测概率", "target": "目标标签"},  # 两个输入端口：预测概率和目标标签
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {"label": "聚合方式", "type": "enum", "value": "mean", "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"}},  # 损失聚合方式
        "weight": {"label": "类别权重", "type": "list", "value": [1.0]},  # 各样本的权重
    },
    description="二分类问题的交叉熵损失，输入为概率（需先sigmoid）",  # 节点描述
)
class BCELossNode(BaseNode):  # 继承BaseNode
    """
    BCELoss二分类交叉熵损失节点
    用法：loss = -[target*log(input) + (1-target)*log(1-input)]
    调用示例：
        输入 input: shape=[任意形状]，值应在[0,1]区间
        输入 target: shape=[与input相同]，值应在[0,1]区间
        输出 loss: shape=[根据reduction决��]
    """

    def build(self):  # 构建损失函数
        reduction = self.params["reduction"]["value"]  # 获取聚合方式
        weight = self.params["weight"]["value"]  # 获取权重
        self.bce_loss = nn.BCELoss(reduction=reduction)  # 创建BCELoss层
        if len(weight) > 1 or weight[0] != 1.0:  # 如果有自定义权重
            self.weight_tensor = torch.tensor(weight, dtype=torch.float)  # 创建权重张量

    def compute(self, input):  # 计算方法
        input_tensor = input.get("input")  # 获取预测概率
        target = input.get("target")  # 获取目标标签

        # 如果有自定义权重，需要手动计算加权损失
        if hasattr(self, "weight_tensor"):
            # 将权重广播到正确的形状
            weight = self.weight_tensor.to(input_tensor.device)  # 移动到相同设备
            # 简单实现：假设权重作用于batch维度
            if weight.dim() == 1 and weight.size(0) == input_tensor.size(0):
                # 扩展权重维度以匹配输入
                weight = weight.view(-1, *([1] * (input_tensor.dim() - 1)))  # 扩展维度
                loss = F.binary_cross_entropy(input_tensor, target, weight=weight, reduction="none")  # 无聚合
                if self.params["reduction"]["value"] == "mean":  # 平均值
                    loss = loss.mean()
                elif self.params["reduction"]["value"] == "sum":  # 总和
                    loss = loss.sum()
                return {"loss": loss}  # 返回损失值
            else:
                # 权重不匹配，使用默认损失
                pass

        loss = self.bce_loss(input_tensor, target)  # 计算二分类交叉熵损失
        return {"loss": loss}  # 返回损失值
