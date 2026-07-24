"""
nodes/loss.py - 损失函数节点组

提供损失函数相关节点
"""

import torch  # 导入torch用于张量操作
import torch.nn as nn  # 导入nn模块用于构建层
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
        "input": {
            "input": "预测值",
            "target": "目标值",
        },  # 两个输入端口：预测值和目标值
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {
            "label": "聚合方式",
            "type": "enum",
            "value": "mean",
            "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"},
        },  # 损失聚合方式
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
        reduction = self.params.get("reduction", "mean")  # 获取聚合方式
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
        "input": {
            "input": "预测logits",
            "target": "目标类别",
        },  # 两个输入端口：预测logits和目标类别
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {
            "label": "聚合方式",
            "type": "enum",
            "value": "mean",
            "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"},
        },  # 损失聚合方式
        "ignore_index": {
            "label": "忽略索引",
            "type": "int",
            "value": -100,
            "range": [-100, 65536],
        },  # 要忽略的目标索引
        "label_smoothing": {
            "label": "标签平滑",
            "type": "float",
            "value": 0.0,
            "range": [0, 1],
        },  # 标签平滑因子
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
        reduction = self.params.get("reduction", "mean")  # 获取聚合方式
        ignore_index = self.params.get("ignore_index", -100)  # 获取忽略索引
        label_smoothing = self.params.get("label_smoothing", 0.0)  # 获取标签平滑
        self.ce_loss = nn.CrossEntropyLoss(  # 创建交叉熵损失层
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def compute(self, input):  # 计算方法
        input_tensor = input.get("input")  # 获取预测logits
        target = input.get("target")  # 获取目标类别或同形类别分数
        if target.shape == input_tensor.shape:  # 默认Input同时连接两端时将同形目标解释为类别分数
            target = target.argmax(dim=1)  # PyTorch交叉熵的类别维固定为第1维
        loss = self.ce_loss(input_tensor, target.long())  # 标准索引目标保持原语义并统一索引类型
        return {"loss": loss}  # 返回损失值


@node(  # 注册L1Loss节点
    opcode="l1_loss",  # 节点操作码
    label="L1损失",  # 节点显示名称
    ports={  # 端口定义
        "input": {
            "input": "预测值",
            "target": "目标值",
        },  # 两个输入端口：预测值和目标值
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {
            "label": "聚合方式",
            "type": "enum",
            "value": "mean",
            "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"},
        },  # 损失聚合方式
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
        reduction = self.params.get("reduction", "mean")  # 获取聚合方式
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
        "input": {
            "input": "预测logits",
            "target": "目标标签",
        },  # 两个输入端口：预测logits和目标标签
        "output": {"loss": "损失值"},  # 一个输出端口：损失值
    },
    params={  # 参数定义
        "reduction": {
            "label": "聚合方式",
            "type": "enum",
            "value": "mean",
            "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"},
        },  # 损失聚合方式
        "pos_weight": {"label": "正类权重", "type": "float", "value": 1.0, "range": [0.0, 100.0]},  # 正样本损失倍率
    },
    description="数值稳定的二分类交叉熵损失，输入为未经过sigmoid的logits",  # 节点描述
)
class BCELossNode(BaseNode):  # 继承BaseNode
    """
    BCEWithLogitsLoss二分类交叉熵损失节点
    用法：内部合并Sigmoid与BCE，避免概率接近0或1时数值不稳定
    调用示例：
        输入 input: shape=[任意形状]，值为未归一化logits
        输入 target: shape=[与input相同]，超出[0,1]时自动从[-1,1]映射
        输出 loss: shape=[根据reduction决定]
    """

    def build(self):  # 构建损失函数
        reduction = self.params.get("reduction", "mean")  # 获取聚合方式
        pos_weight = self.params.get("pos_weight", 1.0)  # 获取正类权重，默认不额外加权
        self.bce_loss = nn.BCEWithLogitsLoss(  # 创建数值稳定的logits损失层
            reduction=reduction,
            pos_weight=torch.tensor(pos_weight, dtype=torch.float),
        )

    def compute(self, input):  # 计算方法
        input_tensor = input.get("input")  # 获取预测logits
        target = input.get("target")  # 获取目标标签
        if torch.any((target < 0) | (target > 1)):  # 默认随机Input位于[-1,1)，需要转换为合法软标签
            target = ((target + 1.0) / 2.0).clamp(0.0, 1.0)  # 映射并兜底限制到目标区间
        loss = self.bce_loss(input_tensor, target.to(input_tensor.dtype))  # 计算稳定的二分类交叉熵损失
        return {"loss": loss}  # 返回损失值
