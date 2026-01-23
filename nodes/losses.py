"""
损失函数节点组

提供常用的损失函数节点：CrossEntropy, MSE, L1, BCE 等。
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List

from decorators import category, node
from nodes import create_loss_node


# ==================== 分类定义 ====================

@category(
    id="loss",
    name="损失函数",
    color="#FF6B6B"
)
def loss_category():
    pass


# ==================== 节点定义 ====================

@node(
    id="cross_entropy",
    name="交叉熵损失 (CrossEntropy)",
    inputs=["input", "target"],
    outputs=["loss"],
    params={}
)
def cross_entropy_loss():
    """多分类交叉熵损失"""
    return create_loss_node(
        nn.CrossEntropyLoss,
        target_dtype=torch.long
    )


@node(
    id="mse_loss",
    name="均方误差损失 (MSE)",
    inputs=["input", "target"],
    outputs=["loss"],
    params={"reduction": "mean"}
)
def mse_loss():
    """均方误差损失"""
    return create_loss_node(
        nn.MSELoss,
        build_kwargs=lambda p: {"reduction": p.get("reduction", "mean")}
    )


@node(
    id="l1_loss",
    name="L1损失 (MAE)",
    inputs=["input", "target"],
    outputs=["loss"],
    params={"reduction": "mean"}
)
def l1_loss():
    """L1损失（平均绝对误差）"""
    return create_loss_node(
        nn.L1Loss,
        build_kwargs=lambda p: {"reduction": p.get("reduction", "mean")}
    )


@node(
    id="bce_loss",
    name="二元交叉熵损失 (BCE)",
    inputs=["input", "target"],
    outputs=["loss"],
    params={}
)
def bce_loss():
    """二元交叉熵损失（需要先经过 Sigmoid）"""
    return create_loss_node(
        nn.BCELoss,
        target_dtype=torch.float32
    )


@node(
    id="bce_with_logits",
    name="BCE With Logits",
    inputs=["input", "target"],
    outputs=["loss"],
    params={}
)
def bce_with_logits_loss():
    """二元交叉熵损失（内置 Sigmoid）"""
    return create_loss_node(
        nn.BCEWithLogitsLoss,
        target_dtype=torch.float32
    )


@node(
    id="nll_loss",
    name="负对数似然损失 (NLL)",
    inputs=["input", "target"],
    outputs=["loss"],
    params={}
)
def nll_loss():
    """负对数似然损失（需要先经过 LogSoftmax）"""
    return create_loss_node(
        nn.NLLLoss,
        target_dtype=torch.long
    )


@node(
    id="smooth_l1_loss",
    name="Smooth L1损失",
    inputs=["input", "target"],
    outputs=["loss"],
    params={"beta": 1.0}
)
def smooth_l1_loss():
    """Smooth L1损失（Huber损失）"""
    return create_loss_node(
        nn.SmoothL1Loss,
        build_kwargs=lambda p: {"beta": p.get("beta", 1.0)}
    )
