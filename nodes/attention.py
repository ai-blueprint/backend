"""
nodes/attention.py - 注意力节点组

提供注意力机制相关节点：MultiheadAttention多头注意力、ScaledDotProductAttention缩放点积注意力、CrossAttention跨注意力
"""

import torch  # 导入torch用于张量操作
import torch.nn as nn  # 导入nn模块用于构建层
import torch.nn.functional as F  # 导入F用于缩放点积注意力
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册注意力分类
    id="attention",  # 分类唯一标识
    label="注意力",  # 分类显示名称
    color="#9d4edd",  # 分类颜色，紫色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册MultiheadAttention节点
    opcode="multihead_attention",  # 节点操作码
    label="多头注意力",  # 节点显示名称
    ports={  # 端口定义
        "input": {"q": "查询", "k": "键", "v": "值"},  # 三个输入端口：查询、键、值
        "output": {
            "out": "输出",
            "attn_weights": "注意力权重",
        },  # 两个输出端口：输出、注意力权重
    },
    params={  # 参数定义
        "embed_dim": {
            "label": "嵌入维度",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 输入特征维度
        "num_heads": {
            "label": "头数",
            "type": "int",
            "value": 2,
            "range": [1, 256],
        },  # 注意力头数量
        "dropout": {
            "label": "Dropout率",
            "type": "float",
            "value": 0.1,
            "range": [0, 1],
        },  # Dropout率
        "bias": {
            "label": "偏置",
            "type": "bool",
            "value": True,
        },  # 是否在Q/K/V投影中使用偏置
        "add_bias_kv": {
            "label": "添加K/V偏置",
            "type": "bool",
            "value": False,
        },  # 是否添加独立的K/V偏置
        "add_zero_attn": {
            "label": "添加零注意力",
            "type": "bool",
            "value": False,
        },  # 是否在注意力计算中添加零
        "kdim": {
            "label": "键维度",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 键特征维度（可选）
        "vdim": {
            "label": "值维度",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 值特征维度（可选）
    },
    description="Transformer标准多头注意力机制",  # 节点描述
)
class MultiheadAttentionNode(BaseNode):  # 继承BaseNode
    """
    MultiheadAttention多头注意力节点
    用法：标准的Transformer注意力机制 out = Attention(Q, K, V)
    调用示例：
        输入 q: shape=[batch, seq_len, embed_dim]
        输入 k: shape=[batch, seq_len, embed_dim]
        输入 v: shape=[batch, seq_len, embed_dim]
        输出 out: shape=[batch, seq_len, embed_dim]
        输出 attn_weights: shape=[batch, num_heads, seq_len, seq_len]
    """

    def build(self):  # 构建层
        embedDim = self.params.get("embed_dim", 8)  # 查询特征需要平均分配到每个注意力头
        numHeads = self.params.get("num_heads", 2)  # 默认契约将8维特征拆成两个4维头
        if numHeads <= 0 or embedDim % numHeads != 0:
            raise ValueError("embed_dim必须能被num_heads整除")  # 提前返回清晰的节点参数错误
        self.multihead_attention = nn.MultiheadAttention(  # 创建多头注意力层
            embed_dim=embedDim,  # 嵌入维度
            num_heads=numHeads,  # 头数
            dropout=self.params.get("dropout", 0.1),  # Dropout率
            bias=self.params.get("bias", True),  # 偏置
            add_bias_kv=self.params.get("add_bias_kv", False),  # 添加K/V偏置
            add_zero_attn=self.params.get("add_zero_attn", False),  # 添加零注意力
            kdim=self.params.get("kdim", 8),  # 键维度
            vdim=self.params.get("vdim", 8),  # 值维度
            batch_first=True,  # 节点文档约定输入顺序为[batch, seq, feature]
        )

    def compute(self, input):  # 计算方法
        q = input.get("q")  # 获取查询张量
        k = input.get("k")  # 获取键张量
        v = input.get("v")  # 获取值张量
        out, attn_weights = self.multihead_attention(q, k, v, average_attn_weights=False)  # 返回每个头的独立权重
        return {"out": out, "attn_weights": attn_weights}  # 返回两个输出


@node(  # 注册ScaledDotProductAttention节点
    opcode="scaled_dot_product_attention",  # 节点操作码
    label="缩放点积注意力",  # 节点显示名称
    ports={  # 端口定义
        "input": {"q": "查询", "k": "键", "v": "值"},  # 三个输入端口：查询、键、值
        "output": {
            "out": "输出",
            "attn_weights": "注意力权重",
        },  # 两个输出端口：输出、注意力权重
    },
    params={  # 参数定义
        "dropout": {
            "label": "Dropout率",
            "type": "float",
            "value": 0.1,
            "range": [0, 1],
        },  # Dropout率
        "is_causal": {
            "label": "因果注意力",
            "type": "bool",
            "value": False,
        },  # 是否使用因果掩码（自回归）
        "scale": {
            "label": "缩放因子",
            "type": "float",
            "value": 0.0,
            "range": [0, 100],
        },  # 缩放因子，0表示自动计算
    },
    description="显式计算并返回权重的缩放点积注意力",  # 节点描述
)
class ScaledDotProductAttentionNode(BaseNode):  # 继承BaseNode
    """
    ScaledDotProductAttention缩放点积注意力节点
    用法：显式计算注意力分数、权重和加权结果
    调用示例：
        输入 q: shape=[batch, seq_len, features]
        输入 k: shape=[batch, seq_len, features]
        输入 v: shape=[batch, seq_len, features]
        输出 out: shape=[batch, seq_len, features]
        输出 attn_weights: shape=[batch, seq_len, seq_len]
    """

    def compute(self, input):  # 计算方法
        q = input.get("q")  # 获取查询张量
        k = input.get("k")  # 获取键张量
        v = input.get("v")  # 获取值张量
        dropout = self.params.get("dropout", 0.1)  # 获取Dropout率
        is_causal = self.params.get("is_causal", False)  # 获取因果注意力标志
        scale = self.params.get("scale", 0.0)  # 获取缩放因子

        scaleValue = scale if scale else q.size(-1) ** -0.5  # 与PyTorch默认缩放规则保持一致
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue  # 显式计算分数以同时返回权重
        if is_causal:
            queryLength, keyLength = q.size(-2), k.size(-2)  # 读取查询和键的序列长度
            causalMask = torch.ones(queryLength, keyLength, device=q.device, dtype=torch.bool).tril()  # 构造下三角可见区域
            scores = scores.masked_fill(~causalMask, float("-inf"))  # 屏蔽尚未出现的键位置
        attn_weights = torch.softmax(scores, dim=-1)  # 将分数转换为可解释的注意力权重
        usedDropout = dropout if self.training else 0.0  # 推理模式必须关闭随机失活
        droppedWeights = F.dropout(attn_weights, p=usedDropout, training=self.training)  # 训练时仅对参与加权的副本失活
        out = torch.matmul(droppedWeights, v)  # 使用注意力权重聚合值张量
        return {"out": out, "attn_weights": attn_weights}  # 返回两个输出


@node(  # 注册CrossAttention节点
    opcode="cross_attention",  # 节点操作码
    label="跨注意力",  # 节点显示名称
    ports={  # 端口定义
        "input": {"q": "查询", "k": "键", "v": "值"},  # 三个输入端口：查询、键、值
        "output": {
            "out": "输出",
            "attn_weights": "注意力权重",
        },  # 两个输出端口：输出、注意力权重
    },
    params={  # 参数定义
        "embed_dim": {
            "label": "嵌入维度",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 查询特征维度
        "kdim": {
            "label": "键维度",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 键特征维度
        "vdim": {
            "label": "值维度",
            "type": "int",
            "value": 8,
            "range": [1, 65536],
        },  # 值特征维度
        "num_heads": {
            "label": "头数",
            "type": "int",
            "value": 2,
            "range": [1, 256],
        },  # 注意力头数量
        "dropout": {
            "label": "Dropout率",
            "type": "float",
            "value": 0.1,
            "range": [0, 1],
        },  # Dropout率
        "bias": {"label": "偏置", "type": "bool", "value": True},  # 是否使用偏置
    },
    description="跨模态/跨序列注意力，处理不同来源的查询、键、值",  # 节点描述
)
class CrossAttentionNode(BaseNode):  # 继承BaseNode
    """
    CrossAttention跨注意力节点
    用法：处理来自不同来源的Q、K、V，常用于编码器-解码器架构
    调用示例：
        输入 q: shape=[batch, seq_len_q, embed_dim]  # 解码器查询
        输入 k: shape=[batch, seq_len_k, kdim]       # 编码器键
        输入 v: shape=[batch, seq_len_v, vdim]       # 编码器值
        输出 out: shape=[batch, seq_len_q, embed_dim]
        输出 attn_weights: shape=[batch, num_heads, seq_len_q, seq_len_k]
    """

    def build(self):  # 构建层
        embed_dim = self.params.get("embed_dim", 8)  # 获取查询嵌入维度
        kdim = self.params.get("kdim", 8)  # 获取键维度
        vdim = self.params.get("vdim", 8)  # 获取值维度
        num_heads = self.params.get("num_heads", 2)  # 获取头数
        dropout = self.params.get("dropout", 0.1)  # 获取Dropout率
        bias = self.params.get("bias", True)  # 获取偏置标志
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim必须能被num_heads整除")  # 每个头必须获得相同宽度的查询特征
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias, kdim=kdim, vdim=vdim, batch_first=True)  # 原生层负责Q/K/V投影、分头计算和输出投影

    def compute(self, input):  # 计算方法
        q = input.get("q")  # 获取查询张量
        k = input.get("k")  # 获取键张量
        v = input.get("v")  # 获取值张量

        out, attn_weights = self.attention(q, k, v, average_attn_weights=False)  # 原生实现返回[batch, head, query, key]独立多头权重
        return {"out": out, "attn_weights": attn_weights}  # 返回两个输出
