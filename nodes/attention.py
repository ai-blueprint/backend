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
        "output": {"out": "输出", "attn_weights": "注意力权重"},  # 两个输出端口：输出、注意力权重
    },
    params={  # 参数定义
        "embed_dim": {"label": "嵌入维度", "type": "int", "value": 512, "range": [1, 65536]},  # 输入特征维度
        "num_heads": {"label": "头数", "type": "int", "value": 8, "range": [1, 256]},  # 注意力头数量
        "dropout": {"label": "Dropout率", "type": "float", "value": 0.1, "range": [0, 1]},  # Dropout率
        "bias": {"label": "偏置", "type": "bool", "value": True},  # 是否在Q/K/V投影中使用偏置
        "add_bias_kv": {"label": "添加K/V偏置", "type": "bool", "value": False},  # 是否添加独立的K/V偏置
        "add_zero_attn": {"label": "添加零注意力", "type": "bool", "value": False},  # 是否在注意力计算中添加零
        "kdim": {"label": "键维度", "type": "int", "value": 512, "range": [1, 65536]},  # 键特征维度（可选）
        "vdim": {"label": "值维度", "type": "int", "value": 512, "range": [1, 65536]},  # 值特征维度（可选）
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
        self.multihead_attention = nn.MultiheadAttention(  # 创建多头注意力层
            embed_dim=self.params["embed_dim"]["value"],  # 嵌入维度
            num_heads=self.params["num_heads"]["value"],  # 头数
            dropout=self.params["dropout"]["value"],  # Dropout率
            bias=self.params["bias"]["value"],  # 偏置
            add_bias_kv=self.params["add_bias_kv"]["value"],  # 添加K/V偏置
            add_zero_attn=self.params["add_zero_attn"]["value"],  # 添加零注意力
            kdim=self.params["kdim"]["value"],  # 键维度
            vdim=self.params["vdim"]["value"],  # 值维度
        )

    def compute(self, input):  # 计算方法
        q = input.get("q")  # 获取查询张量
        k = input.get("k")  # 获取键张量
        v = input.get("v")  # 获取值张量
        out, attn_weights = self.multihead_attention(q, k, v)  # 多头注意力计算
        return {"out": out, "attn_weights": attn_weights}  # 返回两个输出


@node(  # 注册ScaledDotProductAttention节点
    opcode="scaled_dot_product_attention",  # 节点操作码
    label="缩放点积注意力",  # 节点显示名称
    ports={  # 端口定义
        "input": {"q": "查询", "k": "键", "v": "值"},  # 三个输入端口：查询、键、值
        "output": {"out": "输出", "attn_weights": "注意力权重"},  # 两个输出端口：输出、注意力权重
    },
    params={  # 参数定义
        "dropout": {"label": "Dropout率", "type": "float", "value": 0.1, "range": [0, 1]},  # Dropout率
        "is_causal": {"label": "因果注意力", "type": "bool", "value": False},  # 是否使用因果掩码（自回归）
        "scale": {"label": "缩放因子", "type": "float", "value": 0.0, "range": [0, 100]},  # 缩放因子，0表示自动计算
    },
    description="PyTorch 2.0+原生高效的缩放点积注意力",  # 节点描述
)
class ScaledDotProductAttentionNode(BaseNode):  # 继承BaseNode
    """
    ScaledDotProductAttention缩放点积注意力节点
    用法：PyTorch 2.0+原生高效的缩放点积注意力
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
        dropout = self.params["dropout"]["value"]  # 获取Dropout率
        is_causal = self.params["is_causal"]["value"]  # 获取因果注意力标志
        scale = self.params["scale"]["value"]  # 获取缩放因子

        # 使用PyTorch原生缩放点积注意力
        out, attn_weights = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=None if scale == 0 else scale,  # scale=None表示自动计算
            return_attn_weights=True
        )
        return {"out": out, "attn_weights": attn_weights}  # 返回两个输出


@node(  # 注册CrossAttention节点
    opcode="cross_attention",  # 节点操作码
    label="跨注意力",  # 节点显示名称
    ports={  # 端口定义
        "input": {"q": "查询", "k": "键", "v": "值"},  # 三个输入端口：查询、键、值
        "output": {"out": "输出", "attn_weights": "注意力权重"},  # 两个输出端口：输出、注意力权重
    },
    params={  # 参数定义
        "embed_dim": {"label": "嵌入维度", "type": "int", "value": 512, "range": [1, 65536]},  # 查询特征维度
        "kdim": {"label": "键维度", "type": "int", "value": 512, "range": [1, 65536]},  # 键特征维度
        "vdim": {"label": "值维度", "type": "int", "value": 512, "range": [1, 65536]},  # 值特征维度
        "num_heads": {"label": "头数", "type": "int", "value": 8, "range": [1, 256]},  # 注意力头数量
        "dropout": {"label": "Dropout率", "type": "float", "value": 0.1, "range": [0, 1]},  # Dropout率
        "bias": {"label": "偏置", "type": "bool", "value": True},  # 是否使用偏置
        "project": {"label": "投影维度", "type": "bool", "value": True},  # 是否对K/V进行维度投影
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
        embed_dim = self.params["embed_dim"]["value"]  # 获取查询嵌入维度
        kdim = self.params["kdim"]["value"]  # 获取键维度
        vdim = self.params["vdim"]["value"]  # 获取值维度
        num_heads = self.params["num_heads"]["value"]  # 获取头数
        dropout = self.params["dropout"]["value"]  # 获取Dropout率
        bias = self.params["bias"]["value"]  # 获取偏置标志
        project = self.params["project"]["value"]  # 获取投影标志

        self.embed_dim = embed_dim  # 保存嵌入维度
        self.num_heads = num_heads  # 保存头数
        self.project = project  # 保存投影标志

        if project:  # 如果需要对K/V进行维度投影
            # 创建Q、K、V的线性投影层
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 查询投影
            self.k_proj = nn.Linear(kdim, embed_dim, bias=bias)  # 键投影到embed_dim
            self.v_proj = nn.Linear(vdim, embed_dim, bias=bias)  # 值投影到embed_dim
        else:  # 不使用投影，直接使用输入维度
            # 确保K/V维度与embed_dim匹配
            if kdim != embed_dim or vdim != embed_dim:
                raise ValueError("不使用投影时，kdim和vdim必须等于embed_dim")

        # 创建最终输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出投影

        # 设置Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def compute(self, input):  # 计算方法
        q = input.get("q")  # 获取查询张量
        k = input.get("k")  # 获取键张量
        v = input.get("v")  # 获取值张量

        if self.project:  # 如果使用投影
            q = self.q_proj(q)  # 投影查询
            k = self.k_proj(k)  # 投影键
            v = self.v_proj(v)  # 投影值

        # 实现跨注意力计算
        # 1. 缩放点积注意力
        d_k = q.size(-1)  # 获取查询维度
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # 计算注意力分数

        # 2. 应用softmax
        attn_weights = F.softmax(scores, dim=-1)  # 计算注意力权重

        # 3. 应用Dropout（如果有）
        if self.dropout:
            attn_weights = self.dropout(attn_weights)

        # 4. 计算加权和
        out = torch.matmul(attn_weights, v)  # 应用注意力权重到值

        # 5. 输出投影
        out = self.out_proj(out)

        return {"out": out, "attn_weights": attn_weights}  # 返回两个输出