"""
nodes/loss_extra.py - PyTorch 扩展损失函数节点

补充回归、概率分布、排序和度量学习损失。节点会把默认随机张量转换为各损失函数
要求的合法目标格式，用户提供符合 PyTorch 契约的输入时仍保留标准计算语义。
"""

import torch  # 张量能力，用于把教学随机输入转换为合法目标
import torch.nn as nn  # PyTorch 损失函数能力，用于构建官方实现

from registry import BaseNode, category, node  # 节点注册能力，将扩展损失暴露给编辑器


category(id="loss", label="损失", color="#e44d60", icon="")  # 所有损失节点统一归入损失分类


reductionParams = {"reduction": {"label": "聚合方式", "type": "enum", "value": "mean", "options": {"mean": "平均值", "sum": "总和", "none": "无聚合"}}}  # 大多数损失共享聚合参数


class TwoValueLossNode(BaseNode):
    """统一承载同形预测值和目标值损失。"""

    lossClass = nn.MSELoss  # 子类覆盖实际损失类型

    def build(self):
        self.loss = self.createLoss()  # 构建一次损失模块

    def createLoss(self):
        return self.lossClass(reduction=self.params.get("reduction", "mean"))  # 默认使用标准聚合参数

    def compute(self, input):
        return {"loss": self.loss(input.get("input"), input.get("target"))}  # 同形值直接交给官方损失


@node(opcode="smooth_l1_loss", label="平滑L1损失", ports={"input": {"input": "预测值", "target": "目标值"}, "output": {"loss": "损失值"}}, params={**reductionParams, "beta": {"label": "二次区宽度", "type": "float", "value": 1.0, "range": [0.000001, 100]}}, description="小误差使用平方，大误差使用绝对值")
class SmoothL1LossNode(TwoValueLossNode):
    def createLoss(self):
        return nn.SmoothL1Loss(reduction=self.params.get("reduction", "mean"), beta=self.params.get("beta", 1.0))  # beta 控制平方区间宽度


@node(opcode="huber_loss", label="Huber损失", ports={"input": {"input": "预测值", "target": "目标值"}, "output": {"loss": "损失值"}}, params={**reductionParams, "delta": {"label": "切换阈值", "type": "float", "value": 1.0, "range": [0.000001, 100]}}, description="结合均方误差稳定性和绝对误差鲁棒性")
class HuberLossNode(TwoValueLossNode):
    def createLoss(self):
        return nn.HuberLoss(reduction=self.params.get("reduction", "mean"), delta=self.params.get("delta", 1.0))  # delta 决定平方与线性切换点


@node(opcode="poisson_nll_loss", label="泊松负对数似然", ports={"input": {"input": "对数率", "target": "计数目标"}, "output": {"loss": "损失值"}}, params={**reductionParams, "log_input": {"label": "输入为对数率", "type": "bool", "value": True}, "full": {"label": "加入完整项", "type": "bool", "value": False}, "eps": {"label": "防零极小值", "type": "float", "value": 0.00000001, "range": [1e-12, 1]}}, description="适合计数数据的泊松分布损失")
class PoissonNLLLossNode(BaseNode):
    def build(self):
        self.loss = nn.PoissonNLLLoss(log_input=self.params.get("log_input", True), full=self.params.get("full", False), eps=self.params.get("eps", 1e-8), reduction=self.params.get("reduction", "mean"))  # 创建泊松似然损失

    def compute(self, input):
        target = input.get("target").clamp_min(0.0)  # 泊松计数目标不能为负
        prediction = input.get("input")  # 对数率允许任意实数，普通率需要保持正数
        if not self.params.get("log_input", True):
            prediction = prediction.abs().clamp_min(self.params.get("eps", 1e-8))  # 非对数模式把随机输入转换为合法率
        return {"loss": self.loss(prediction, target)}  # 计算逐元素或聚合泊松损失


@node(opcode="gaussian_nll_loss", label="高斯负对数似然", ports={"input": {"input": "预测均值", "target": "目标值", "variance": "预测方差"}, "output": {"loss": "损失值"}}, params={**reductionParams, "full": {"label": "加入常数项", "type": "bool", "value": False}, "eps": {"label": "最小方差", "type": "float", "value": 0.000001, "range": [1e-12, 1]}}, description="同时评价预测均值和不确定性方差")
class GaussianNLLLossNode(BaseNode):
    def build(self):
        self.loss = nn.GaussianNLLLoss(full=self.params.get("full", False), eps=self.params.get("eps", 1e-6), reduction=self.params.get("reduction", "mean"))  # 创建高斯似然损失

    def compute(self, input):
        variance = input.get("variance").abs().clamp_min(self.params.get("eps", 1e-6))  # 方差必须严格为正
        return {"loss": self.loss(input.get("input"), input.get("target"), variance)}  # 同形均值、目标和方差直接计算


@node(opcode="kl_div_loss", label="KL散度损失", ports={"input": {"input": "输入分布", "target": "目标分布"}, "output": {"loss": "损失值"}}, params={"reduction": {"label": "聚合方式", "type": "enum", "value": "batchmean", "options": {"batchmean": "按批平均", "mean": "全部平均", "sum": "总和", "none": "无聚合"}}, "log_target": {"label": "目标为对数概率", "type": "bool", "value": False}}, description="衡量两个概率分布之间的信息差异")
class KLDivLossNode(BaseNode):
    def build(self):
        self.loss = nn.KLDivLoss(reduction=self.params.get("reduction", "batchmean"), log_target=self.params.get("log_target", False))  # 创建标准 KL 散度

    def compute(self, input):
        logPrediction = torch.log_softmax(input.get("input"), dim=-1)  # 任意随机输入转换为合法对数概率
        targetLog = torch.log_softmax(input.get("target"), dim=-1)  # 目标先统一为归一化对数概率
        target = targetLog if self.params.get("log_target", False) else targetLog.exp()  # 按配置提供概率或对数概率目标
        return {"loss": self.loss(logPrediction, target)}  # 沿最后维比较分布差异


@node(opcode="margin_ranking_loss", label="边际排序损失", ports={"input": {"x": "分数1", "y": "分数2", "target": "排序目标"}, "output": {"loss": "损失值"}}, params={**reductionParams, "margin": {"label": "最小间隔", "type": "float", "value": 0.0, "range": [0, 100]}}, description="约束两个分数按正负目标保持指定顺序")
class MarginRankingLossNode(BaseNode):
    def build(self):
        self.loss = nn.MarginRankingLoss(margin=self.params.get("margin", 0.0), reduction=self.params.get("reduction", "mean"))  # 创建成对排序损失

    def compute(self, input):
        target = torch.where(input.get("target") >= 0, torch.ones_like(input.get("target")), -torch.ones_like(input.get("target")))  # 随机目标转换为PyTorch要求的正负1
        return {"loss": self.loss(input.get("x"), input.get("y"), target)}  # 同形分数逐元素评价排序


@node(opcode="triplet_margin_loss", label="三元组边际损失", ports={"input": {"anchor": "锚点", "positive": "正样本", "negative": "负样本"}, "output": {"loss": "损失值"}}, params={**reductionParams, "margin": {"label": "最小间隔", "type": "float", "value": 1.0, "range": [0.000001, 100]}, "p": {"label": "距离阶数", "type": "float", "value": 2.0, "range": [0.001, 100]}, "eps": {"label": "防零极小值", "type": "float", "value": 0.000001, "range": [1e-12, 1]}, "swap": {"label": "交换距离", "type": "bool", "value": False}}, description="让锚点更接近正样本并远离负样本")
class TripletMarginLossNode(BaseNode):
    def build(self):
        self.loss = nn.TripletMarginLoss(margin=self.params.get("margin", 1.0), p=self.params.get("p", 2.0), eps=self.params.get("eps", 1e-6), swap=self.params.get("swap", False), reduction=self.params.get("reduction", "mean"))  # 创建欧氏或p范数三元组损失

    def compute(self, input):
        return {"loss": self.loss(input.get("anchor"), input.get("positive"), input.get("negative"))}  # 最后一维作为嵌入特征


@node(opcode="cosine_embedding_loss", label="余弦嵌入损失", ports={"input": {"x": "嵌入1", "y": "嵌入2", "target": "关系目标"}, "output": {"loss": "损失值"}}, params={**reductionParams, "margin": {"label": "负样本边际", "type": "float", "value": 0.0, "range": [-1, 1]}}, description="按关系目标拉近或推远两个嵌入方向")
class CosineEmbeddingLossNode(BaseNode):
    def build(self):
        self.loss = nn.CosineEmbeddingLoss(margin=self.params.get("margin", 0.0), reduction=self.params.get("reduction", "mean"))  # 创建余弦方向损失

    def compute(self, input):
        first = input.get("x").reshape(-1, input.get("x").shape[-1])  # 合并前置维形成嵌入样本批
        second = input.get("y").reshape(-1, input.get("y").shape[-1])  # 第二输入使用相同样本划分
        rawTarget = input.get("target").reshape(-1, input.get("target").shape[-1]).mean(dim=-1)  # 每个嵌入生成一个关系目标
        target = torch.where(rawTarget >= 0, torch.ones_like(rawTarget), -torch.ones_like(rawTarget))  # 关系严格转换为正负1
        return {"loss": self.loss(first, second, target)}  # 比较每对最后维嵌入方向
