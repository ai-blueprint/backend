# ================= 损失函数节点组 =================
import torch                                                                    # 导入 PyTorch 库
import torch.nn as nn                                                           # 导入神经网络模块
import torch.nn.functional as F                                                 # 导入函数式接口
from decorators import category, node                                           # 导入装饰器

@category(                                                                      # 注册"损失函数"分类
    id="loss",                                                                  # 分类唯一标识
    name="损失函数",                                                              # 分类显示名称
    color="#FF6B6B",                                                            # 分类主题颜色
    icon="base64…"                                                              # 分类图标
)
def loss_category():                                                            # 分类定义函数
    pass                                                                        # 仅作为装饰器载体

@node(                                                                          # 注册"交叉熵损失"节点
    opcode="cross_entropy",                                                     # 算子唯一标识
    name="交叉熵损失 (CrossEntropy)",                                             # 节点显示名称
    ports={"in": ["input", "target"], "out": ["loss"]},                         # 定义输入输出端口
    params={}                                                                   # 无参数
)
def cross_entropy_loss():                                                       # 交叉熵损失节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"loss": [1]}                                                    # 损失值是标量

    def build(params):                                                          # 构建层
        return nn.CrossEntropyLoss()                                            # 实例化交叉熵损失

    def compute(inputs, layer):                                                 # 执行计算
        pred = inputs["input"]                                                  # 获取预测值
        target = inputs["target"]                                               # 获取目标值
        if target.dtype != torch.long:                                          # 如果目标不是长整型
            target = target.long()                                              # 转换为长整型
        return layer(pred, target)                                              # 计算损失

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"MSE损失"节点
    opcode="mse_loss",                                                          # 算子唯一标识
    name="均方误差损失 (MSE)",                                                     # 节点显示名称
    ports={"in": ["input", "target"], "out": ["loss"]},                         # 定义输入输出端口
    params={"reduction": "mean"}                                                # 归约方式参数
)
def mse_loss():                                                                 # MSE损失节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"loss": [1]}                                                    # 损失值是标量

    def build(params):                                                          # 构建层
        reduction = params.get("reduction", "mean")                             # 获取归约方式
        return nn.MSELoss(reduction=reduction)                                  # 实例化MSE损失

    def compute(inputs, layer):                                                 # 执行计算
        pred = inputs["input"]                                                  # 获取预测值
        target = inputs["target"]                                               # 获取目标值
        return layer(pred, target)                                              # 计算损失

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"L1损失"节点
    opcode="l1_loss",                                                           # 算子唯一标识
    name="L1损失 (MAE)",                                                         # 节点显示名称
    ports={"in": ["input", "target"], "out": ["loss"]},                         # 定义输入输出端口
    params={"reduction": "mean"}                                                # 归约方式参数
)
def l1_loss():                                                                  # L1损失节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"loss": [1]}                                                    # 损失值是标量

    def build(params):                                                          # 构建层
        reduction = params.get("reduction", "mean")                             # 获取归约方式
        return nn.L1Loss(reduction=reduction)                                   # 实例化L1损失

    def compute(inputs, layer):                                                 # 执行计算
        pred = inputs["input"]                                                  # 获取预测值
        target = inputs["target"]                                               # 获取目标值
        return layer(pred, target)                                              # 计算损失

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"二元交叉熵损失"节点
    opcode="bce_loss",                                                          # 算子唯一标识
    name="二元交叉熵损失 (BCE)",                                                   # 节点显示名称
    ports={"in": ["input", "target"], "out": ["loss"]},                         # 定义输入输出端口
    params={}                                                                   # 无参数
)
def bce_loss():                                                                 # BCE损失节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"loss": [1]}                                                    # 损失值是标量

    def build(params):                                                          # 构建层
        return nn.BCELoss()                                                     # 实例化BCE损失

    def compute(inputs, layer):                                                 # 执行计算
        pred = inputs["input"]                                                  # 获取预测值
        target = inputs["target"]                                               # 获取目标值
        return layer(pred, target.float())                                      # 计算损失

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"BCE With Logits损失"节点
    opcode="bce_with_logits",                                                   # 算子唯一标识
    name="BCE With Logits",                                                     # 节点显示名称
    ports={"in": ["input", "target"], "out": ["loss"]},                         # 定义输入输出端口
    params={}                                                                   # 无参数
)
def bce_with_logits_loss():                                                     # BCE With Logits损失节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"loss": [1]}                                                    # 损失值是标量

    def build(params):                                                          # 构建层
        return nn.BCEWithLogitsLoss()                                           # 实例化损失函数

    def compute(inputs, layer):                                                 # 执行计算
        pred = inputs["input"]                                                  # 获取预测值
        target = inputs["target"]                                               # 获取目标值
        return layer(pred, target.float())                                      # 计算损失

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"NLL损失"节点
    opcode="nll_loss",                                                          # 算子唯一标识
    name="负对数似然损失 (NLL)",                                                   # 节点显示名称
    ports={"in": ["input", "target"], "out": ["loss"]},                         # 定义输入输出端口
    params={}                                                                   # 无参数
)
def nll_loss():                                                                 # NLL损失节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"loss": [1]}                                                    # 损失值是标量

    def build(params):                                                          # 构建层
        return nn.NLLLoss()                                                     # 实例化NLL损失

    def compute(inputs, layer):                                                 # 执行计算
        pred = inputs["input"]                                                  # 获取预测值（需要是log_softmax后的）
        target = inputs["target"]                                               # 获取目标值
        if target.dtype != torch.long:                                          # 如果目标不是长整型
            target = target.long()                                              # 转换为长整型
        return layer(pred, target)                                              # 计算损失

    return infer, build, compute                                                # 返回逻辑函数
