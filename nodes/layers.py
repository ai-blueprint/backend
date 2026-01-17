# ================= 神经网络层节点组 =================
import torch                                                                    # 导入 PyTorch 库，用于神经网络层定义
import torch.nn as nn                                                           # 导入神经网络模块
from decorators import category, node                                           # 导入装饰器，用于注册分类和节点

@category(                                                                      # 注册"神经网络层"分类
    id="layers",                                                                # 分类唯一标识
    name="神经网络层",                                                            # 分类显示名称
    color="#82CBFA",                                                            # 分类主题颜色
    icon="base64…"                                                              # 分类图标
)
def layers_category():                                                          # 分类定义函数
    pass                                                                        # 仅作为装饰器载体

@node(                                                                          # 注册"线性层"节点
    opcode="linear",                                                            # 算子唯一标识
    name="全连接层 (Linear)",                                                     # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={"in_features": 128, "out_features": 64, "bias": True},              # 定义层参数
)
def linear_layer():                                                             # 全连接层节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape = list(input_shapes.get("x", [1, 128]))                           # 获取输入形状
        shape[-1] = params["out_features"]                                      # 修改最后一个维度为输出特征数
        return {"out": shape}                                                   # 返回输出形状字典

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        in_feat = params["in_features"]                                         # 输入特征数
        out_feat = params["out_features"]                                       # 输出特征数
        return nn.Linear(in_feat, out_feat, bias=params.get("bias", True))      # 实例化线性层

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x)                                                         # 直接调用层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"卷积层"节点
    opcode="conv2d",                                                            # 算子唯一标识
    name="卷积层 (Conv2d)",                                                       # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={                                                                    # 定义层参数
        "in_channels": 3,                                                       # 输入通道数
        "out_channels": 64,                                                     # 输出通道数
        "kernel_size": 3,                                                       # 卷积核大小
        "stride": 1,                                                            # 步长
        "padding": 1,                                                           # 填充
    },
)
def conv2d_layer():                                                             # 卷积层节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape = list(input_shapes.get("x", [1, 3, 32, 32]))                     # 获取输入形状 [B, C, H, W]
        out_c = params["out_channels"]                                          # 输出通道数
        k = params["kernel_size"]                                               # 卷积核大小
        s = params["stride"]                                                    # 步长
        p = params["padding"]                                                   # 填充
        h_out = (shape[2] + 2 * p - k) // s + 1                                 # 计算输出高度
        w_out = (shape[3] + 2 * p - k) // s + 1                                 # 计算输出宽度
        return {"out": [shape[0], out_c, h_out, w_out]}                         # 返回输出形状

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.Conv2d(                                                       # 实例化卷积层
            in_channels=params["in_channels"],                                  # 输入通道数
            out_channels=params["out_channels"],                                # 输出通道数
            kernel_size=params["kernel_size"],                                  # 卷积核大小
            stride=params["stride"],                                            # 步长
            padding=params["padding"],                                          # 填充
        )

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x)                                                         # 直接调用层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"批归一化层"节点
    opcode="batchnorm2d",                                                       # 算子唯一标识
    name="批归一化 (BatchNorm2d)",                                                # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={"num_features": 64},                                                # 定义层参数
)
def batchnorm2d_layer():                                                        # 批归一化节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"out": input_shapes.get("x")}                                   # 形状不变

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.BatchNorm2d(params["num_features"])                           # 实例化批归一化层

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x)                                                         # 直接调用层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"Dropout层"节点
    opcode="dropout",                                                           # 算子唯一标识
    name="Dropout",                                                             # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={"p": 0.5},                                                          # 定义丢弃概率
)
def dropout_layer():                                                            # Dropout节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return {"out": input_shapes.get("x")}                                   # 形状不变

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.Dropout(p=params.get("p", 0.5))                               # 实例化Dropout层

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x)                                                         # 直接调用层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"最大池化层"节点
    opcode="maxpool2d",                                                         # 算子唯一标识
    name="最大池化 (MaxPool2d)",                                                  # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={"kernel_size": 2, "stride": 2, "padding": 0},                       # 定义层参数
)
def maxpool2d_layer():                                                          # 最大池化节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape = list(input_shapes.get("x", [1, 64, 32, 32]))                    # 获取输入形状
        k = params["kernel_size"]                                               # 池化核大小
        s = params["stride"]                                                    # 步长
        p = params["padding"]                                                   # 填充
        h_out = (shape[2] + 2 * p - k) // s + 1                                 # 计算输出高度
        w_out = (shape[3] + 2 * p - k) // s + 1                                 # 计算输出宽度
        return {"out": [shape[0], shape[1], h_out, w_out]}                      # 返回输出形状

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.MaxPool2d(                                                    # 实例化最大池化层
            kernel_size=params["kernel_size"],                                  # 池化核大小
            stride=params["stride"],                                            # 步长
            padding=params["padding"],                                          # 填充
        )

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x)                                                         # 直接调用层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"平均池化层"节点
    opcode="avgpool2d",                                                         # 算子唯一标识
    name="平均池化 (AvgPool2d)",                                                  # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={"kernel_size": 2, "stride": 2, "padding": 0},                       # 定义层参数
)
def avgpool2d_layer():                                                          # 平均池化节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape = list(input_shapes.get("x", [1, 64, 32, 32]))                    # 获取输入形状
        k = params["kernel_size"]                                               # 池化核大小
        s = params["stride"]                                                    # 步长
        p = params["padding"]                                                   # 填充
        h_out = (shape[2] + 2 * p - k) // s + 1                                 # 计算输出高度
        w_out = (shape[3] + 2 * p - k) // s + 1                                 # 计算输出宽度
        return {"out": [shape[0], shape[1], h_out, w_out]}                      # 返回输出形状

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.AvgPool2d(                                                    # 实例化平均池化层
            kernel_size=params["kernel_size"],                                  # 池化核大小
            stride=params["stride"],                                            # 步长
            padding=params["padding"],                                          # 填充
        )

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x)                                                         # 直接调用层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"展平层"节点
    opcode="flatten",                                                           # 算子唯一标识
    name="展平 (Flatten)",                                                       # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={"start_dim": 1, "end_dim": -1},                                     # 定义展平维度范围
)
def flatten_layer():                                                            # 展平节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape = list(input_shapes.get("x", [1, 64, 8, 8]))                      # 获取输入形状
        start = params.get("start_dim", 1)                                      # 起始维度
        batch_size = shape[0]                                                   # 批次大小
        flat_size = 1                                                           # 展平后的大小
        for s in shape[start:]:                                                 # 遍历需要展平的维度
            flat_size *= s                                                      # 累乘
        return {"out": [batch_size, flat_size]}                                 # 返回输出形状

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.Flatten(                                                      # 实例化展平层
            start_dim=params.get("start_dim", 1),                               # 起始维度
            end_dim=params.get("end_dim", -1),                                  # 结束维度
        )

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x)                                                         # 直接调用层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"LSTM层"节点
    opcode="lstm",                                                              # 算子唯一标识
    name="LSTM",                                                                # 节点显示名称
    ports={"in": ["x"], "out": ["out", "hidden"]},                              # 定义输入输出端口
    params={                                                                    # 定义层参数
        "input_size": 128,                                                      # 输入特征数
        "hidden_size": 256,                                                     # 隐藏层大小
        "num_layers": 1,                                                        # 层数
        "batch_first": True,                                                    # 批次优先
    },
)
def lstm_layer():                                                               # LSTM节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape = list(input_shapes.get("x", [1, 10, 128]))                       # 获取输入形状 [B, T, F]
        hidden = params["hidden_size"]                                          # 隐藏层大小
        return {"out": [shape[0], shape[1], hidden], "hidden": [shape[0], hidden]}

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.LSTM(                                                         # 实例化LSTM层
            input_size=params["input_size"],                                    # 输入特征数
            hidden_size=params["hidden_size"],                                  # 隐藏层大小
            num_layers=params["num_layers"],                                    # 层数
            batch_first=params.get("batch_first", True),                        # 批次优先
        )

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        out, (hn, cn) = layer(x)                                                # LSTM前向计算
        return {"out": out, "hidden": hn[-1]}                                   # 返回输出和最后隐藏状态

    return infer, build, compute                                                # 返回逻辑函数元组

@node(                                                                          # 注册"Embedding层"节点
    opcode="embedding",                                                         # 算子唯一标识
    name="嵌入层 (Embedding)",                                                    # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},                                        # 定义输入输出端口
    params={"num_embeddings": 10000, "embedding_dim": 128},                     # 定义层参数
)
def embedding_layer():                                                          # Embedding节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape = list(input_shapes.get("x", [1, 10]))                            # 获取输入形状
        shape.append(params["embedding_dim"])                                   # 添加嵌入维度
        return {"out": shape}                                                   # 返回输出形状

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        return nn.Embedding(                                                    # 实例化嵌入层
            num_embeddings=params["num_embeddings"],                            # 词表大小
            embedding_dim=params["embedding_dim"],                              # 嵌入维度
        )

    def compute(x, layer):                                                      # 执行计算（引擎已解包输入）
        return layer(x.long())                                                  # 转为长整型后嵌入

    return infer, build, compute                                                # 返回逻辑函数元组
