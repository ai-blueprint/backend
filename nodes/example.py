# ================= 示例节点组 =================
import torch                                                                    # 导入 PyTorch 库，用于神经网络层定义
from decorators import category, node                                           # 导入装饰器，用于注册分类和节点

@category(                                                                      # 注册“示例”分类
    id="example",                                                               # 分类唯一标识
    name="示例",                                                                # 分类显示名称
    color="#82CBFA",                                                            # 分类主题颜色
    icon="base64…"                                                              # 分类图标
)
def example_category():                                                         # 示例分类定义函数
    pass                                                                        # 仅作为装饰器载体

@node(                                                                          # 注册“全连接层”节点
    opcode="linear",                                                            # 算子唯一标识
    name="全连接层",                                                              # 节点显示名称
    ports={"in": ["x"], "out": ["y"]},                                          # 定义输入输出端口
    params={"输出特征数": 128, "bias": True},                                       # 定义层参数
)
def linear_node():                                                              # 全连接层节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        output_shape = list(input_shapes.get("x", []))                          # 获取输入 X 的形状
        if output_shape:                                                        # 如果形状有效
            output_shape[-1] = params["输出特征数"]                              # 修改最后一个维度为输出特征数
        return {"y": output_shape}                                              # 返回输出端口的形状字典

    def build(input_shapes, params):                                            # 构建 PyTorch 层
        in_feat = input_shapes.get("x", [0, 0])[-1]                             # 从输入形状中获取输入特征数
        out_feat = params["输出特征数"]                                           # 从参数中获取输出特征数
        return torch.nn.Linear(in_feat, out_feat, bias=params["bias"])          # 实例化并返回 PyTorch 线性层

    def compute(inputs, layer):                                                 # 执行计算
        return layer(inputs)                                                    # 直接调用线性层进行前向计算

    return infer, build, compute                                                # 返回逻辑函数元组
