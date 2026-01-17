# ================= 激活函数节点组 =================
import torch                                                                    # 导入 PyTorch 库
import torch.nn.functional as F                                                 # 导入函数式接口，用于激活函数计算
from decorators import category, node                                           # 导入装饰器，用于注册分类和节点

@category(                                                                      # 注册"激活函数"分类
    id="activations",                                                           # 分类唯一标识
    name="激活函数",                                                              # 分类显示名称
    color="#FFB6C1",                                                            # 分类主题颜色
    icon="base64…"                                                              # 分类图标
)
def activations_category():                                                     # 激活函数分类定义函数
    pass                                                                        # 仅作为装饰器载体

@node(                                                                          # 注册"ReLU"节点
    opcode="relu",                                                              # 算子唯一标识
    name="ReLU 激活",                                                            # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={}                                                                   # 无参数
)
def relu_node():                                                                # ReLU 节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # ReLU 不改变形状，直接返回输入形状

    def build(params):                                                          # 构建层
        return None                                                             # ReLU 运算不需要实例化层

    def compute(x, layer):                                                      # 执行计算（引擎自动解包单输入为张量）
        return F.relu(x)                                                        # 执行张量 ReLU 激活

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"Sigmoid"节点
    opcode="sigmoid",                                                           # 算子唯一标识
    name="Sigmoid 激活",                                                         # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={}                                                                   # 无参数
)
def sigmoid_node():                                                             # Sigmoid 节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # Sigmoid 不改变形状，直接返回输入形状

    def build(params):                                                          # 构建层
        return None                                                             # Sigmoid 运算不需要实例化层

    def compute(x, layer):                                                      # 执行计算（引擎自动解包单输入为张量）
        return torch.sigmoid(x)                                                 # 执行张量 Sigmoid 激活

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"Tanh"节点
    opcode="tanh",                                                              # 算子唯一标识
    name="Tanh 激活",                                                            # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={}                                                                   # 无参数
)
def tanh_node():                                                                # Tanh 节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # Tanh 不改变形状

    def build(params):                                                          # 构建层
        return None                                                             # Tanh 运算不需要实例化层

    def compute(x, layer):                                                      # 执行计算（引擎自动解包单输入为张量）
        return torch.tanh(x)                                                    # 执行张量 Tanh 激活

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"LeakyReLU"节点
    opcode="leaky_relu",                                                        # 算子唯一标识
    name="LeakyReLU 激活",                                                       # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={"negative_slope": 0.01}                                             # 负斜率参数
)
def leaky_relu_node():                                                          # LeakyReLU 节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # LeakyReLU 不改变形状

    def build(params):                                                          # 构建层
        return params.get("negative_slope", 0.01)                               # 返回负斜率参数

    def compute(x, layer):                                                      # 执行计算（引擎自动解包单输入为张量）
        return F.leaky_relu(x, negative_slope=layer)                            # 执行 LeakyReLU 激活

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"Softmax"节点
    opcode="softmax",                                                           # 算子唯一标识
    name="Softmax 激活",                                                         # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={"dim": -1}                                                          # 作用维度参数
)
def softmax_node():                                                             # Softmax 节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # Softmax 不改变形状

    def build(params):                                                          # 构建层
        return params.get("dim", -1)                                            # 返回作用维度

    def compute(x, layer):                                                      # 执行计算（引擎自动解包单输入为张量）
        return F.softmax(x, dim=layer)                                          # 执行 Softmax 激活

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"GELU"节点
    opcode="gelu",                                                              # 算子唯一标识
    name="GELU 激活",                                                            # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={}                                                                   # 无参数
)
def gelu_node():                                                                # GELU 节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # GELU 不改变形状

    def build(params):                                                          # 构建层
        return None                                                             # GELU 运算不需要实例化层

    def compute(x, layer):                                                      # 执行计算（引擎自动解包单输入为张量）
        return F.gelu(x)                                                        # 执行 GELU 激活

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"SiLU/Swish"节点
    opcode="silu",                                                              # 算子唯一标识
    name="SiLU/Swish 激活",                                                      # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={}                                                                   # 无参数
)
def silu_node():                                                                # SiLU 节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # SiLU 不改变形状

    def build(params):                                                          # 构建层
        return None                                                             # SiLU 运算不需要实例化层

    def compute(x, layer):                                                      # 执行计算（引擎自动解包单输入为张量）
        return F.silu(x)                                                        # 执行 SiLU 激活

    return infer, build, compute                                                # 返回逻辑函数
