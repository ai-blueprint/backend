# ================= 数学运算节点组 =================
import torch                                                                    # 导入 PyTorch 库，用于张量数学运算
from decorators import category, node                                           # 导入装饰器，用于注册分类和节点

@category(                                                                      # 注册“数学运算”分类
    id="math",                                                                  # 分类唯一标识
    name="数学运算",                                                              # 分类显示名称
    color="#FFB6C1",                                                            # 分类主题颜色
    icon="base64…"                                                              # 分类图标
)
def math_category():                                                            # 数学分类定义函数
    pass                                                                        # 仅作为装饰器载体

@node(                                                                          # 注册“加法”节点
    opcode="add",                                                               # 算子唯一标识
    name="加法 (+)",                                                             # 节点显示名称
    ports={"in": ["x", "y"], "out": ["result"]},                               # 定义输入输出端口
    params={}                                                                   # 无参数
)
def add_node():                                                                 # 加法节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        return input_shapes.get("x")                                            # 返回第一个输入的形状

    def build(params):                                                          # 构建层
        return None                                                             # 纯数学运算不需要层

    def compute(inputs, layer):                                                 # 执行计算
        return inputs["x"] + inputs["y"]                                        # 执行张量加法

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册“矩阵乘法”节点
    opcode="matmul",                                                            # 算子唯一标识
    name="矩阵乘法 (@)",                                                          # 节点显示名称
    ports={"in": ["x", "y"], "out": ["result"]},                               # 定义输入输出端口
    params={}                                                                   # 无参数
)
def matmul_node():                                                              # 矩阵乘法节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        shape_x = input_shapes.get("x", [0, 0])                                 # 获取矩阵 X 的形状
        shape_y = input_shapes.get("y", [0, 0])                                 # 获取矩阵 Y 的形状
        return [shape_x[0], shape_y[1]]                                         # 返回矩阵乘法后的形状 [M, P]

    def build(params):                                                          # 构建层
        return None                                                             # 矩阵乘法不需要层

    def compute(inputs, layer):                                                 # 执行计算
        return torch.matmul(inputs["x"], inputs["y"])                           # 执行张量矩阵乘法

    return infer, build, compute                                                # 返回逻辑函数

@node(                                                                          # 注册"求和"节点
    opcode="sum",                                                               # 算子唯一标识
    name="求和 (Σ)",                                                             # 节点显示名称
    ports={"in": ["x"], "out": ["result"]},                                     # 定义输入输出端口
    params={"dim": None, "keepdim": False}                                      # 定义求和维度和是否保持维度
)
def sum_node():                                                                 # 求和节点定义
    def infer(input_shapes, params):                                            # 推断输出形状
        if params.get("dim") is None: return [1]                                # 全局求和返回标量形状
        shape = list(input_shapes.get("x", []))                                 # 复制输入形状
        if not params.get("keepdim"): shape.pop(params["dim"])                  # 如果不保持维度，则移除该维度
        return shape                                                            # 返回计算后的形状

    def build(params):                                                          # 构建层
        return {"dim": params.get("dim"), "keepdim": params.get("keepdim", False)} # 返回参数字典供 compute 使用

    def compute(inputs, layer):                                                 # 执行计算
        dim = layer.get("dim") if layer else None                               # 从 layer 中获取维度参数
        keepdim = layer.get("keepdim", False) if layer else False               # 从 layer 中获取是否保持维度
        return torch.sum(inputs["x"], dim=dim, keepdim=keepdim)                 # 执行张量求和

    return infer, build, compute                                                # 返回逻辑函数
