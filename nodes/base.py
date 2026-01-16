# ================= 基础节点组 =================
import torch                                                                    # 导入 PyTorch 库，用于张量操作

from decorators import category, node                                           # 导入装饰器，用于注册分类和节点

@category(                                                                      # 注册“基础”分类
    id="basic",                                                                 # 分类唯一标识
    name="基础",                                                                # 分类显示名称
    color="#8B92E5",                                                            # 分类主题颜色
    icon="base64…"                                                              # 分类图标
)
def basic_category():                                                           # 分类定义函数
    pass                                                                        # 仅作为装饰器载体

@node(                                                                          # 注册“输入”节点
    opcode="input",                                                             # 算子唯一标识
    name="输入",                                                                # 节点显示名称
    ports={"out": ["out"]},                                                     # 定义输出端口
    params={"输出维度": [1, 10]},                                                # 定义节点参数
)
def input_node():                                                               # 输入节点定义
    def infer(input_shapes, params):                                            # 推断输出形状的函数
        return {"out": params["输出维度"]}                                       # 返回参数中定义的维度

    def build(input_shapes, params):                                            # 构建 PyTorch 层的函数
        return None                                                             # 输入节点不需要实例化层

    def compute(inputs, layer):                                                 # 执行计算的函数
        return None                                                             # 输入节点由引擎直接透传数据

    return infer, build, compute                                                # 返回核心逻辑函数元组

@node(                                                                          # 注册“输出”节点
    opcode="output",                                                            # 算子唯一标识
    name="输出",                                                                # 节点显示名称
    ports={"in": ["in"]},                                                       # 定义输入端口
    params={},                                                                  # 无参数
)
def output_node():                                                              # 输出节点定义
    def infer(input_shapes, params):                                            # 推断输出形状的函数
        return None                                                             # 输出节点没有进一步输出

    def build(input_shapes, params):                                            # 构建 PyTorch 层的函数
        return None                                                             # 输出节点不需要实例化层

    def compute(inputs, layer):                                                 # 执行计算的函数
        return inputs.get("in")                                                 # 直接返回输入的数据

    return infer, build, compute                                                # 返回核心逻辑函数元组
