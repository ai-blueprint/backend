"""
nodes/base.py - 基础节点组

提供基础的输入/输出/调试节点
"""

from decorators import category, node  # 从装饰器模块导入category和node装饰器
import torch  # 导入torch模块
from torch import nn  # 导入torch.nn模块

# ==================== 分类定义 ====================

category(  # 调用category注册分类
    id="basic",  # 分类唯一标识
    label="基础",  # 分类显示名称
    color="#8B92E5",  # 分类颜色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 使用node装饰器注册节点
    opcode="input",  # 节点操作码，唯一标识
    label="输入",  # 节点显示名称
    ports={"in": [], "out": ["out"]},  # 输入输出端口定义
    params={"输出形状": [2, 4, 6]},  # 节点参数定义
)
def inputNode():
    """
    输入节点 - 蓝图的入口点
    """

    def infer(inputShapes, params):
        """
        形状推断函数 - 根据参数返回输出形状
        """
        return {"out": params["输出形状"]}  # 返回参数中定义的输出形状

    def build(shape, params):
        """
        构建层函数 - 输入节点不需要构建层
        """
        # 这里把layer作为随机数
        # 首先就是将params["输出形状"]转换成tensor的维度，因为params["输出形状"]是数组
        dim = torch.Size(params["输出形状"])
        layer = torch.rand(dim)  # 生成随机数
        return layer

    def compute(inputs, layer):
        """
        计算函数 - 输入节点直接返回空，由引擎透传数据
        """
        # 根据输出形状来生成随机数据
        return {"out": layer}  

    return {
        "infer": infer,
        "build": build,
        "compute": compute,
    }  # 返回包含三个函数的字典


@node(  # 使用node装饰器注册节点
    opcode="output",  # 节点操作码，唯一标识
    label="输出",  # 节点显示名称
    ports={"in": ["in"], "out": []},  # 输入输出端口定义
    params={},  # 节点参数定义
)
def outputNode():
    """
    输出节点 - 蓝图的出口点，直接透传输入数据
    """

    def infer(inputShapes, params):
        """
        形状推断函数 - 透传输入形状
        """
        return {"out": inputShapes.get("in")}  # 透传输入的形状

    def build(shape, params):
        """
        构建层函数 - 输出节点不需要构建层
        """
        return None  # 输出节点不需要层

    def compute(inputs, layer):
        """
        计算函数 - 直接透传输入数据
        """
        return {"out": inputs.get("in")}  # 透传输入数据

    return {
        "infer": infer,
        "build": build,
        "compute": compute,
    }  # 返回包含三个函数的字典


@node(  # 使用node装饰器注册节点
    opcode="constant",  # 节点操作码，唯一标识
    label="常量",  # 节点显示名称
    ports={"in": [], "out": ["out"]},  # 输入输出端口定义
    params={"value": 0},  # 节点参数定义
)
def constantNode():
    """
    常量节点 - 输出一个固定值
    """
    import torch  # 导入torch

    def infer(inputShapes, params):
        """
        形状推断函数 - 常量输出形状为[1]
        """
        return {"out": [1]}  # 常量形状固定为[1]

    def build(shape, params):
        """
        构建层函数 - 创建常量张量
        """
        value = params.get("value", 0)  # 获取常量值
        return torch.tensor([value], dtype=torch.float32)  # 返回张量

    def compute(inputs, layer):
        """
        计算函数 - 直接返回构建的常量张量
        """
        return {"out": layer}  # 返回层（即常量张量）

    return {
        "infer": infer,
        "build": build,
        "compute": compute,
    }  # 返回包含三个函数的字典


@node(  # 使用node装饰器注册节点
    opcode="debug",  # 节点操作码，唯一标识
    label="调试输出",  # 节点显示名称
    ports={"in": ["x"], "out": ["out"]},  # 输入输出端口定义
    params={"label": "debug"},  # 节点参数定义
)
def debugNode():
    """
    调试节点 - 打印输入数据并透传
    """

    def infer(inputShapes, params):
        """
        形状推断函数 - 透传输入形状
        """
        return {"out": inputShapes.get("x")}  # 透传x的形状

    def build(shape, params):
        """
        构建层函数 - 返回调试标签
        """
        return params.get("label", "debug")  # 返回标签字符串

    def compute(inputs, layer):
        """
        计算函数 - 打印调试信息并透传
        """
        x = inputs.get("x")  # 获取输入x
        label = layer  # 层就是标签字符串

        shapeStr = x.shape if hasattr(x, "shape") else "N/A"  # 获取形状字符串
        dtypeStr = x.dtype if hasattr(x, "dtype") else type(x)  # 获取类型字符串
        print(f"🔍 [{label}] shape={shapeStr}, dtype={dtypeStr}")  # 打印调试信息

        return {"out": x}  # 透传x

    return {
        "infer": infer,
        "build": build,
        "compute": compute,
    }  # 返回包含三个函数的字典
