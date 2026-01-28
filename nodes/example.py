"""
nodes/example.py - 示例节点定义

这个文件展示了如何定义节点分类和节点
当loader.loadAll()被调用时，这个文件会被自动导入
里面的@category和@node装饰器会自动注册分类和节点
"""

from decorators import category, node  # 从装饰器模块导入category和node装饰器
from torch import nn  # 导入torch.nn模块

# 定义一个示例分类
category(  # 调用category装饰器注册分类
    id="example_category",  # 分类唯一标识
    label="示例节点定义",  # 分类显示名称
    color="#FFB6C1",  # 分类颜色，粉色
    icon="",  # 分类图标，可以是base64格式字符串
)


@node(  # 使用node装饰器注册节点
    opcode="example_node",  # 节点操作码，唯一标识
    label="示例节点",  # 节点显示名称
    ports={"in": ["x", "y"], "out": ["result"]},  # 输入输出端口定义
    params={"形状参数": [2, 4], "数字参数": 1, "布尔参数": False},  # 节点参数定义
)
def exampleNode():
    """
    示例节点 - 展示节点的定义方法模版
    """

    def infer(inputShapes, params):
        """
        形状推断函数 - 根据输入形状和参数推断输出形状
        """
        xShape = inputShapes.get("x", None)  # 获取输入x的形状
        yShape = inputShapes.get("y", None)  # 获取输入y的形状
        # 确保两个形状一样
        if xShape != yShape:
            raise ValueError("输入x和y的形状必须相同")
        return {"result": params["形状参数"]}

    def build(shape, params):
        """
        构建层函数 - 根据形状和参数构建层实例
        """
        layer = nn.linear(shape["result"], params["形状参数"], bias=params["布尔参数"])
        return layer

    def compute(inputs, layer):
        """
        计算函数 - 执行实际的计算
        """
        x = inputs["x"]  # 另一种获取方式
        y = inputs["y"]
        result = layer(x) + layer(y)  # 使用层实例进行计算

        return {"result": result}  # 返回结果字典

    return {
        "infer": infer,
        "build": build,
        "compute": compute,
    }  # 返回包含三个函数的字典
