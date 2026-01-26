"""
nodes/example.py - 示例节点定义

用法：
    这个文件展示了如何定义节点分类和节点
    当loader.loadAll()被调用时，这个文件会被自动导入
    里面的@category和@node装饰器会自动注册分类和节点
    
示例：
    见下方代码
"""

from decorators import category, node  # 从装饰器模块导入category和node装饰器


# 定义一个示例分类
category(  # 调用category装饰器注册分类
    id="example_category",  # 分类唯一标识
    label="示例节点定义",  # 分类显示名称
    color="#FFB6C1",  # 分类颜色，粉色
    icon=""  # 分类图标，可以是base64格式字符串
)


@node(  # 使用node装饰器注册节点
    opcode="example_node",  # 节点操作码，唯一标识
    label="示例节点",  # 节点显示名称
    ports={"in": ["x", "y"], "out": ["result"]},  # 输入输出端口定义
    params={"数字参数": 1, "布尔参数": False}  # 节点参数定义
)
def exampleNode():
    """
    示例节点 - 展示节点的基本结构
    
    用法：
        这是一个示例节点，展示了infer、build、compute三个函数的写法
        
    示例：
        输入两个数x和y，输出它们的和
    """
    
    def infer(inputs, params):
        """
        形状推断函数 - 根据输入形状和参数推断输出形状
        
        用法：
            shape = infer({"x": [32, 64], "y": [32, 64]}, {"数字参数": 1})
            
        示例：
            infer({"x": [32, 64]}, {})  # 返回{"result": [32, 64]}
        """
        xShape = inputs.get("x", None)  # 获取输入x的形状
        yShape = inputs.get("y", None)  # 获取输入y的形状
        
        if xShape is not None:  # 如果有x的形状
            return {"result": xShape}  # 输出形状和x一样
        
        if yShape is not None:  # 如果有y的形状
            return {"result": yShape}  # 输出形状和y一样
        
        return {"result": None}  # 没有输入形状，返回None
    
    def build(shape, params):
        """
        构建层函数 - 根据形状和参数构建层实例
        
        用法：
            layer = build({"result": [32, 64]}, {"数字参数": 1})
            
        示例：
            这个示例节点不需要构建层，返回None
        """
        return None  # 示例节点不需要层，返回None
    
    def compute(inputs, params, layer, ctx):
        """
        计算函数 - 执行实际的计算
        
        用法：
            result = compute({"x": 10, "y": 20}, {"数字参数": 1}, None, ctx)
            
        示例：
            compute({"x": 5, "y": 3}, {"数字参数": 2}, None, ctx)  # 返回{"result": 10}
        """
        x = inputs.get("x", 0)  # 获取输入x，默认为0
        y = inputs.get("y", 0)  # 获取输入y，默认为0
        multiplier = params.get("数字参数", 1)  # 获取数字参数，默认为1
        
        result = (x + y) * multiplier  # 计算结果：(x + y) * 数字参数
        
        return {"result": result}  # 返回结果字典
    
    func = {}  # 创建空字典准备装三个函数
    func["infer"] = infer  # 存入infer函数
    func["build"] = build  # 存入build函数
    func["compute"] = compute  # 存入compute函数
    return func  # 返回包含三个函数的字典
