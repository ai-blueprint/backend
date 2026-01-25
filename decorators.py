"""
decorators.py - 装饰器

用法：
    from decorators import category, node
    
    @category(id="math", label="数学运算", color="#FF6B6B", icon="")
    def math_category():
        pass
    
    @node(opcode="add", label="加法", inputs=["a", "b"], outputs=["result"], params={})
    def addNode():
        def infer(inputs, params): ...
        def build(shape, params): ...
        def compute(inputs, params, layer, ctx): ...
        return infer, build, compute
        
示例：
    见nodes/example.py
"""

import registry  # 注册表模块，用于注册分类和节点

currentCategory = ""  # 当前分类id，用于自动归类后续注册的节点


def category(id="", label="", color="#888888", icon=""):
    """
    分类装饰器 - 用于注册一个节点分类
    
    用法：
        @category(id="math", label="数学运算", color="#FF6B6B", icon="base64...")
        def math_category():
            pass
    """
    global currentCategory  # 声明使用全局变量
    currentCategory = id  # 设置当前分类为这个分类的id
    registry.registerCategory(id, label, color, icon)  # 调用registry注册这个分类
    
    def wrapper(fn):  # 定义内部包装函数，用来接收被装饰的函数
        return fn  # 原样返回函数，不作任何修改
    
    return wrapper  # 返回包装函数，让Python用它来装饰下方的函数


def node(opcode="", label="", category="", ports=None, inputs=None, outputs=None, params=None):
    """
    节点装饰器 - 用于注册一个节点定义
    
    用法：
        @node(opcode="add", label="加法", inputs=["a", "b"], outputs=["result"], params={})
        def addNode():
            def infer(inputs, params): ...
            def build(shape, params): ...
            def compute(inputs, params, layer, ctx): ...
            return infer, build, compute
    """
    # 优先处理新版的 inputs/outputs 传参方式
    if ports is None:  # 如果没有直接传 ports
        in_ports = inputs if inputs is not None else []  # 获取输入端口列表，默认为空
        out_ports = outputs if outputs is not None else []  # 获取输出端口列表，默认为空
        ports = {"in": in_ports, "out": out_ports}  # 组装成注册表需要的格式
    
    if params is None:  # 如果没有传params参数
        params = {}  # 使用默认空参数
    
    nodeCategory = category if category else currentCategory  # 如果没传category就用当前分类
    
    def decorator(fn):  # 定义装饰器函数，接收被装饰的函数
        """
        装饰器内部函数 - 接收被装饰的函数并注册节点
        """
        func = fn()  # 调用被装饰的函数，获取返回的(infer, build, compute)元组或字典
        registry.registerNode(opcode, label, nodeCategory, ports, params, func)  # 调用registry注册这个节点
        return fn  # 返回原函数，保持可调用性
    
    return decorator  # 返回装饰器函数
