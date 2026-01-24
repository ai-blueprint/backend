"""
decorators.py - 装饰器

用法：
    from decorators import category, node
    
    @category(id="math", label="数学运算", color="#FF6B6B", icon="")
    
    @node(opcode="add", label="加法", category="math", ports={"in": ["a", "b"], "out": ["result"]}, params={})
    def addNode():
        def infer(inputs, params): ...
        def build(shape, params): ...
        def compute(inputs, params, layer, ctx): ...
        return {"infer": infer, "build": build, "compute": compute}
        
示例：
    见nodes/example.py
"""

import registry  # 注册表模块，用于注册分类和节点


def category(id="", label="", color="#888888", icon=""):
    """
    分类装饰器 - 用于注册一个节点分类
    
    用法：
        @category(id="math", label="数学运算", color="#FF6B6B", icon="base64...")
        
    示例：
        @category(id="layers", label="神经网络层", color="#4ECDC4", icon="")
        @category(id="activations", label="激活函数", color="#FFE66D")
    """
    registry.registerCategory(id, label, color, icon)  # 调用registry注册这个分类
    return None  # 这个装饰器不需要装饰任何东西，只是用来注册分类


def node(opcode="", label="", category="", ports=None, params=None):
    """
    节点装饰器 - 用于注册一个节点定义
    
    用法：
        @node(opcode="add", label="加法", category="math", ports={"in": ["a", "b"], "out": ["result"]}, params={})
        def addNode():
            def infer(inputs, params): ...
            def build(shape, params): ...
            def compute(inputs, params, layer, ctx): ...
            return {"infer": infer, "build": build, "compute": compute}
            
    示例：
        @node(opcode="linear", label="线性层", category="layers", 
              ports={"in": ["x"], "out": ["y"]}, params={"units": 64})
        def linearNode():
            ...
    """
    if ports is None:  # 如果没有传ports参数
        ports = {"in": [], "out": []}  # 使用默认空端口
    
    if params is None:  # 如果没有传params参数
        params = {}  # 使用默认空参数
    
    def decorator(fn):  # 定义装饰器函数，接收被装饰的函数
        """
        装饰器内部函数 - 接收被装饰的函数并注册节点
        """
        func = fn()  # 调用被装饰的函数，获取返回的{infer, build, compute}字典
        registry.registerNode(opcode, label, category, ports, params, func)  # 调用registry注册这个节点
        return fn  # 返回原函数，保持可调用性
    
    return decorator  # 返回装饰器函数
