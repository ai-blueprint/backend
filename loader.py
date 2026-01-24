"""
loader.py - 动态加载模块

用法：
    import loader
    loader.loadAll()  # 加载nodes文件夹下的所有节点模块
    
示例：
    loader.loadAll()  # 扫描nodes文件夹，动态导入所有.py文件
"""

import os  # 操作系统模块，用于文件路径操作
import importlib  # 动态导入模块的库


def importModule(filepath):
    """
    动态导入模块
    
    用法：
        importModule("nodes/math.py")  # 导入nodes/math.py模块
        importModule("nodes/layers.py")  # 导入nodes/layers.py模块
        
    示例：
        importModule("nodes/example.py")  # 会被转换成nodes.example模块并导入
    """
    relative = filepath.replace("\\", "/")  # 把Windows路径的反斜杠替换成正斜杠
    noExt = relative.replace(".py", "")  # 去掉.py后缀
    moduleName = noExt.replace("/", ".")  # 把路径分隔符替换成点号，变成模块名格式
    importlib.import_module(moduleName)  # 使用importlib动态导入这个模块


def loadAll():
    """
    加载所有节点模块
    
    用法：
        loadAll()  # 扫描nodes文件夹，加载所有.py节点文件
        
    示例：
        loadAll()  # 自动加载nodes/*.py，里面的@category和@node装饰器会自动注册节点
    """
    nodesDir = os.path.join(os.path.dirname(__file__), "nodes")  # 获取nodes文件夹的绝对路径
    
    for filename in os.listdir(nodesDir):  # 遍历nodes文件夹下的所有文件
        
        if filename == "__pycache__":  # 如果是__pycache__文件夹
            continue  # 跳过，不处理
        
        if filename == "__init__.py":  # 如果是__init__.py文件
            continue  # 跳过，不处理
        
        if not filename.endswith(".py"):  # 如果不是.py文件
            continue  # 跳过，不处理
        
        filepath = os.path.join("nodes", filename)  # 拼接相对路径，比如nodes/math.py
        importModule(filepath)  # 动态导入这个模块
        print(f"已加载节点模块: {filepath}")  # 打印加载信息
