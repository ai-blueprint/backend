import os  # 操作系统模块，用于文件路径操作
import sys  # 系统模块，用于操作模块缓存
import importlib.util  # 按文件路径动态导入模块的库
import registry  # 节点注册表模块，用于清空重建


def importModule(filepath):
    """
    示例：importModule("nodes/example.py")  # 按文件路径直接导入模块，不做手工路径转换
    """
    absoluteFilepath = filepath if os.path.isabs(filepath) else os.path.join(os.path.dirname(__file__), filepath)  # 相对路径转绝对路径后再加载
    moduleName = os.path.splitext(os.path.basename(absoluteFilepath))[0]  # 用文件名生成模块名
    spec = importlib.util.spec_from_file_location(moduleName, absoluteFilepath)  # 从文件路径创建模块加载规范
    module = importlib.util.module_from_spec(spec)  # 根据规范创建模块对象
    spec.loader.exec_module(module)  # 执行模块代码完成导入


def loadAll(folder="nodes"):
    """
    示例：loadAll()  # 自动递归加载nodes/**/*.py，里面的@category和@node装饰器会自动注册节点
    """
    nodesDir = os.path.join(os.path.dirname(__file__), folder)  # 获取nodes文件夹的绝对路径

    for root, dirs, files in os.walk(nodesDir):  # 递归遍历nodes目录和所有子目录
        dirs[:] = sorted([dirName for dirName in dirs if dirName != "__pycache__"])  # 跳过__pycache__并保证目录遍历顺序稳定
        for filename in sorted(files):  # 遍历当前目录下的文件并保证文件遍历顺序稳定
            if filename == "__init__.py":  # 如果是__init__.py文件
                continue  # 跳过，不处理

            if not filename.endswith(".py"):  # 如果不是.py文件
                continue  # 跳过，不处理

            absoluteFilepath = os.path.join(root, filename)  # 先得到当前文件绝对路径
            relativeFilepath = os.path.relpath(absoluteFilepath, os.path.dirname(__file__))  # 再转成相对当前工程根目录的路径
            importModule(relativeFilepath)  # 动态导入这个模块
            print(f"已加载节点模块: {relativeFilepath}")  # 打印加载信息


def reloadAll(folder="nodes"):
    """
    示例：reloadAll()  # 清空registry，清除模块缓存，重新导入所有节点文件
    """
    registry.clearAll()  # 清空节点和分类注册表
    for key in list(sys.modules.keys()):  # 遍历所有已加载模块
        if key.startswith("nodes."):  # 如果是nodes目录下的模块
            del sys.modules[key]  # 从缓存中删除，确保重新导入
    loadAll(folder)  # 重新扫描并导入所有节点文件
