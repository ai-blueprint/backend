import os  # 操作系统模块，用于文件路径操作
import importlib  # 动态导入模块的库


def importModule(filepath):
    """
    示例：importModule("nodes/example.py")  # 会被转换成nodes.example模块并导入
    """
    relative = filepath.replace("\\", "/")  # 把Windows路径的反斜杠替换成正斜杠
    noExt = relative.replace(".py", "")  # 去掉.py后缀
    moduleName = noExt.replace("/", ".")  # 把路径分隔符替换成点号，变成模块名格式
    importlib.import_module(moduleName)  # 使用importlib动态导入这个模块


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
