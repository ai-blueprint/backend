import os
import importlib.util
import sys
from decorators import CATEGORIES, NODES

def load_all_nodes(nodes_dir):
    """
    动态加载指定目录下的所有 Python 文件，触发装饰器注册节点。
    """
    # 清空之前的注册信息，防止重复加载
    CATEGORIES.clear()
    NODES.clear()
    
    if not os.path.exists(nodes_dir):
        print(f"Warning: Nodes directory {nodes_dir} does not exist.")
        return {}, {}

    # 遍历目录
    for root, dirs, files in os.walk(nodes_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = os.path.join(root, file)
                module_name = f"nodes.{file[:-3]}"
                
                # 动态加载模块
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # 将模块加入 sys.modules 以便内部 import 正常工作
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                    except Exception as e:
                        print(f"Error loading node module {file_path}: {e}")

    return CATEGORIES, NODES
