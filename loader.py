import os                                                                      # 导入操作系统接口库，用于文件和目录操作
import importlib.util                                                           # 导入动态导入模块的工具库
import sys                                                                      # 导入系统参数和函数库，用于管理模块路径
from decorators import CATEGORIES, NODES                                        # 导入装饰器中定义的全局存储字典

def load_all_nodes(nodes_dir):                                                  # 定义加载所有算子的主函数
    """ 动态加载指定目录下的所有 Python 文件，触发装饰器注册算子 """                        # 函数文档字符串
    
    _reset_registry()                                                           # 第一步：重置注册表，防止重复加载导致数据混乱
    
    if not os.path.exists(nodes_dir):                                           # 第二步：检查目标目录是否存在
        return {}, {}                                                           # 如果不存在，直接返回空的分类和节点字典

    _scan_and_load_directory(nodes_dir)                                         # 第三步：扫描并加载目录下的所有模块

    return CATEGORIES, NODES                                                    # 返回最终收集到的所有分类和节点定义

def _reset_registry():                                                          # 内部方法：重置全局注册表
    CATEGORIES.clear()                                                          # 清空分类字典
    NODES.clear()                                                               # 清空节点字典

def _scan_and_load_directory(nodes_dir):                                        # 内部方法：递归扫描目录
    for root, _, files in os.walk(nodes_dir):                                   # 递归遍历指定目录
        for file in files:                                                      # 遍历目录下的每一个文件
            if _is_node_file(file):                                             # 判断是否为有效的算子定义文件
                _load_module(root, file)                                        # 加载该 Python 模块

def _is_node_file(file):                                                        # 内部方法：判断文件是否为算子文件
    return file.endswith(".py") and file != "__init__.py"                       # 必须是 .py 结尾且不是初始化文件

def _load_module(root, file):                                                   # 内部方法：动态加载单个模块
    file_path = os.path.join(root, file)                                        # 获取文件的完整绝对路径
    module_name = f"nodes.{file[:-3]}"                                          # 构造模块名称（例如 nodes.math）
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)       # 创建模块的导入规范
    if spec is None or spec.loader is None:                                     # 如果规范或加载器无效
        return                                                                  # 直接跳过

    module = importlib.util.module_from_spec(spec)                              # 根据规范创建一个新的模块对象
    sys.modules[module_name] = module                                           # 将模块注册到全局 sys.modules 中
    
    try:                                                                        # 尝试执行模块代码
        spec.loader.exec_module(module)                                         # 执行模块，这会触发文件内的 @node 装饰器
    except Exception as e:                                                      # 如果加载过程中发生错误
        print(f"❌ 加载算子模块失败 {file_path}: {e}")                             # 打印错误信息，方便调试
