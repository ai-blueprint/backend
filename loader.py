"""
节点加载器模块

负责动态加载 nodes/ 目录下的所有节点定义文件。
"""

import os
import sys
import importlib.util
from typing import Dict, Tuple, Optional, List

import registry


def load_all_nodes(nodes_dir: str) -> Tuple[Dict, Dict]:  # 加载所有节点
    """
    加载指定目录下的所有节点定义

    参数:
        nodes_dir: 节点目录路径

    返回:
        (categories, nodes) 元组
    """
    registry.reset_registry()  # 重置注册表

    if not _validate_directory(nodes_dir):  # 获取 nodes 文件夹路径
        return {}, {}

    _load_directory(nodes_dir)  # 遍历文件夹

    return registry._categories, registry._nodes  # 返回加载数量


def reload_nodes(nodes_dir: str) -> Tuple[Dict, Dict]:
    """
    重新加载节点（用于开发时热更新）

    参数:
        nodes_dir: 节点目录路径

    返回:
        (categories, nodes) 元组
    """
    _unload_node_modules()
    return load_all_nodes(nodes_dir)


# ==================== 内部函数 ====================

def _validate_directory(nodes_dir: str) -> bool:
    """验证目录是否存在"""
    if not os.path.exists(nodes_dir):
        print(f"⚠️ 节点目录不存在: {nodes_dir}")
        return False
    if not os.path.isdir(nodes_dir):
        print(f"⚠️ 路径不是目录: {nodes_dir}")
        return False
    return True


def _load_directory(nodes_dir: str):  # 遍历文件夹
    """递归加载目录下的所有节点模块"""
    for root, _, files in os.walk(nodes_dir):
        for file in files:
            if _is_node_file(file):  # 如果是 .py 文件
                _load_module_safe(root, file)  # 动态导入模块


def _is_node_file(filename: str) -> bool:
    """判断文件是否为有效的节点定义文件"""
    if not filename.endswith(".py"):
        return False
    if filename.startswith("_") or filename == "__pycache__":  # 跳过 __pycache__
        return False
    if filename == "__init__.py":  # 跳过 __init__.py
        return False
    return True


def _load_module_safe(root: str, filename: str):  # 动态导入模块
    """安全地加载单个模块"""
    file_path = os.path.join(root, filename)
    module_name = _generate_module_name(filename)  # 根据文件路径计算模块名

    try:
        _load_module(file_path, module_name)  # 使用 importlib 加载
        # 模块中的 @category 和 @node 装饰器会自动注册
    except Exception as e:  # 捕获异常并打印警告
        print(f"❌ 加载算子模块失败 {file_path}: {e}")


def _generate_module_name(filename: str) -> str:
    """生成模块名称"""
    base_name = filename[:-3]  # 移除 .py
    return f"nodes.{base_name}"


def _load_module(file_path: str, module_name: str):  # 动态导入模块
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)  # 根据文件路径计算模块名

    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模块规范: {file_path}")

    module = importlib.util.module_from_spec(spec)  # 使用 importlib 加载
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _unload_node_modules():
    """卸载已加载的节点模块（用于重新加载）"""
    to_remove = [name for name in sys.modules if name.startswith("nodes.")]
    for name in to_remove:
        del sys.modules[name]


# ==================== 工具函数 ====================

def get_loaded_nodes() -> List[str]:
    """获取已加载的节点列表"""
    return list(registry._nodes.keys())


def get_loaded_categories() -> List[str]:
    """获取已加载的分类列表"""
    return list(registry._categories.keys())


def get_node_info(opcode: str) -> Optional[Dict]:
    """获取指定节点的信息"""
    return registry._nodes.get(opcode)


def get_category_info(cat_id: str) -> Optional[Dict]:
    """获取指定分类的信息"""
    return registry._categories.get(cat_id)
