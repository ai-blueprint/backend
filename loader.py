"""
节点加载器模块

负责动态加载 nodes/ 目录下的所有节点定义文件。

工作原理：
1. 扫描指定目录下的所有 Python 文件
2. 动态导入每个模块
3. 模块中的装饰器会自动将节点注册到全局字典
"""

import os
import sys
import importlib.util
from typing import Dict, Tuple, Optional, List

from decorators import CATEGORIES, NODES


def load_all_nodes(nodes_dir: str) -> Tuple[Dict, Dict]:
    """
    加载指定目录下的所有节点定义
    
    参数:
        nodes_dir: 节点目录路径
    
    返回:
        (categories, nodes) 元组
    """
    _reset_registry()
    
    if not _validate_directory(nodes_dir):
        return {}, {}
    
    _load_directory(nodes_dir)
    
    return CATEGORIES, NODES


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

def _reset_registry():
    """重置全局注册表"""
    CATEGORIES.clear()
    NODES.clear()


def _validate_directory(nodes_dir: str) -> bool:
    """验证目录是否存在"""
    if not os.path.exists(nodes_dir):
        print(f"⚠️ 节点目录不存在: {nodes_dir}")
        return False
    if not os.path.isdir(nodes_dir):
        print(f"⚠️ 路径不是目录: {nodes_dir}")
        return False
    return True


def _load_directory(nodes_dir: str):
    """递归加载目录下的所有节点模块"""
    for root, _, files in os.walk(nodes_dir):
        for file in files:
            if _is_node_file(file):
                _load_module_safe(root, file)


def _is_node_file(filename: str) -> bool:
    """判断文件是否为有效的节点定义文件"""
    if not filename.endswith(".py"):
        return False
    if filename.startswith("_"):
        return False
    if filename == "__init__.py":
        return False
    return True


def _load_module_safe(root: str, filename: str):
    """安全地加载单个模块"""
    file_path = os.path.join(root, filename)
    module_name = _generate_module_name(filename)
    
    try:
        _load_module(file_path, module_name)
    except Exception as e:
        print(f"❌ 加载算子模块失败 {file_path}: {e}")


def _generate_module_name(filename: str) -> str:
    """生成模块名称"""
    base_name = filename[:-3]  # 移除 .py
    return f"nodes.{base_name}"


def _load_module(file_path: str, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模块规范: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
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
    return list(NODES.keys())


def get_loaded_categories() -> List[str]:
    """获取已加载的分类列表"""
    return list(CATEGORIES.keys())


def get_node_info(opcode: str) -> Optional[Dict]:
    """获取指定节点的信息"""
    return NODES.get(opcode)


def get_category_info(cat_id: str) -> Optional[Dict]:
    """获取指定分类的信息"""
    return CATEGORIES.get(cat_id)
