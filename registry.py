"""
节点注册表模块

管理所有节点定义，并为前端生成配置数据。
对应开发目标.txt L48-68
"""

import json
import os
from typing import Any, Dict, List, Optional

from utils.safe import safe_get


# ==================== 全局变量 ====================

_nodes: Dict[str, Dict[str, Any]] = {}  # _nodes：节点定义字典
_categories: Dict[str, Dict[str, Any]] = {}  # _categories：分类定义字典


# ==================== 注册函数 ====================


def register_category(id: str, name: str, color: str = "#888888"):  # 注册分类
    """
    注册分类定义

    参数:
        id: 分类唯一标识符
        name: 分类显示名称
        color: 分类主题颜色
    """
    _categories[id] = {"id": id, "name": name, "color": color}  # 存入 _categories


def register_node(  # 注册节点
    id: str,
    name: str,
    category: str,
    inputs: List[str],
    outputs: List[str],
    params: Dict[str, Any],
    func: Any,
):
    """
    注册节点定义

    参数:
        id: 节点唯一标识符
        name: 节点显示名称
        category: 所属分类ID
        inputs: 输入端口列表
        outputs: 输出端口列表
        params: 参数定义字典
        func: 节点工厂函数
    """
    _nodes[id] = {  # 存入 _nodes
        "id": id,
        "name": name,
        "category": category,
        "inputs": inputs,
        "outputs": outputs,
        "params": params,
        "func": func,
    }


# ==================== 查询函数 ====================


def get_function(opcode: str) -> Optional[Dict[str, Any]]:  # 获取节点定义
    """
    获取节点定义

    参数:
        opcode: 节点类型标识符

    返回:
        节点定义字典，或 None
    """
    return _nodes.get(opcode)  # 从 _nodes 查询


def get_all_nodes() -> Dict[str, Dict[str, Any]]:  # 获取所有节点
    """
    获取所有节点定义

    返回:
        节点定义字典的副本
    """
    return _nodes.copy()  # 返回 _nodes 的副本


def get_category(cat_id: str) -> Optional[Dict[str, Any]]:
    """
    获取分类定义

    参数:
        cat_id: 分类标识符

    返回:
        分类定义字典，或 None
    """
    return _categories.get(cat_id)


def get_nodes_by_category(cat_id: str) -> List[str]:  # 获取分类下的节点
    """
    获取指定分类下的所有节点

    参数:
        cat_id: 分类ID

    返回:
        节点ID列表
    """
    return [  # 遍历 _nodes，筛选出指定分类的
        node_id for node_id, node in _nodes.items() if node.get("category") == cat_id
    ]


def list_all_categories() -> List[str]:
    """列出所有分类ID"""
    return list(_categories.keys())


def list_all_nodes() -> List[str]:
    """列出所有节点ID"""
    return list(_nodes.keys())


# ==================== 前端数据导出 ====================


def get_all_for_frontend() -> Dict[str, Any]:  # 获取前端格式数据
    """
    获取前端格式的数据

    返回:
        包含 categories 和 nodes 的字典
    """
    frontend_categories = {}  # 遍历 _categories，构建分类列表
    for cat_id, cat_info in _categories.items():
        frontend_categories[cat_id] = {
            "label": cat_info.get("name", cat_id),
            "color": cat_info.get("color", "#888888"),
            "nodes": get_nodes_by_category(cat_id),
        }

    frontend_nodes = {}  # 遍历 _nodes，构建节点列表
    for node_id, node_info in _nodes.items():
        frontend_nodes[node_id] = {
            "label": node_info.get("name", node_id),
            "opcode": node_id,
            "category": node_info.get("category", ""),
            "inputs": _format_ports(node_info.get("inputs", [])),  # 转换 inputs 格式
            "outputs": _format_ports(node_info.get("outputs", [])),  # 转换 outputs 格式
            "params": _convert_params(node_info.get("params", {})),  # 转换 params 格式
        }

    return {  # 返回 {categories, nodes}
        "categories": frontend_categories,
        "nodes": frontend_nodes,
    }


def export_to_frontend(output_file: str = "node_registry.json"):
    """
    导出前端配置文件

    参数:
        output_file: 输出文件路径
    """
    frontend_data = get_all_for_frontend()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(frontend_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 前端注册表已生成：{output_file}")


# ==================== 辅助函数 ====================


def _format_ports(port_list: List[str]) -> List[Dict[str, str]]:
    """格式化端口列表"""
    return [{"id": port, "label": port} for port in port_list]


def _convert_params(params_dict: Dict) -> Dict[str, Dict]:
    """将参数字典转换为前端格式"""
    result = {}
    for key, value in params_dict.items():
        param_type = _infer_param_type(value)
        result[key] = {
            "label": key,
            "type": param_type,
            "default": _format_default_value(value, param_type),
        }
    return result


def _infer_param_type(value: Any) -> str:
    """推断参数类型"""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) or value is None:
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"


def _format_default_value(value: Any, param_type: str) -> Any:
    """格式化默认值"""
    if value is None:
        return ""
    if param_type == "number" and isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return value
    return value


def reset_registry():
    """重置注册表（用于测试或重新加载）"""
    global _nodes, _categories
    _nodes.clear()
    _categories.clear()


# ==================== 主程序 ====================

if __name__ == "__main__":
    from loader import load_all_nodes

    nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
    load_all_nodes(nodes_dir)

    print(f"已加载 {len(_nodes)} 个节点，{len(_categories)} 个分类")
    export_to_frontend()
