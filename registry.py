"""
节点注册表模块

管理所有节点定义，并为前端生成配置数据。

职责：
1. 加载和存储节点定义
2. 提供节点查询接口
3. 生成前端所需的配置格式
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from loader import load_all_nodes
from utils.safe import safe_get
from utils.validation import validate_type


class Registry:
    """
    节点注册表
    
    使用示例:
        registry = Registry()
        registry.load_nodes()
        node_def = registry.get_function("linear")
        registry.export_to_frontend("node_registry.json")
    """
    
    def __init__(self):
        self.categories: Dict[str, Dict] = {}
        self.nodes: Dict[str, Dict] = {}
    
    # ==================== 加载与查询 ====================
    
    def load_nodes(self, nodes_dir: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        加载所有节点定义
        
        参数:
            nodes_dir: 节点目录，默认为当前目录下的 nodes/
        
        返回:
            (categories, nodes) 元组
        """
        if nodes_dir is None:
            nodes_dir = self._get_default_nodes_dir()
        
        self.categories, self.nodes = load_all_nodes(nodes_dir)
        return self.categories, self.nodes
    
    def get_function(self, opcode: str) -> Optional[Dict]:
        """
        获取节点定义
        
        参数:
            opcode: 节点类型标识符
        
        返回:
            节点定义字典，或 None
        """
        return self.nodes.get(opcode)
    
    def get_category(self, cat_id: str) -> Optional[Dict]:
        """
        获取分类定义
        
        参数:
            cat_id: 分类标识符
        
        返回:
            分类定义字典，或 None
        """
        return self.categories.get(cat_id)
    
    def list_nodes(self) -> List[str]:
        """列出所有已注册的节点"""
        return list(self.nodes.keys())
    
    def list_categories(self) -> List[str]:
        """列出所有已注册的分类"""
        return list(self.categories.keys())
    
    def get_nodes_by_category(self, cat_id: str) -> List[str]:
        """获取指定分类下的所有节点"""
        return [
            node_id for node_id, node in self.nodes.items()
            if node.get('category') == cat_id
        ]
    
    # ==================== 前端数据导出 ====================
    
    def export_to_frontend(self, output_file: str = "node_registry.json"):
        """
        导出前端配置文件
        
        参数:
            output_file: 输出文件路径
        """
        frontend_data = self._prepare_frontend_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 前端注册表已生成：{output_file}")
    
    def _prepare_frontend_data(self) -> Dict[str, Any]:
        """准备前端格式的数据"""
        return {
            "categories": self._build_frontend_categories(),
            "nodes": self._build_frontend_nodes()
        }
    
    def _build_frontend_categories(self) -> Dict[str, Dict]:
        """构建前端分类数据"""
        result = {}
        for cat_id, cat_info in self.categories.items():
            result[cat_id] = {
                "label": cat_info.get('name', cat_id),
                "color": cat_info.get('color', '#888888'),
                "icon": cat_info.get('icon', ''),
                "nodes": self.get_nodes_by_category(cat_id)
            }
        return result
    
    def _build_frontend_nodes(self) -> Dict[str, Dict]:
        """构建前端节点数据"""
        result = {}
        for node_id, node_info in self.nodes.items():
            result[node_id] = self._convert_node_to_frontend(node_id, node_info)
        return result
    
    def _convert_node_to_frontend(
        self, 
        node_id: str, 
        node_info: Dict
    ) -> Dict[str, Any]:
        """将单个节点转换为前端格式"""
        ports = node_info.get('ports', {})
        params = node_info.get('params', {})
        
        return {
            "label": node_info.get('name', node_id),
            "opcode": node_id,
            "inputs": self._format_ports(ports.get('in', [])),
            "outputs": self._format_ports(ports.get('out', [])),
            "params": self._convert_params(params)
        }
    
    @staticmethod
    def _format_ports(port_list: List[str]) -> List[Dict[str, str]]:
        """格式化端口列表"""
        return [{"id": port, "label": port} for port in port_list]
    
    def _convert_params(self, params_dict: Dict) -> Dict[str, Dict]:
        """将参数字典转换为前端格式"""
        result = {}
        for key, value in params_dict.items():
            param_type = self._infer_param_type(value)
            result[key] = {
                "label": key,
                "type": param_type,
                "default": self._format_default_value(value, param_type)
            }
        return result
    
    @staticmethod
    def _infer_param_type(value: Any) -> str:
        """推断参数类型"""
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)) or value is None:
            return "number"
        if isinstance(value, list):
            return "array"
        return "string"
    
    @staticmethod
    def _format_default_value(value: Any, param_type: str) -> Any:
        """格式化默认值"""
        if value is None:
            return ""
        if param_type == "number" and isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return value
        return value
    
    # ==================== 辅助方法 ====================
    
    def _get_default_nodes_dir(self) -> str:
        """获取默认的节点目录路径"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'nodes')
    
    def __len__(self) -> int:
        """返回已注册节点数量"""
        return len(self.nodes)
    
    def __contains__(self, opcode: str) -> bool:
        """检查节点是否已注册"""
        return opcode in self.nodes
    
    def __repr__(self) -> str:
        return f"Registry(categories={len(self.categories)}, nodes={len(self.nodes)})"


# ==================== 便捷函数 ====================

def create_registry() -> Registry:
    """创建并加载注册表"""
    registry = Registry()
    registry.load_nodes()
    return registry


# ==================== 主程序 ====================

if __name__ == "__main__":
    registry = create_registry()
    print(f"已加载 {len(registry)} 个节点")
    registry.export_to_frontend()
