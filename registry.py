""" 
读取backend/nodes目录下的所有文件，根据节点组定义，注册节点信息到registry中
registry有两个，一个是给VM调用的，一个是给前端调用的
"""
import json
import os
from loader import load_all_nodes

class Registry:
    def __init__(self):
        self.registry = {}  # 存储节点函数对象，供Engine使用
        self.categories = {}  # 存储类别信息
        self.nodes = {}  # 存储节点信息
        
    def load_nodes(self, nodes_dir=None):
        """加载所有节点"""
        if nodes_dir is None:
            # 默认加载backend/nodes目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            nodes_dir = os.path.join(current_dir, 'nodes')
        
        # 调用loader加载所有节点
        self.categories, self.nodes = load_all_nodes(nodes_dir)
        
        # 构建registry字典，供Engine使用
        for node_id, node_info in self.nodes.items():
            self.registry[node_id] = node_info
            
                
        
    def get_function(self, opcode):
        """根据opcode获取节点函数对象，供Engine调用"""
        return self.registry.get(opcode)

    def generate_category_files(self, output_dir=None):
        """为每个分类生成单独的JSON文件，保存到src/assets/node-groups目录"""
        if output_dir is None:
            # 默认输出到src/assets/node-groups目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            output_dir = os.path.join(project_root, 'src', 'assets', 'node-groups')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个分类生成JSON文件
        for category_id, category_info in self.categories.items():
            # 获取该分类下的所有节点
            category_nodes = [n for n in self.nodes.values() if n.get('category') == category_id]
            
            # 转换节点信息
            nodes_data = []
            for node_info in category_nodes:
                node_id = node_info.get('opcode') or list(self.nodes.keys())[list(self.nodes.values()).index(node_info)]
                node_data = self._convert_node_data(node_id, node_info)
                nodes_data.append(node_data)
            
            # 构建分类文件数据
            category_data = {
                "id": category_id,
                "name": category_info.get('name', category_id),
                "color": category_info.get('color', '#82CBFA'),
                "icon": category_info.get('icon', 'base64…'),
                "nodes": nodes_data
            }
            
            # 生成输出文件路径
            output_file = os.path.join(output_dir, f"{category_id}.json")
            
            # 写入JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(category_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 分类文件已生成：{output_file}")
        
        return True
    
    def _convert_node_data(self, node_id, node_info):
        """转换节点数据格式，特别是params字段"""
        # 转换params格式：从字典 {key: value} 转换为数组 [{'label': key, 'type': type, 'default': value}]}
        params_dict = node_info.get('params', {})
        params_list = []
        
        for key, value in params_dict.items():
            # 根据值的类型推断type字段
            if isinstance(value, bool):
                param_type = "boolean"
                value = bool(value)
            elif isinstance(value, (int, float)):
                param_type = "number"
                # 数值类型默认值转换为字符串
                value = str(value)
            elif value is None:
                param_type = "number"  # 对于sum和mean的dim参数，默认为None
                value = ""
            else:
                param_type = "string"
            
            param_item = {
                "label": key,
                "type": param_type,
                "default": value
            }
            params_list.append(param_item)
        
        return {
            "label": node_info.get('name', node_id),
            "opcode": node_id,
            "inputs": [{"id": p, "label": p} for p in node_info.get('ports', {}).get('in', [])],
            "outputs": [{"id": p, "label": p} for p in node_info.get('ports', {}).get('out', [])],
            "params": {p['label']: {"label": p['label'], "type": p['type'], "default": p['default']} for p in params_list}
        }

    def export_to_frontend(self, output_file="node_registry.json"):
        """导出统一的注册表文件给前端"""
        frontend_data = {
            "categories": {},
            "nodes": {}
        }
        
        for cat_id, cat_info in self.categories.items():
            frontend_data["categories"][cat_id] = {
                "label": cat_info.get('name'),
                "color": cat_info.get('color'),
                "nodes": [node_id for node_id, n in self.nodes.items() if n.get('category') == cat_id]
            }
            
        for node_id, node_info in self.nodes.items():
            frontend_data["nodes"][node_id] = self._convert_node_data(node_id, node_info)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 前端注册表已生成：{output_file}")

# 使用示例
if __name__ == "__main__":
    registry = Registry()
    registry.load_nodes()
    registry.export_to_frontend()