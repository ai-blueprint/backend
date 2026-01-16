import json                                                                    # 导入 JSON 处理库，用于导出配置文件
import os                                                                      # 导入操作系统接口库，用于路径处理
from loader import load_all_nodes                                               # 导入动态加载节点的函数

class Registry:                                                                 # 定义注册表类
    """ 注册表：负责管理所有算子的定义，并为前端生成配置文件 """                               # 类文档字符串

    def __init__(self):                                                         # 构造函数，初始化注册表
        self.categories = {}                                                    # 存储算子分类信息（如：基础、数学、激活函数）
        self.nodes = {}                                                         # 存储算子详细配置（如：端口、参数、名称）

    def load_nodes(self, nodes_dir=None):                                       # 加载算子的主方法
        """ 扫描目录并加载所有算子定义 """                                             # 方法文档字符串
        if nodes_dir is None:                                                   # 如果未指定目录
            current_dir = os.path.dirname(os.path.abspath(__file__))            # 获取当前文件所在目录
            nodes_dir = os.path.join(current_dir, 'nodes')                      # 默认指向同级目录下的 nodes 文件夹
        
        self.categories, self.nodes = load_all_nodes(nodes_dir)                 # 调用加载器获取分类和节点数据
        return self.categories, self.nodes                                      # 返回加载后的结果

    def get_function(self, opcode):                                             # 获取算子定义的方法
        """ 根据算子 ID 获取其定义信息 """                                             # 方法文档字符串
        return self.nodes.get(opcode)                                           # 从字典中返回对应的算子配置

    def export_to_frontend(self, output_file="node_registry.json"):             # 导出配置给前端的方法
        """ 生成符合前端 React Flow 要求的注册表文件 """                                # 方法文档字符串
        frontend_data = self._prepare_frontend_data()                           # 第一步：准备前端需要的数据结构
            
        with open(output_file, 'w', encoding='utf-8') as f:                     # 第二步：打开输出文件
            json.dump(frontend_data, f, ensure_ascii=False, indent=2)           # 第三步：将数据以格式化的 JSON 写入文件
            
        print(f"✅ 前端注册表已生成：{output_file}")                                   # 打印成功提示信息

    def _prepare_frontend_data(self):                                           # 准备前端数据的内部方法
        data = {"categories": {}, "nodes": {}}                                  # 初始化空的数据结构
        self._build_categories(data)                                            # 填充分类部分
        self._build_nodes(data)                                                 # 填充节点部分
        return data                                                             # 返回构建完成的数据

    def _build_categories(self, data):                                          # 构建分类数据的内部方法
        for cat_id, cat_info in self.categories.items():                        # 遍历所有已加载的分类
            data["categories"][cat_id] = {                                      # 填充单个分类信息
                "label": cat_info.get('name'),                                  # 分类在前端显示的名称
                "color": cat_info.get('color'),                                 # 分类的主题颜色
                "nodes": self._get_nodes_in_category(cat_id)                    # 获取该分类下的所有节点 ID 列表
            }

    def _get_nodes_in_category(self, cat_id):                                   # 获取分类下节点列表的内部方法
        return [node_id for node_id, n in self.nodes.items() if n.get('category') == cat_id] # 过滤出属于该分类的节点

    def _build_nodes(self, data):                                               # 构建节点数据的内部方法
        for node_id, node_info in self.nodes.items():                           # 遍历所有已加载的节点
            data["nodes"][node_id] = self._convert_to_frontend_format(node_id, node_info) # 转换为前端格式并保存

    def _convert_to_frontend_format(self, node_id, node_info):                  # 转换单个节点格式的内部方法
        params_dict = node_info.get('params', {})                               # 获取节点的原始参数字典
        converted_params = self._convert_params(params_dict)                    # 将参数字典转换为前端需要的格式
        
        return {                                                                # 返回前端要求的节点对象结构
            "label": node_info.get('name', node_id),                            # 节点显示名称
            "opcode": node_id,                                                  # 节点唯一标识符
            "inputs": [{"id": p, "label": p} for p in node_info.get('ports', {}).get('in', [])], # 输入端口列表
            "outputs": [{"id": p, "label": p} for p in node_info.get('ports', {}).get('out', [])], # 输出端口列表
            "params": converted_params                                          # 转换后的参数配置
        }

    def _convert_params(self, params_dict):                                     # 转换参数字典的内部方法
        result = {}                                                             # 存储转换后的参数结果
        for key, value in params_dict.items():                                  # 遍历原始参数
            param_type = self._infer_type(value)                                # 推断参数的类型（number/boolean/string）
            result[key] = {                                                     # 构建前端参数对象
                "label": key,                                                   # 参数标签
                "type": param_type,                                             # 参数类型
                "default": str(value) if isinstance(value, (int, float)) else value # 默认值处理
            }
        return result                                                           # 返回转换后的参数字典

    def _infer_type(self, value):                                               # 推断参数类型的内部方法
        if isinstance(value, bool): return "boolean"                            # 如果是布尔值，返回 boolean
        if isinstance(value, (int, float)) or value is None: return "number"    # 如果是数值或空，返回 number
        return "string"                                                         # 其他情况一律视为 string

if __name__ == "__main__":                                                      # 主程序入口
    reg = Registry()                                                            # 实例化注册表
    reg.load_nodes()                                                            # 加载所有算子
    reg.export_to_frontend()                                                    # 导出前端配置文件
