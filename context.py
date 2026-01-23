"""
执行上下文模块

管理蓝图执行过程中的所有状态数据。
"""

from typing import Any, Dict, List


class ExecutionContext:  # 执行上下文类
    """
    执行上下文

    职责：
    - 存储每个节点的输出结果
    - 缓存已构建的层实例
    - 管理用户传入的初始输入数据
    """

    def __init__(self, inputs: Dict[str, Any] = None):  # 初始化
        """
        初始化执行上下文

        参数:
            inputs: 用户传入的初始数据 {node_id: {port: value}}
        """
        self.results: Dict[str, Dict[str, Any]] = {}  # results：存每个节点的输出 {node_id: {port: value}}
        self.layers: Dict[str, Any] = {}  # layers：存已构建的层 {node_id: layer_instance}
        self.inputs: Dict[str, Any] = inputs or {}  # inputs：用户传入的初始数据

    def get_inputs(  # 获取节点输入
        self,
        node_id: str,  # 参数：node_id
        edges: List[Dict[str, Any]]  # 参数：edges
    ) -> Dict[str, Any]:
        """
        根据边连接关系收集节点的输入数据

        参数:
            node_id: 目标节点ID
            edges: 所有边的列表

        返回:
            输入数据字典 {port_name: value}
        """
        inputs = {}

        for edge in edges:  # 遍历所有 edges
            if edge.get('target') == node_id:  # 如果 edge.target 是当前节点
                source_node = edge.get('source')
                source_port = edge.get('sourceHandle', 'out')
                target_port = edge.get('targetHandle', 'in')

                # 从 results 里取 source 节点的对应端口值
                if source_node in self.results:
                    source_output = self.results[source_node]
                    if isinstance(source_output, dict) and source_port in source_output:
                        inputs[target_port] = source_output[source_port]
                    elif not isinstance(source_output, dict):
                        inputs[target_port] = source_output

        return inputs  # 返回收集到的输入字典

    def store_result(self, node_id: str, output_dict: Dict[str, Any]):  # 存储结果
        """
        存储节点的输出结果

        参数:
            node_id: 节点ID
            output_dict: 输出数据字典
        """
        self.results[node_id] = output_dict  # results[node_id] = output_dict

    def get_result(self, node_id: str) -> Any:  # 获取结果
        """
        获取节点的输出结果

        参数:
            node_id: 节点ID

        返回:
            节点输出数据
        """
        return self.results.get(node_id)  # 返回 results[node_id]

    def store_layer(self, node_id: str, layer: Any):  # 存储层
        """
        缓存节点的层实例

        参数:
            node_id: 节点ID
            layer: 层实例
        """
        self.layers[node_id] = layer  # layers[node_id] = layer

    def get_layer(self, node_id: str) -> Any:  # 获取层
        """
        获取节点的层实例

        参数:
            node_id: 节点ID

        返回:
            层实例，如果不存在则返回None
        """
        return self.layers.get(node_id)  # 返回 layers.get(node_id)

    def get_initial_input(self, node_id: str) -> Any:  # 获取初始输入
        """
        获取用户为特定节点传入的初始数据

        参数:
            node_id: 节点ID

        返回:
            初始输入数据
        """
        return self.inputs.get(node_id)  # 返回 inputs.get(node_id)
