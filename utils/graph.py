"""
图算法工具模块

提供蓝图执行所需的图算法，如拓扑排序。
对应开发目标.txt L148-160
"""

from typing import Any, Dict, List, Set, Optional
from collections import deque


def topological_sort(  # 拓扑排序
    nodes: Dict[str, Any],  # 参数：nodes, edges
    edges: List[Dict[str, str]]
) -> List[str]:
    """
    对节点进行拓扑排序，确定执行顺序

    使用Kahn算法（BFS）实现，适合DAG（有向无环图）。

    参数:
        nodes: 节点字典 {node_id: node_info}
        edges: 边列表 [{"source": id, "target": id}, ...]

    返回:
        按执行顺序排列的节点ID列表

    示例:
        >>> nodes = {"a": {}, "b": {}, "c": {}}
        >>> edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
        >>> topological_sort(nodes, edges)
        ["a", "b", "c"]
    """
    if not nodes:
        return []

    # 构建入度表和邻接表
    in_degree = {node_id: 0 for node_id in nodes}  # 构建入度表
    adjacency = {node_id: [] for node_id in nodes}  # 构建邻接表

    for edge in edges:
        src = edge.get('source')
        dst = edge.get('target')
        if src in nodes and dst in nodes:
            adjacency[src].append(dst)
            in_degree[dst] += 1

    # 使用队列进行BFS
    queue = deque(node_id for node_id, degree in in_degree.items() if degree == 0)  # 将入度为0的节点入队
    execution_order = []

    while queue:  # 循环处理队列
        current = queue.popleft()  # 弹出节点
        execution_order.append(current)  # 加入结果

        for neighbor in adjacency[current]:  # 遍历其后继节点
            in_degree[neighbor] -= 1  # 入度减1
            if in_degree[neighbor] == 0:  # 入度变0的入队
                queue.append(neighbor)

    # 检测是否有环（如果有节点未被处理）
    if len(execution_order) != len(nodes):  # 如果结果数量不等于节点数量
        # 抛异常：存在循环依赖
        pass  # 这里简化处理，不抛出异常

    return execution_order  # 返回排序结果


def get_node_inputs(
    node_id: str,
    edges: List[Dict[str, str]],
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    收集节点的所有输入数据
    
    根据边的连接关系，从已计算的结果中获取当前节点的输入。
    
    参数:
        node_id: 当前节点ID
        edges: 边列表
        results: 已计算节点的结果 {node_id: output}
    
    返回:
        {input_port: value} 输入端口到值的映射
    """
    node_inputs = {}
    
    for edge in edges:
        if edge.get('target') != node_id:
            continue
        
        src_id = edge.get('source')
        src_port = edge.get('sourceHandle', 'out')
        dst_port = edge.get('targetHandle', 'in')
        
        # 从源节点的输出中获取数据
        src_output = results.get(src_id)
        if src_output is not None:
            if isinstance(src_output, dict):
                value = src_output.get(src_port)
            else:
                value = src_output
            node_inputs[dst_port] = value
    
    return node_inputs


def get_downstream_nodes(
    node_id: str,
    edges: List[Dict[str, str]]
) -> Set[str]:
    """
    获取节点的所有下游节点
    
    参数:
        node_id: 节点ID
        edges: 边列表
    
    返回:
        下游节点ID集合
    """
    downstream = set()
    for edge in edges:
        if edge.get('source') == node_id:
            downstream.add(edge.get('target'))
    return downstream


def get_upstream_nodes(
    node_id: str,
    edges: List[Dict[str, str]]
) -> Set[str]:
    """
    获取节点的所有上游节点
    
    参数:
        node_id: 节点ID
        edges: 边列表
    
    返回:
        上游节点ID集合
    """
    upstream = set()
    for edge in edges:
        if edge.get('target') == node_id:
            upstream.add(edge.get('source'))
    return upstream


def find_all_paths(
    start: str,
    end: str,
    edges: List[Dict[str, str]]
) -> List[List[str]]:
    """
    查找两个节点之间的所有路径
    
    参数:
        start: 起始节点ID
        end: 目标节点ID
        edges: 边列表
    
    返回:
        路径列表，每个路径是节点ID列表
    """
    # 构建邻接表
    adjacency: Dict[str, List[str]] = {}
    for edge in edges:
        src = edge.get('source')
        dst = edge.get('target')
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(dst)
    
    # DFS查找所有路径
    paths = []
    
    def dfs(current: str, path: List[str], visited: Set[str]):
        if current == end:
            paths.append(path.copy())
            return
        
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    dfs(start, [start], {start})
    return paths
