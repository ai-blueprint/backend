"""
sort.py - 拓扑排序

用法：
    import sort
    sortedIds = sort.topoSort(nodes, edges)  # 获取拓扑排序后的节点id列表
    
示例：
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
    result = sort.topoSort(nodes, edges)  # 返回["a", "b", "c"]
"""

from collections import deque  # 双端队列，用于BFS


def topoSort(nodes, edges):
    """
    拓扑排序 - 根据节点和边计算执行顺序
    
    用法：
        sortedIds = topoSort(nodes, edges)
        
    示例：
        nodes = [{"id": "input"}, {"id": "hidden"}, {"id": "output"}]
        edges = [
            {"source": "input", "target": "hidden"},
            {"source": "hidden", "target": "output"}
        ]
        result = topoSort(nodes, edges)  # 返回["input", "hidden", "output"]
    """
    inDegree = {}  # 入度表，记录每个节点有多少个前置节点
    adjacency = {}  # 邻接表，记录每个节点指向哪些后继节点
    
    for node in nodes:  # 遍历所有节点
        nodeId = node.get("id", "")  # 获取节点id
        inDegree[nodeId] = 0  # 初始化入度为0
        adjacency[nodeId] = []  # 初始化邻接列表为空
    
    for edge in edges:  # 遍历所有边
        source = edge.get("source", "")  # 获取边的源节点
        target = edge.get("target", "")  # 获取边的目标节点
        
        if source not in adjacency:  # 如果源节点不在邻接表中
            continue  # 跳过这条边
        
        if target not in inDegree:  # 如果目标节点不在入度表中
            continue  # 跳过这条边
        
        adjacency[source].append(target)  # 把目标节点加入源节点的邻接列表
        inDegree[target] = inDegree[target] + 1  # 目标节点的入度加1
    
    queue = deque()  # 创建队列，用于BFS
    
    for nodeId in inDegree:  # 遍历所有节点
        if inDegree[nodeId] == 0:  # 如果节点入度为0
            queue.append(nodeId)  # 加入队列
    
    result = []  # 结果列表，存储排序后的节点id
    
    while len(queue) > 0:  # 循环处理队列直到队列为空
        current = queue.popleft()  # 弹出队首节点
        result.append(current)  # 加入结果列表
        
        for neighbor in adjacency[current]:  # 遍历当前节点的所有后继节点
            inDegree[neighbor] = inDegree[neighbor] - 1  # 后继节点入度减1
            
            if inDegree[neighbor] == 0:  # 如果后继节点入度变成0
                queue.append(neighbor)  # 加入队列
    
    if len(result) != len(nodes):  # 如果结果数量不等于节点数量
        raise Exception("存在循环依赖，无法进行拓扑排序")  # 抛出异常
    
    return result  # 返回排序结果数组
