"""
sort.py - 拓扑排序

用法：
    import sort
    sortedIds = sort.topoSort(nodes, edges)  # 获取拓扑排序后的节点id列表
    sortedIds = sort.topoSort(nodes, edges, strict=True)  # 严格模式下遇到非法边会直接报错

示例：
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
    result = sort.topoSort(nodes, edges)  # 返回["a", "b", "c"]
"""

from collections import deque  # 双端队列，用于BFS


def topoSort(nodes, edges, strict=False):
    inDegree = {}  # 入度表，记录每个节点有多少个前置节点
    adjacency = {}  # 邻接表，记录每个节点指向哪些后继节点
    nodeIds = set()  # 节点id集合，用于检查重复id

    for node in nodes:  # 遍历所有节点
        nodeId = node.get("id", "")  # 获取节点id
        if nodeId in nodeIds:  # 如果节点id已经出现过
            raise Exception(f"存在重复节点ID，无法进行拓扑排序: {nodeId}")  # 抛出重复节点错误，避免图结构歧义
        nodeIds.add(nodeId)  # 记录当前节点id
        inDegree[nodeId] = 0  # 初始化入度为0
        adjacency[nodeId] = []  # 初始化邻接列表为空

    for edge in edges:  # 遍历所有边
        source = edge.get("source", "")  # 获取边的源节点
        target = edge.get("target", "")  # 获取边的目标节点

        if source not in adjacency:  # 如果源节点不在邻接表中
            if strict:  # 严格模式下直接报错，提示边的源节点非法
                raise Exception(f"存在非法边，源节点不存在: {source} -> {target}")  # 抛出非法边错误
            continue  # 跳过这条边

        if target not in inDegree:  # 如果目标节点不在入度表中
            if strict:  # 严格模式下直接报错，提示边的目标节点非法
                raise Exception(f"存在非法边，目标节点不存在: {source} -> {target}")  # 抛出非法边错误
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
        cycleNodeIds = sorted([nodeId for nodeId, degree in inDegree.items() if degree > 0])  # 找出仍然有入度的节点，它们都在循环依赖中
        cycleNodesText = ",".join(cycleNodeIds)  # 转成可读字符串方便日志和前端展示
        raise Exception(f"存在循环依赖，无法进行拓扑排序，涉及节点: {cycleNodesText}")  # 抛出包含节点列表的异常

    return result  # 返回排序结果数组
