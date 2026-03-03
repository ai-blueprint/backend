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

from collections import deque


# 由于这个是工具函数，只需要使用，不需要了解原理，所以这里不添加详细注释
def topoSort(nodes, edges, strict=False):
    inDegree = {}
    adjacency = {}
    seen = set()

    for node in nodes:
        nodeId = node.get("id", "")
        if nodeId in seen:
            raise Exception(f"存在重复节点ID，无法进行拓扑排序: {nodeId}")
        seen.add(nodeId)
        inDegree[nodeId] = 0
        adjacency[nodeId] = []

    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")

        if source not in adjacency or target not in inDegree:
            if strict:
                raise Exception(f"存在非法边: {source} -> {target}")
            continue

        adjacency[source].append(target)
        inDegree[target] += 1

    queue = deque(nodeId for nodeId, deg in inDegree.items() if deg == 0)
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)
        for neighbor in adjacency[current]:
            inDegree[neighbor] -= 1
            if inDegree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(inDegree):
        cycleNodes = sorted(nodeId for nodeId, deg in inDegree.items() if deg > 0)
        raise Exception(f"存在循环依赖，涉及节点: {','.join(cycleNodes)}")

    return result
