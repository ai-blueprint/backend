"""
context.py - 执行上下文

用法：
    from context import Context
    ctx = Context(blueprintData)  # 创建执行上下文
    inputs = ctx.getNodeInputs(nodeId)  # 获取节点的输入值
    
示例：
    ctx = Context({"nodes": [...], "edges": [...]})
    ctx.shapes["node1"] = {"out": [32, 64]}  # 存储形状推断结果
    ctx.layers["node1"] = linearLayer  # 存储构建的层
    ctx.results["node1"] = {"out": tensor}  # 存储执行结果
    inputs = ctx.getNodeInputs("node2")  # 获取node2的输入
"""


class Context:
    """
    执行上下文类 - 存储蓝图执行过程中的所有状态
    
    用法：
        ctx = Context(blueprintData)
        
    示例：
        ctx = Context({"nodes": [...], "edges": [...]})
        ctx.shapes["node1"]["out"] = [32, 64]
        ctx.layers["node1"] = nn.Linear(32, 64)
        ctx.results["node1"]["out"] = tensor
    """
    
    def __init__(self, blueprintData):
        """
        初始化执行上下文
        
        用法：
            ctx = Context(blueprintData)
            
        示例：
            ctx = Context({"nodes": [{"id": "node1"}], "edges": [{"source": "node1", "target": "node2"}]})
        """
        self.blueprint = blueprintData  # 存储蓝图数据，包含nodes和edges
        self.shapes = {}  # 存每个节点的形状，格式：{nodeId: {port: shapeValue}}
        self.layers = {}  # 存已构建的层，格式：{nodeId: layer}
        self.results = {}  # 存每个节点的输出，格式：{nodeId: {port: outputValue}}
    
    def getNodeInputs(self, nodeId):
        """
        获取节点的输入值
        
        用法：
            inputs = ctx.getNodeInputs("node2")
            
        示例：
            inputs = ctx.getNodeInputs("node2")  # 返回{"x": tensor, "y": tensor}
            # 会根据edges找到连接到node2的所有边，从results里取对应的值
        """
        inputs = {}  # 创建空字典准备装输入值
        edges = self.blueprint.get("edges", [])  # 从蓝图中获取所有边
        
        for edge in edges:  # 遍历所有边
            
            targetId = edge.get("target", "")  # 获取边的目标节点id
            if targetId != nodeId:  # 如果目标不是当前节点
                continue  # 跳过这条边
            
            sourceId = edge.get("source", "")  # 获取源节点id
            sourcePort = edge.get("sourceHandle", "out")  # 获取源端口，默认out
            targetPort = edge.get("targetHandle", "in")  # 获取目标端口，默认in
            
            sourceResults = self.results.get(sourceId, {})  # 获取源节点的输出结果
            value = sourceResults.get(sourcePort, None)  # 获取对应端口的值
            inputs[targetPort] = value  # 把值存入inputs字典
        
        return inputs  # 返回收集到的输入字典
