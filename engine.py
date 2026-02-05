"""
engine.py - 蓝图执行引擎

用法：
    import engine
    await engine.run(blueprint, onMessage, onError)  # 运行蓝图

示例：
    async def onMsg(nodeId, result):
        print(f"节点{nodeId}执行完成: {result}")
    async def onErr(nodeId, error):
        print(f"节点{nodeId}执行出错: {error}")
    await engine.run(blueprintData, onMsg, onErr)
"""

import loader  # 加载器模块，用于加载所有节点
import registry  # 注册表模块，用于获取节点定义和创建节点实例
import sort  # 拓扑排序模块，用于确定执行顺序


loader.loadAll()  # 初始化时加载所有节点模块，装饰器会自动注册节点


async def run(blueprint, onMessage, onError):  # 异步运行蓝图的主函数
    """
    运行蓝图

    用法：
        await run(blueprint, onMessage, onError)

    示例：
        await run(
            {"nodes": [...], "edges": [...]},  # 蓝图数据
            async def(nodeId, result): pass,   # 节点完成回调
            async def(nodeId, error): pass     # 节点错误回调
        )
    """
    
    nodes = blueprint.get("nodes", [])  # 从蓝图中提取节点列表
    edges = blueprint.get("edges", [])  # 从蓝图中提取边列表

    sortedIds = sort.topoSort(nodes, edges)  # 调用拓扑排序，得到执行顺序
    print(f"拓扑排序结果: {sortedIds}")  # 打印排序结果用于调试

    nodeMap = {}  # 创建节点id到节点数据的映射字典
    for node in nodes:  # 遍历所有节点
        nodeId = node.get("id", "")  # 获取节点id
        nodeMap[nodeId] = node  # 存入映射字典方便后续查找

    instances = {}  # 存储所有节点的实例，格式：{nodeId: BaseNode实例}
    results = {}  # 存储所有节点的输出结果，格式：{nodeId: {port: value}}

    # ========== 阶段1：创建所有节点实例 ==========
    print("开始创建节点实例...")  # 打印阶段信息
    for nodeId in sortedIds:  # 按拓扑顺序遍历节点id
        node = nodeMap.get(nodeId)  # 根据id获取节点数据
        if node is None:  # 如果找不到节点数据
            await onError(nodeId, f"节点数据不存在: {nodeId}")  # 发送错误回调
            return  # 终止执行

        data = node.get("data", {})  # 获取节点的data字段
        opcode = data.get("opcode", "")  # 从data中获取opcode
        params = data.get("params", {})  # 从data中获取params参数字典

        if opcode not in registry.nodes:  # 检查opcode是否已注册
            await onError(nodeId, f"未知的节点类型: {opcode}")  # 发送错误回调
            return  # 终止执行

        try:  # 尝试创建节点实例
            instance = registry.createNode(opcode, nodeId, params)  # 调用registry创建实例
            instances[nodeId] = instance  # 存入实例字典
            print(f"节点实例创建成功: {nodeId} ({opcode})")  # 打印成功信息
        except Exception as e:  # 如果创建失败
            await onError(nodeId, f"创建节点实例失败: {str(e)}")  # 发送错误回调
            return  # 终止执行

    # ========== 阶段2：按拓扑顺序执行所有节点 ==========
    print("开始执行节点...")  # 打印阶段信息
    for nodeId in sortedIds:  # 按拓扑顺序遍历节点id
        instance = instances.get(nodeId)  # 获取当前节点的实例
        if instance is None:  # 如果实例不存在（理论上不会发生）
            await onError(nodeId, f"节点实例不存在: {nodeId}")  # 发送错误回调
            return  # 终止执行

        # 收集当前节点的输入
        inputValues = {}  # 创建空字典准备装输入值
        for edge in edges:  # 遍历所有边
            targetId = edge.get("target", "")  # 获取边的目标节点id
            if targetId != nodeId:  # 如果目标不是当前节点
                continue  # 跳过这条边

            sourceId = edge.get("source", "")  # 获取源节点id
            sourcePort = edge.get("sourceHandle", "out")  # 获取源端口名，默认out
            targetPort = edge.get("targetHandle", "in")  # 获取目标端口名，默认in

            sourceResult = results.get(sourceId, {})  # 获取源节点的输出结果字典
            value = sourceResult.get(sourcePort, None)  # 获取对应端口的值
            inputValues[targetPort] = value  # 存入输入字典

        # 执行节点的compute方法
        try:  # 尝试执行计算
            output = instance.compute(inputValues)  # 调用实例的compute方法
            results[nodeId] = output  # 存储输出结果
            await onMessage(nodeId, "正常")  # 发送成功回调
            print(f"节点执行成功: {nodeId}, 输出: {output}")  # 打印执行结果
        except Exception as e:  # 如果执行出错
            await onError(nodeId, f"执行出错: {str(e)}")  # 发送错误回调
            print(f"节点执行失败: {nodeId}, 错误: {str(e)}")  # 打印错误信息
            return  # 终止执行

    print("蓝图执行完成")  # 打印完成信息
    return "运行正常"