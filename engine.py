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
import registry  # 注册表模块，用于获取节点定义
import sort  # 拓扑排序模块，用于确定执行顺序
from context import Context  # 执行上下文类


loader.loadAll()  # 初始化时加载所有节点模块，装饰器会自动注册节点


async def run(blueprint, onMessage, onError):
    """
    运行蓝图

    用法：
        await run(blueprint, onMessage, onError)

    示例：
        await run(
            {"nodes": [...], "edges": [...]},  # 蓝图数据
            async def(nodeId, result): pass,  # 节点完成回调
            async def(nodeId, error): pass  # 节点错误回调
        )
    """
    
    nodes = blueprint.get("nodes", [])  # 从蓝图中提取节点列表
    edges = blueprint.get("edges", [])  # 从蓝图中提取边列表

    sortedIds = sort.topoSort(nodes, edges)  # 调用拓扑排序，得到执行顺序
    print(f"拓扑排序结果: {sortedIds}")  # 打印排序结果

    ctx = Context(blueprint)  # 创建执行上下文

    nodeMap = {}  # 创建节点id到节点数据的映射
    for node in nodes:  # 遍历所有节点
        nodeId = node.get("id", "")  # 获取节点id
        nodeMap[nodeId] = node  # 存入映射字典

    # 形状推断阶段
    print("开始形状推断...")  # 打印阶段信息
    for nodeId in sortedIds:  # 按拓扑顺序遍历节点
        node = nodeMap.get(nodeId)  # 获取节点数据
        opcode = node.get("data", {}).get("opcode")  # 从data中获取opcode（如input、output、debug）
        params = node.get("data", {}).get("params", {})  # 获取节点参数

        nodeDef = registry.nodes.get(opcode)  # 从注册表获取节点定义
        if nodeDef is None:  # 如果找不到节点定义
            print(f"节点{nodeId}未注册")
            continue  # 跳过这个节点

        func = nodeDef.get("func")  # 获取节点的func
        if func is None:  # 如果没有func
            print(f"节点{nodeId}没有func")
            continue  # 跳过这个节点

        infer = func.get("infer")  # 获取infer函数
        if infer is None:  # 如果没有infer函数
            print(f"节点{nodeId}没有infer函数")
            continue  # 跳过形状推断

        inputShapes = {}  # 收集输入形状
        for edge in edges:  # 遍历所有边
            if edge.get("target", "") != nodeId:  # 如果目标不是当前节点
                continue  # 跳过这条边
            sourceId = edge.get("source", "")  # 获取源节点id
            sourcePort = edge.get("sourceHandle", "out")  # 获取源端口
            targetPort = edge.get("targetHandle", "in")  # 获取目标端口
            sourceShapes = ctx.shapes.get(sourceId, {})  # 获取源节点的形状字典，默认空字典
            sourceShape = sourceShapes.get(sourcePort) if sourceShapes else None  # 安全获取对应端口的形状
            inputShapes[targetPort] = sourceShape  # 存入输入形状字典

        shape = infer(inputShapes, params)  # 调用infer函数进行形状推断
        ctx.shapes[nodeId] = shape  # 存入context.shapes
        print(f"节点{nodeId}形状推断: {shape}")  # 打印推断结果

    # 构建层实例阶段
    print("开始构建层...")  # 打印阶段信息
    for nodeId in sortedIds:  # 按拓扑顺序遍历节点
        node = nodeMap.get(nodeId)  # 获取节点数据
        opcode = node.get("data", {}).get("opcode")  # 从data中获取opcode（如input、output、debug）
        params = node.get("data", {}).get("params", {})  # 获取节点参数

        nodeDef = registry.nodes.get(opcode)  # 从注册表获取节点定义
        if nodeDef is None:  # 如果找不到节点定义
            continue  # 跳过这个节点

        func = nodeDef.get("func")  # 获取节点的func
        if func is None:  # 如果没有func
            continue  # 跳过这个节点

        build = func.get("build")  # 获取build函数
        if build is None:  # 如果没有build函数
            continue  # 跳过层构建

        shape = ctx.shapes.get(nodeId)  # 获取当前节点的形状
        layer = build(shape, params)  # 调用build函数构建层
        ctx.layers[nodeId] = layer  # 存入context.layers
        print(f"节点{nodeId}层构建完成")  # 打印构建结果
    
    print(f"上下文状态: {ctx.shapes}")  # 打印上下文状态
    
    # 逐节点执行阶段
    print("开始执行节点...")  # 打印阶段信息
    for nodeId in sortedIds:  # 按拓扑顺序遍历节点
        node = nodeMap.get(nodeId)  # 获取节点数据
        opcode = node.get("data", {}).get("opcode")  # 从data中获取opcode（如input、output、debug）

        nodeDef = registry.nodes.get(opcode)  # 从注册表获取节点定义
        if nodeDef is None:  # 如果找不到节点定义
            errorMsg = f"未知的节点类型: {opcode}"  # 构造错误信息
            await onError(nodeId, errorMsg)  # 发送错误
            break  # 终止执行

        func = nodeDef.get("func")  # 获取节点的func
        if func is None:  # 如果没有func
            errorMsg = f"节点{opcode}没有定义执行函数"  # 构造错误信息
            await onError(nodeId, errorMsg)  # 发送错误
            break  # 终止执行

        compute = func.get("compute")  # 获取compute函数
        if compute is None:  # 如果没有compute函数
            errorMsg = f"节点{opcode}没有定义compute函数"  # 构造错误信息
            await onError(nodeId, errorMsg)  # 发送错误
            break  # 终止执行

        try:  # 尝试执行节点
            nodeInputs = ctx.getNodeInputs(nodeId)  # 获取节点的输入值
            layer = ctx.layers.get(nodeId)  # 获取已构建的层
            result = compute(nodeInputs, layer)  # 调用compute函数执行计算（inputs, layer）
            ctx.results[nodeId] = result  # 存入context.results
            await onMessage(nodeId, result)  # 发送节点执行结果
            print(f"节点{nodeId}执行完成")  # 打印执行结果
        except Exception as e:  # 如果执行出错
            errorMsg = str(e)  # 获取错误信息
            await onError(nodeId, errorMsg)  # 发送错误
            print(f"节点{nodeId}执行出错: {errorMsg}")  # 打印错误信息
            break  # 终止执行，跳出遍历

    print("蓝图执行完成")  # 打印完成信息
