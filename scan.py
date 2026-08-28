"""
scan.py - Block 静态扫描指令

炼丹蓝图的目标是设计可重复堆叠的Block：输出张量形状必须等于输入张量形状。
扫描不评价数值好坏，只用一次无梯度追踪回答"这张蓝图能否作为Block堆叠"。
用法：result = scan.scanBlueprint(blueprint)
"""

import torch  # 无梯度追踪能力，用于按真实节点实现推导每个端口形状

import engine  # 蓝图编译能力，复用缓存模型保证扫描与运行看到同一结构


# --- 追踪一次传播并记录每个端口形状 ---
def traceShapes(model):
    nodeShapes = {}  # 保存每个节点每个张量端口的形状
    nodeErrors = []  # 保存追踪期间的结构化节点错误
    nodeResults = {}  # 当前追踪保存的端口值，仅用于形状推导
    with torch.no_grad():
        for nodeID in model.sortedIDs:
            incomingEdges = model.edgesByTarget.get(nodeID, [])  # 读取当前节点全部上游连线
            if any(edge.get("source", "") not in nodeResults for edge in incomingEdges):
                continue  # 上游没有结果时跳过，形状自然标记为不可推导
            try:
                inputValues = model._collectInputs(nodeID, nodeResults, {})  # 输入节点使用自身形状参数生成探测张量
                if model._getOpcode(nodeID) != "input" and any(value is None for value in inputValues.values()):
                    continue  # 上游明确返回空值时当前节点无法追踪
                outputValues = model.nodeModules[model.moduleKeys[nodeID]](inputValues)  # 用真实节点实现推导形状，避免维护每节点推导规则
            except Exception as error:
                nodeError = error if isinstance(error, engine.BlueprintError) else engine.BlueprintError("nodeExecutionFailed", f"节点执行失败: {error}", {"nodeId": nodeID, "opcode": model._getOpcode(nodeID)})
                nodeErrors.append(nodeError.toData())  # 记录错误但继续追踪独立分支
                continue
            nodeResults[nodeID] = outputValues  # 写入数据流供下游追踪
            nodeShapes[nodeID] = {port: list(value.shape) for port, value in outputValues.items() if isinstance(value, torch.Tensor)}  # 只记录张量端口形状
    return nodeShapes, nodeErrors  # 返回全图形状和追踪错误


# --- 汇总Block可堆叠结论 ---
def scanBlueprint(blueprint):
    """
    用法：scanBlueprint(blueprint) 返回 {"status", "issues", "nodeShapes", "inputShapes", "outputShapes"}
    status只有三种：stackable可堆叠、notStackable不可堆叠、unknown无法确定。
    """
    try:
        model = engine.compileBlueprintCached(blueprint)  # 与运行共享缓存模型，扫描结论和实际执行一致
    except engine.BlueprintError as error:
        return {"status": "notStackable", "issues": [error.toData()], "nodeShapes": {}, "inputShapes": {}, "outputShapes": {}}  # 结构无法编译时必然不可堆叠

    issues = []  # 收集影响结论的结构化问题
    nodeShapes, nodeErrors = traceShapes(model)  # 用一次探测传播得到全图端口形状
    issues.extend(nodeErrors)  # 节点执行错误意味着形状无法确定

    if not model.outputIDs:
        issues.append({"code": "missingOutput", "message": "Block必须包含输出节点", "details": {}})  # 没有输出无法定义堆叠出口

    inputShapes = {nodeID: nodeShapes.get(nodeID, {}).get("out") for nodeID in model.inputIDs}  # 每个输入节点的探测形状
    outputShapes = {}  # 每个输出节点实际接收到的形状
    validInputShapes = [shape for shape in inputShapes.values() if shape]  # 参与同形比较的输入形状
    for outputID in model.outputIDs:
        receivedShape = nodeShapes.get(outputID, {}).get("out")  # 输出节点透传上游值，out端口即接收形状
        outputShapes[outputID] = receivedShape  # 无论是否可比都反馈给前端
        if receivedShape is None:
            issues.append({"code": "outputNotReached", "message": "输出节点没有接收到任何张量", "details": {"nodeId": outputID}})  # 断路或上游错误导致出口无值
            continue
        if validInputShapes and receivedShape not in validInputShapes:
            issues.append({"code": "shapeMismatch", "message": f"输出形状{receivedShape}与输入形状{validInputShapes}不一致，Block无法堆叠", "details": {"nodeId": outputID, "outputShape": receivedShape, "inputShapes": validInputShapes}})  # 同形契约被打破

    structuralCodes = {"missingOutput", "outputNotReached", "shapeMismatch"}  # 这些问题直接判定不可堆叠
    hasStructuralIssue = any(issue.get("code") in structuralCodes for issue in issues)  # 结构问题优先于执行错误
    if hasStructuralIssue:
        status = "notStackable"  # 出口缺失或形状不同必然无法堆叠
    elif issues:
        status = "unknown"  # 只有节点执行错误时无法给出确定结论
    else:
        status = "stackable"  # 所有出口都与输入同形

    return {"status": status, "issues": issues, "nodeShapes": nodeShapes, "inputShapes": inputShapes, "outputShapes": outputShapes}  # 返回完整扫描反馈
