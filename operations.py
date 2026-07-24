"""
operations.py - 蓝图跑分、训练、检查点和导出指令

这些函数不发送 WebSocket 消息，只读取请求数据、执行一次清晰业务动作并返回结果。
server.py 负责触发和反馈，因此耗时指令也可以在线程中安全复用。
"""

import json  # JSON 能力，用于保存蓝图、清单和 Python 导出数据
import os  # 文件替换能力，用于原子写入检查点目录
import shutil  # 目录清理能力，用于失败时回收临时文件
import tempfile  # 临时目录能力，用于避免半成品检查点
import time  # 高精度计时能力，用于跑分和训练反馈
from pathlib import Path  # 路径能力，用于约束所有产物位于 artifacts 根目录

import torch  # 张量和检查点能力，用于训练、计时与状态保存
import torch.nn as nn  # 损失和导出包装能力，用于统一模型输出

import engine  # 蓝图编译和输入解析能力，所有操作共享同一图语义


backendRoot = Path(__file__).resolve().parent  # 后端根目录是所有可信产物的共同锚点
artifactsRoot = backendRoot / "artifacts"  # 用户可写产物只能落在该目录内


# --- 校验并创建产物路径 ---
def getArtifactPath(relativePath, suffix=None):
    if not isinstance(relativePath, str) or not relativePath.strip():
        raise engine.BlueprintError("invalidPath", "artifact路径必须是非空字符串")  # 空路径无法安全定位产物
    candidate = (artifactsRoot / relativePath).resolve()  # 先解析 .. 和绝对路径影响
    if candidate == artifactsRoot.resolve() or artifactsRoot.resolve() not in candidate.parents:
        raise engine.BlueprintError("invalidPath", "artifact路径必须位于backend/artifacts内")  # 禁止目录穿越和根目录覆盖
    if suffix and candidate.suffix.lower() != suffix:
        candidate = candidate.with_suffix(suffix)  # 导出类型决定稳定扩展名
    candidate.parent.mkdir(parents=True, exist_ok=True)  # 路径通过验证后才创建父目录
    return candidate  # 后续写入只能使用已验证的绝对路径


# --- 统计模型参数规模 ---
def countParameters(model):
    totalParameters = sum(parameter.numel() for parameter in model.parameters())  # 统计全部可学习和冻结参数
    trainableParameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)  # 单独统计优化器会更新的参数
    return totalParameters, trainableParameters  # 返回两个指标供跑分和训练反馈使用


# --- 跑分蓝图延迟和参数 ---
def scoreBlueprint(blueprint, scoreConfig=None, model=None):
    scoreConfig = scoreConfig or {}  # 缺省配置提供可直接工作的本地跑分
    warmupRuns = max(0, min(int(scoreConfig.get("warmupRuns", 1)), 20))  # 限制预热次数避免恶意长任务
    measuredRuns = max(1, min(int(scoreConfig.get("runs", 5)), 100))  # 限制测量次数并保证至少执行一次
    model = model or engine.compileBlueprint(blueprint)  # 会话模型存在时测量训练或加载后的真实参数
    model.eval()  # 跑分关闭 dropout，保证输入相同时结果稳定
    modelInputs = engine.validateModelInputs(model, engine.parseInputs(scoreConfig.get("inputs")))  # 显式输入必须匹配模型边界

    with torch.no_grad():
        for _ in range(warmupRuns):
            model(modelInputs)  # 预热缓存和底层算子，不计入延迟结果
        latencies = []  # 保存每次完整图执行耗时
        for _ in range(measuredRuns):
            startedAt = time.perf_counter()  # 每次测量使用单调高精度时钟
            model(modelInputs)  # 执行完整图并保留同一组模型参数
            latencies.append((time.perf_counter() - startedAt) * 1000)  # 将秒转换为前端友好的毫秒

    totalParameters, trainableParameters = countParameters(model)  # 模型执行成功后统计参数规模
    return {
        "status": "complete",  # 明确跑分已成功完成
        "latencyMs": {"mean": sum(latencies) / len(latencies), "min": min(latencies), "max": max(latencies), "runs": measuredRuns},  # 返回聚合延迟和样本数
        "parameters": {"total": totalParameters, "trainable": trainableParameters},  # 返回总参数和可训练参数
        "outputs": engine.serializeValue(model.lastNodeResults.get(model.outputIDs[0], {}) if model.outputIDs else {}),  # 附带首个终点的有界输出用于确认执行有效
    }


# --- 创建受支持的优化器 ---
def createOptimizer(name, parameters, learningRate, optimizerConfig=None):
    optimizerConfig = optimizerConfig or {}  # 缺省配置足以启动本地训练
    normalizedName = str(name).lower()  # 优化器名称大小写不影响协议
    if normalizedName == "adam":
        return torch.optim.Adam(parameters, lr=learningRate, weight_decay=float(optimizerConfig.get("weightDecay", 0.0)))  # Adam 提供稳定默认训练体验
    if normalizedName == "adamw":
        return torch.optim.AdamW(parameters, lr=learningRate, weight_decay=float(optimizerConfig.get("weightDecay", 0.01)))  # AdamW 提供解耦权重衰减选项
    if normalizedName == "sgd":
        return torch.optim.SGD(parameters, lr=learningRate, momentum=float(optimizerConfig.get("momentum", 0.0)), weight_decay=float(optimizerConfig.get("weightDecay", 0.0)))  # SGD 支持常用动量和权重衰减
    raise engine.BlueprintError("invalidOptimizer", f"不支持的优化器: {name}")  # 拒绝动态导入任意优化器


# --- 读取训练所需标量损失 ---
def selectLoss(model, modelOutputs, trainingConfig, syntheticTarget):
    lossOutput = trainingConfig.get("lossOutput")  # 可选选择器读取任意节点的指定输出端口
    if lossOutput:
        nodeID = lossOutput.get("nodeId", "")  # 节点 ID 指向本次 forward 的完整结果
        port = lossOutput.get("port", "loss")  # 损失节点通常使用 loss 端口
        loss = model.lastNodeResults.get(nodeID, {}).get(port)  # 从显式图数据中读取用户选择值
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise engine.BlueprintError("invalidLossOutput", "lossOutput必须指向单元素张量", {"nodeId": nodeID, "port": port})  # 反向传播要求明确标量
        return loss.reshape(())  # 单元素形状统一转换为标量张量

    outputID = trainingConfig.get("outputId") or next(iter(modelOutputs), "")  # 未指定时使用首个显式模型输出
    prediction = modelOutputs.get(outputID)  # 读取要拟合合成目标的预测值
    if not isinstance(prediction, torch.Tensor):
        raise engine.BlueprintError("invalidTrainingOutput", "训练输出必须是张量", {"outputId": outputID})  # 普通对象无法计算梯度损失
    if prediction.numel() == 1 and trainingConfig.get("useScalarOutputAsLoss", False):
        return prediction.reshape(())  # 用户明确允许时直接把标量模型输出作为损失
    return nn.functional.mse_loss(prediction, syntheticTarget)  # 默认使用安全合成目标完成端到端本地训练


# --- 在线程中训练蓝图 ---
def trainBlueprint(blueprint, trainingConfig=None, progressCallback=None, cancelEvent=None):
    trainingConfig = trainingConfig or {}  # 缺省值形成可工作的安全合成训练任务
    epochs = max(1, min(int(trainingConfig.get("epochs", 3)), 1000))  # 限制总轮数防止无界请求
    batchSize = max(1, min(int(trainingConfig.get("batchSize", 32)), 4096))  # 批大小参与样本数到步数的换算
    sampleCount = max(1, min(int(trainingConfig.get("sampleCount", batchSize * 4)), 1000000))  # 合成样本数仅决定每轮更新次数
    defaultSteps = max(1, (sampleCount + batchSize - 1) // batchSize)  # 向上取整保证尾批次也对应一次更新
    stepsPerEpoch = max(1, min(int(trainingConfig.get("stepsPerEpoch", defaultSteps)), 10000))  # 显式步数仍可覆盖换算结果
    learningRate = float(trainingConfig.get("learningRate", 0.001))  # 读取优化器基础学习率
    if not 0 < learningRate <= 10:
        raise engine.BlueprintError("invalidLearningRate", "learningRate必须在(0, 10]范围内")  # 拒绝明显危险或无效学习率

    model = engine.compileBlueprint(blueprint)  # 持久模型在整个训练期间复用同一组参数
    model.train()  # 启用 dropout 和训练态归一化行为
    trainableParameters = [parameter for parameter in model.parameters() if parameter.requires_grad]  # 只把可训练参数交给优化器
    if not trainableParameters:
        raise engine.BlueprintError("noTrainableParameters", "蓝图没有可训练参数")  # 没有参数时训练没有意义
    optimizer = createOptimizer(trainingConfig.get("optimizer", "adam"), trainableParameters, learningRate, trainingConfig.get("optimizerConfig"))  # 创建白名单优化器
    providedInputs = trainingConfig.get("inputs")  # 固定输入可用于可复现实验
    randomShapes = trainingConfig.get("inputShapes", {})  # 每个输入节点可单独指定安全合成形状
    startedAt = time.perf_counter()  # 记录完整训练时长
    finalLoss = None  # 完成反馈需要最后一步损失

    for epochIndex in range(epochs):
        for stepIndex in range(stepsPerEpoch):
            if cancelEvent and cancelEvent.is_set():
                return {"status": "cancelled", "epoch": epochIndex, "step": stepIndex, "durationMs": (time.perf_counter() - startedAt) * 1000}  # 在批次边界安全取消
            if providedInputs is not None:
                modelInputs = engine.parseInputs(providedInputs)  # 用户数据每步重新构造，避免意外原地修改
            else:
                modelInputs = {}
                for inputID in model.inputIDs:
                    defaultShape = engine.getInputShape(model, inputID)  # 从编译节点读取已解包的正整数形状
                    modelInputs[inputID] = torch.randn(randomShapes.get(inputID, defaultShape))  # 为每步生成独立安全合成样本

            optimizer.zero_grad(set_to_none=True)  # 清空旧梯度并减少不必要内存写入
            modelOutputs = model(modelInputs)  # 前向执行完整蓝图并保留端口结果
            outputID = trainingConfig.get("outputId") or next(iter(modelOutputs), "")  # 默认训练首个显式输出
            prediction = modelOutputs.get(outputID)  # 合成目标必须与实际预测形状一致
            syntheticTarget = torch.zeros_like(prediction) if isinstance(prediction, torch.Tensor) else None  # 零目标无需外部数据依赖且可稳定反向
            loss = selectLoss(model, modelOutputs, trainingConfig, syntheticTarget)  # 将用户选择或默认 MSE 收口为标量损失
            loss.backward()  # 沿图反向计算所有持久模块梯度
            maxGradientNorm = trainingConfig.get("maxGradientNorm", 1.0)  # 默认裁剪避免合成训练数值爆炸
            if maxGradientNorm is not None:
                torch.nn.utils.clip_grad_norm_(trainableParameters, float(maxGradientNorm))  # 在优化前限制整体梯度范数
            optimizer.step()  # 根据本步梯度更新 ModuleDict 中的持久参数

            finalLoss = float(loss.detach().cpu())  # 反馈值脱离计算图并转换为 JSON 数值
            if progressCallback:
                completedSteps = epochIndex * stepsPerEpoch + stepIndex + 1  # 已完成步数用于前端展示连续百分比
                totalSteps = epochs * stepsPerEpoch  # 总步数由轮数和每轮步数共同决定
                progressCallback({"epoch": epochIndex + 1, "epochs": epochs, "step": stepIndex + 1, "stepsPerEpoch": stepsPerEpoch, "loss": finalLoss, "percent": completedSteps / totalSteps * 100})  # 每步反馈明确进度和损失

    totalParameters, trainableCount = countParameters(model)  # 完成后报告实际模型规模
    return {"status": "complete", "loss": finalLoss, "durationMs": (time.perf_counter() - startedAt) * 1000, "parameters": {"total": totalParameters, "trainable": trainableCount}, "model": model}  # 内部携带模型供可选检查点保存


# --- 保存检查点目录 ---
def saveCheckpoint(blueprint, relativePath, model=None, metadata=None):
    targetPath = getArtifactPath(relativePath)  # 路径验证必须先于任何文件写入
    model = model or engine.compileBlueprint(blueprint)  # 未提供训练模型时保存新编译初始状态
    tempPath = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=targetPath.parent))  # 同一父目录保证替换尽量原子
    try:
        (tempPath / "blueprint.json").write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")  # 蓝图定义使用可读 JSON 保存
        torch.save(model.state_dict(), tempPath / "state_dict.pt")  # 参数只保存 state_dict，避免反序列化任意模型对象
        manifest = {"formatVersion": 1, "createdAt": time.time(), "files": {"blueprint": "blueprint.json", "stateDict": "state_dict.pt"}, "metadata": metadata or {}}  # 清单描述格式和文件归属
        (tempPath / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")  # 最后写清单表示临时目录完整
        if targetPath.exists():
            shutil.rmtree(targetPath)  # 目标路径已经验证且仅删除当前检查点目录
        os.replace(tempPath, targetPath)  # 完整目录一次替换到最终位置
    except Exception:
        shutil.rmtree(tempPath, ignore_errors=True)  # 写入失败时不留下临时半成品
        raise
    return {"status": "complete", "path": str(targetPath.relative_to(backendRoot)), "manifest": manifest}  # 只返回后端相对路径避免泄露环境路径


# --- 加载检查点目录 ---
def loadCheckpoint(relativePath):
    checkpointPath = getArtifactPath(relativePath)  # 同一验证规则阻止目录穿越
    manifestPath = checkpointPath / "manifest.json"  # 清单是检查点完整性的入口
    if not manifestPath.is_file():
        raise engine.BlueprintError("checkpointNotFound", "检查点manifest.json不存在")  # 非检查点目录不能加载
    manifest = json.loads(manifestPath.read_text(encoding="utf-8"))  # 读取格式版本和受控文件名
    if manifest.get("formatVersion") != 1:
        raise engine.BlueprintError("unsupportedCheckpoint", "不支持的检查点格式版本")  # 防止错误解释未来格式
    blueprintPath = checkpointPath / manifest["files"]["blueprint"]  # 文件名来自已验证版本清单
    statePath = checkpointPath / manifest["files"]["stateDict"]  # state_dict 文件不得越出检查点目录
    if checkpointPath.resolve() not in blueprintPath.resolve().parents or checkpointPath.resolve() not in statePath.resolve().parents:
        raise engine.BlueprintError("invalidCheckpoint", "检查点文件路径不合法")  # 清单内容也必须通过路径边界检查
    blueprint = json.loads(blueprintPath.read_text(encoding="utf-8"))  # 先恢复图定义再创建相同模块结构
    model = engine.compileBlueprint(blueprint)  # 根据蓝图构造可信类，不从文件反序列化类
    stateDictionary = torch.load(statePath, map_location="cpu", weights_only=True)  # 仅允许权重张量并固定加载到 CPU
    model.load_state_dict(stateDictionary, strict=True)  # 严格校验参数名和形状防止静默错配
    return {"status": "complete", "path": str(checkpointPath.relative_to(backendRoot)), "blueprint": blueprint, "manifest": manifest, "model": model}  # 内部模型可继续执行或导出


class ExportModel(nn.Module):
    """把字典输入输出转换为 ONNX 接受的位置张量接口。"""

    def __init__(self, model):
        super().__init__()
        self.model = model  # 子模块注册保证 ONNX 能遍历全部参数

    def forward(self, *inputValues):
        modelInputs = dict(zip(self.model.inputIDs, inputValues))  # 按编译时输入顺序还原命名映射
        modelOutputs = self.model(modelInputs)  # 复用标准图路由
        return tuple(modelOutputs[outputID] for outputID in self.model.outputIDs)  # 按显式输出顺序返回张量元组


# --- 导出可直接运行的 Python 文件 ---
def exportPython(blueprint, relativePath, model=None):
    targetPath = getArtifactPath(relativePath, ".py")  # Python 产物只能写入 artifacts 并使用 .py 扩展名
    blueprintJSON = json.dumps(blueprint, ensure_ascii=True, separators=(",", ":"))  # ASCII JSON 可安全嵌入 Python 字符串
    model = model or engine.compileBlueprint(blueprint)  # 会话模型存在时导出训练或加载后的真实参数
    weightsPath = targetPath.with_suffix(".pt")  # 权重与Python入口使用相同文件名主体
    torch.save(model.state_dict(), weightsPath)  # 只保存state_dict避免反序列化任意代码
    source = f'''"""AI Blueprint 自动导出文件。"""\nimport json\nimport sys\nfrom pathlib import Path\n\nimport torch\n\ncurrentPath = Path(__file__).resolve()\nbackendRoot = next((parent for parent in currentPath.parents if (parent / "engine.py").is_file()), None)\nif backendRoot is None:\n    raise RuntimeError("导出文件必须保留在包含engine.py的后端artifacts目录中")\nif str(backendRoot) not in sys.path:\n    sys.path.insert(0, str(backendRoot))\n\nimport engine\n\nblueprint = json.loads({blueprintJSON!r})\nweightsPath = currentPath.with_suffix(".pt")\n\ndef createModel():\n    model = engine.compileBlueprint(blueprint)\n    model.load_state_dict(torch.load(weightsPath, map_location="cpu", weights_only=True), strict=True)\n    model.eval()\n    return model\n\nif __name__ == "__main__":\n    model = createModel()\n    print(model())\n'''  # 导出文件加载配套权重后提供可直接复用模型
    targetPath.write_text(source, encoding="utf-8")  # 一次写入完整可执行 Python 模块
    return {"status": "complete", "format": "python", "path": str(targetPath.relative_to(backendRoot)), "weightsPath": str(weightsPath.relative_to(backendRoot))}  # 返回入口和配套权重路径


# --- 导出 ONNX 模型 ---
def exportONNX(blueprint, relativePath, exportConfig=None, model=None):
    exportConfig = exportConfig or {}  # 显式输入决定 ONNX 示例形状
    try:
        import onnx  # noqa: F401  # ONNX 包是导出的可选依赖，运行服务本身不需要
    except ImportError as error:
        raise engine.BlueprintError("exportDependencyMissing", "ONNX导出需要可选依赖，请运行: uv sync --extra onnx", {"dependency": "onnx"}) from error

    model = model or engine.compileBlueprint(blueprint)  # 优先导出训练或加载后的会话模型
    model.eval()  # 导出固定为推理行为
    explicitInputs = engine.parseInputs(exportConfig.get("inputs"))  # 推荐用户提供可复现输入形状
    sampleInputs = []  # ONNX 使用位置参数样例追踪模型
    for inputID in model.inputIDs:
        if inputID in explicitInputs:
            sampleInputs.append(explicitInputs[inputID])  # 使用协议提供的具体样例
            continue
        shape = engine.getInputShape(model, inputID)  # 从编译节点读取已解包的示例形状
        sampleInputs.append(torch.randn(shape))  # 仅用于追踪形状的本地随机张量
    if not sampleInputs:
        raise engine.BlueprintError("onnxNotExportable", "ONNX导出至少需要一个InputNode")  # 无位置输入的图不符合当前导出包装

    targetPath = getArtifactPath(relativePath, ".onnx")  # 依赖检查完成后才创建导出父目录
    try:
        torch.onnx.export(ExportModel(model), tuple(sampleInputs), targetPath, input_names=model.inputIDs, output_names=model.outputIDs, opset_version=int(exportConfig.get("opsetVersion", 18)), dynamo=False)  # 使用兼容面更广的传统导出器
    except Exception as error:
        targetPath.unlink(missing_ok=True)  # 导出失败不能留下看似有效的半成品
        raise engine.BlueprintError("onnxNotExportable", f"蓝图无法导出ONNX: {error}") from error
    return {"status": "complete", "format": "onnx", "path": str(targetPath.relative_to(backendRoot))}  # 返回受控相对路径
