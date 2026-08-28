import json  # WebSocket协议测试需要构造标准消息
import math  # 损失值需要验证为有限数字
import unittest  # 单元测试能力，用于验证真实反向传播

import torch  # 张量能力，用于检查权重和梯度变化

import engine  # 单步训练能力，是本组测试主体
import server  # 训练协议入口


def trainingBlueprint():
    return {
        "name": "训练测试",
        "nodes": [
            {"id": "input-1", "data": {"opcode": "input", "params": {"out_shape": [2, 3]}}},
            {"id": "linear-1", "data": {"opcode": "linear", "params": {"in_features": 3, "out_features": 3, "bias": True}}},
            {"id": "output-1", "data": {"opcode": "output", "params": {}}},
        ],
        "edges": [
            {"id": "edge-model", "source": "input-1", "sourceHandle": "out", "target": "linear-1", "targetHandle": "x"},
            {"id": "edge-prediction", "source": "linear-1", "sourceHandle": "out", "target": "output-1", "targetHandle": "in"},
            {"id": "edge-target", "source": "input-1", "sourceHandle": "out", "target": "output-1", "targetHandle": "target"},
        ],
    }


class FakeWebSocket:
    def __init__(self):
        self.messages = []  # 按发送顺序保存服务端JSON文本

    async def send(self, message):
        self.messages.append(json.loads(message))  # 立即解析协议以便断言字段


class TrainingStepTest(unittest.TestCase):
    def setUp(self):
        engine.clearModelCache()  # 每个用例从新模型和新优化器状态开始

    def test_training_step_updates_weight_and_returns_gradient(self):
        model = engine.compileBlueprintCached(trainingBlueprint())  # 编译含预测和目标两条分支的蓝图
        before = next(model.nodeModules[model.moduleKeys["linear-1"]].parameters()).detach().clone()  # 保存线性层训练前权重

        result = model.trainStep(maxValues=128)  # 执行一次真实前向、反向和权重更新

        after = next(model.nodeModules[model.moduleKeys["linear-1"]].parameters()).detach()  # 读取更新后的权重
        parameter = result["nodes"]["linear-1"]["parameters"][0]  # 读取线性层第一组参数反馈
        self.assertTrue(math.isfinite(result["loss"]))  # loss必须是可显示的有限数字
        self.assertFalse(torch.equal(before, after))  # 训练步骤必须真实改变权重
        self.assertEqual(parameter["gradient"]["kind"], "tensor")  # 必须返回梯度矩阵
        self.assertEqual(parameter["name"], "linear.weight")  # 参数名称必须明确属于当前节点内部的权重
        self.assertEqual(parameter["weight"]["shape"], [3, 3])  # 必须返回权重真实形状
        self.assertEqual(result["prediction"]["shape"], [2, 3])  # 预测张量形状应保持可视化
        self.assertEqual(result["target"]["shape"], [2, 3])  # 目标张量形状应保持可视化

    def test_training_without_target_is_rejected(self):
        blueprint = trainingBlueprint()  # 从完整训练蓝图开始修改
        blueprint["edges"].pop()  # 移除Target到Output的连接，模拟普通前向蓝图
        model = engine.compileBlueprintCached(blueprint)  # 编译仍允许普通前向图

        with self.assertRaises(engine.BlueprintError) as context:
            model.trainStep()  # 没有目标时不能偷偷更新权重

        self.assertEqual(context.exception.code, "trainingNotConfigured")  # 错误应明确指向训练接口未配置

    def test_target_branch_parameters_are_not_reported_as_model_gradients(self):
        blueprint = trainingBlueprint()  # 从完整训练蓝图开始增加目标分支层
        blueprint["nodes"].insert(2, {"id": "target-linear", "data": {"opcode": "linear", "params": {"in_features": 3, "out_features": 3, "bias": True}}})  # 目标分支也使用可学习层来验证隔离
        blueprint["edges"][2]["target"] = "target-linear"  # 输入先经过目标分支线性层
        blueprint["edges"][2]["targetHandle"] = "x"  # 目标层使用普通节点输入端口
        blueprint["edges"].append({"id": "edge-target-layer", "source": "target-linear", "sourceHandle": "out", "target": "output-1", "targetHandle": "target"})  # 目标层输出直接进入Output目标端口
        model = engine.compileBlueprintCached(blueprint)  # 编译新的双线性分支模型

        result = model.trainStep(maxValues=128)  # 执行一次训练并取得按节点分组的梯度

        self.assertIn("linear-1", result["nodes"])  # 预测分支线性层必须有训练反馈
        self.assertNotIn("target-linear", result["nodes"])  # 目标分支线性层不能混入模型梯度反馈

    def test_fixed_training_batch_makes_loss_decrease(self):
        model = engine.compileBlueprintCached(trainingBlueprint())  # 编译输入和目标相同的可学习映射

        losses = [model.trainStep(maxValues=128)["loss"] for _ in range(20)]  # 在同一训练批次上连续更新20步

        self.assertLess(losses[-1], losses[0])  # 固定样本下训练应有可观察的下降趋势

    def test_training_hyperparameters_are_applied(self):
        model = engine.compileBlueprintCached(trainingBlueprint())  # 编译可训练蓝图

        model.trainStep(maxValues=128, optimizerName="adam", learningRate=0.02, gradientClip=0.5)  # 使用用户指定的训练超参数

        self.assertEqual(model.optimizerConfig["name"], "adam")  # 优化器选择必须生效
        self.assertEqual(model.optimizerConfig["learningRate"], 0.02)  # 学习率必须生效
        self.assertEqual(model.optimizer.param_groups[0]["lr"], 0.02)  # PyTorch优化器实际配置必须同步


class TrainingProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_train_step_protocol_returns_training_snapshot(self):
        websocket = FakeWebSocket()  # 收集训练协议返回的单步快照

        await server.handleMessage(websocket, json.dumps({"type": "trainStep", "id": "train-1", "data": {"blueprint": trainingBlueprint(), "maxValues": 128}}))  # 通过真实协议入口训练一步

        self.assertEqual(len(websocket.messages), 1)  # 单步训练只返回一个终态消息
        self.assertEqual(websocket.messages[0]["type"], "trainStep")  # 消息类型保持请求类型
        self.assertIn("loss", websocket.messages[0]["data"])  # 返回损失值供曲线使用
        self.assertIn("nodes", websocket.messages[0]["data"])  # 返回节点权重和梯度快照

    async def test_reset_training_discards_cached_model(self):
        blueprint = trainingBlueprint()  # 构造一张可训练蓝图
        firstModel = engine.compileBlueprintCached(blueprint)  # 获取当前缓存模型
        websocket = FakeWebSocket()  # 收集重置协议反馈

        await server.handleMessage(websocket, json.dumps({"type": "resetTraining", "id": "reset-1", "data": {}}))  # 通过协议请求重置训练

        secondModel = engine.compileBlueprintCached(blueprint)  # 重置后再次编译应得到全新模型
        self.assertIsNot(firstModel, secondModel)  # 后端不能继续使用旧权重
        self.assertEqual(websocket.messages[0]["data"]["status"], "reset")  # 前端应收到成功确认

    async def test_shape_mismatch_returns_train_step_error(self):
        blueprint = trainingBlueprint()  # 构造正常训练蓝图
        blueprint["nodes"].insert(2, {"id": "target-linear", "data": {"opcode": "linear", "params": {"in_features": 3, "out_features": 4, "bias": True}}})  # 目标分支输出不同宽度制造形状冲突
        blueprint["edges"][2]["target"] = "target-linear"  # 输入先进入目标变换节点
        blueprint["edges"][2]["targetHandle"] = "x"  # 目标变换节点使用普通输入端口
        blueprint["edges"].append({"id": "edge-target-layer", "source": "target-linear", "sourceHandle": "out", "target": "output-1", "targetHandle": "target"})  # 目标变换结果进入Output目标端口
        websocket = FakeWebSocket()  # 收集训练失败消息

        await server.handleMessage(websocket, json.dumps({"type": "trainStep", "id": "shape-error", "data": {"blueprint": blueprint}}))  # 通过协议触发形状错误

        self.assertEqual(websocket.messages[0]["type"], "trainStep")  # 错误必须保留原请求类型
        self.assertEqual(websocket.messages[0]["error"]["code"], "trainingShapeMismatch")  # 前端训练处理器需要收到稳定错误代码


if __name__ == "__main__":
    unittest.main()
