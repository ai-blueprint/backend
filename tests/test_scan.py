import unittest  # 单元测试能力，用于锁定模型缓存和Block扫描契约

import torch  # 张量能力，用于比较连续运行的权重身份

import engine  # 模型缓存能力，是本组测试主体之一
import scan  # Block静态扫描能力，是本组测试主体之一
from blueprints import linearBlueprint  # 共享最小蓝图，减少重复图噪声


class ModelCacheTest(unittest.TestCase):
    def setUp(self):
        engine.clearModelCache()  # 每个用例从空缓存开始，避免跨用例串扰

    # --- 验证蓝图未变时复用同一模型和权重 ---
    def test_same_blueprint_reuses_model_and_weights(self):
        blueprint = linearBlueprint()  # 含可训练线性层的最小蓝图

        firstModel = engine.compileBlueprintCached(blueprint)
        secondModel = engine.compileBlueprintCached(linearBlueprint())  # 内容相同的新对象也应命中缓存

        self.assertIs(firstModel, secondModel)  # 连续观察时权重保持稳定
        self.assertIs(next(firstModel.parameters()), next(secondModel.parameters()))

    # --- 验证节点ID固定种子让重新编译得到相同权重 ---
    def test_same_node_id_rebuilds_identical_weights(self):
        firstModel = engine.compileBlueprint(linearBlueprint())  # 直接编译两次而非命中缓存
        secondModel = engine.compileBlueprint(linearBlueprint())  # 每个节点按自身ID设定初始化种子

        firstWeights = list(firstModel.parameters())
        secondWeights = list(secondModel.parameters())

        self.assertTrue(all(torch.equal(first, second) for first, second in zip(firstWeights, secondWeights)))  # 独立编译的权重逐张量一致

    # --- 验证不同节点ID获得不同初始化权重 ---
    def test_different_node_ids_get_different_weights(self):
        blueprint = linearBlueprint()
        renamed = linearBlueprint()
        renamed["nodes"][1]["id"] = "linear-renamed"  # 只改线性层节点ID
        renamed["edges"][0]["target"] = "linear-renamed"  # 同步修正连线目标
        renamed["edges"][1]["source"] = "linear-renamed"  # 同步修正连线来源

        firstWeight = next(engine.compileBlueprint(blueprint).parameters())
        secondWeight = next(engine.compileBlueprint(renamed).parameters())

        self.assertFalse(torch.equal(firstWeight, secondWeight))  # 不同ID的节点不共享同一初始化

    # --- 验证固定种子不影响每轮随机输入 ---
    def test_random_inputs_stay_different_across_runs(self):
        model = engine.compileBlueprint(linearBlueprint())  # 节点初始化使用隔离随机状态

        firstOutput = model()["output-1"]  # 第一轮随机输入
        secondOutput = model()["output-1"]  # 第二轮随机输入

        self.assertFalse(torch.equal(firstOutput, secondOutput))  # 权重固定但输入仍每轮变化

    # --- 验证参数变化后重新编译 ---
    def test_changed_params_rebuild_model(self):
        firstModel = engine.compileBlueprintCached(linearBlueprint())
        changed = linearBlueprint()
        changed["nodes"][1]["data"]["params"]["out_features"] = 4  # 修改层宽属于业务变化

        secondModel = engine.compileBlueprintCached(changed)

        self.assertIsNot(firstModel, secondModel)  # 蓝图变化必须重新初始化

    # --- 验证变量值变化后重新编译 ---
    def test_changed_variable_rebuilds_model(self):
        blueprint = linearBlueprint()
        blueprint["variables"] = [{"id": "var-d", "name": "D", "value": 3}]  # 变量参与缓存键
        blueprint["nodes"][1]["data"]["params"]["in_features"] = {"expr": "D"}  # 线性层宽度引用变量
        firstModel = engine.compileBlueprintCached(blueprint)

        changed = linearBlueprint()
        changed["variables"] = [{"id": "var-d", "name": "D", "value": 3}]
        changed["nodes"][1]["data"]["params"]["in_features"] = {"expr": "D"}
        changed["variables"][0]["value"] = 5  # 只改变量值

        secondModel = engine.compileBlueprintCached(changed)

        self.assertIsNot(firstModel, secondModel)  # 变量值变化等同结构变化

    # --- 验证清空缓存后重新编译 ---
    def test_clear_cache_forces_rebuild(self):
        firstModel = engine.compileBlueprintCached(linearBlueprint())
        engine.clearModelCache()  # 模拟节点热重载后的缓存失效

        secondModel = engine.compileBlueprintCached(linearBlueprint())

        self.assertIsNot(firstModel, secondModel)  # 旧模型不能继续使用过期节点类


class BlockScanTest(unittest.TestCase):
    def setUp(self):
        engine.clearModelCache()  # 扫描与运行共享缓存，用例之间互不影响

    # --- 构造输入输出同形的最小Block ---
    def createStackableBlueprint(self):
        return {
            "nodes": [
                {"id": "input-1", "data": {"opcode": "input", "params": {"out_shape": [2, 4, 8]}}},
                {"id": "linear-1", "data": {"opcode": "linear", "params": {"in_features": 8, "out_features": 8, "bias": True}}},
                {"id": "output-1", "data": {"opcode": "output", "params": {}}},
            ],
            "edges": [
                {"source": "input-1", "sourceHandle": "out", "target": "linear-1", "targetHandle": "x"},
                {"source": "linear-1", "sourceHandle": "out", "target": "output-1", "targetHandle": "in"},
            ],
        }  # 输入经同宽线性层回到输出，形状保持[2, 4, 8]

    # --- 验证同形Block判定为可堆叠 ---
    def test_same_shape_block_is_stackable(self):
        result = scan.scanBlueprint(self.createStackableBlueprint())

        self.assertEqual(result["status"], "stackable")
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["inputShapes"]["input-1"], [2, 4, 8])
        self.assertEqual(result["outputShapes"]["output-1"], [2, 4, 8])
        self.assertEqual(result["nodeShapes"]["linear-1"]["out"], [2, 4, 8])

    # --- 验证形状不一致判定为不可堆叠 ---
    def test_shape_mismatch_is_not_stackable(self):
        blueprint = self.createStackableBlueprint()
        blueprint["nodes"][1]["data"]["params"]["out_features"] = 4  # 出口宽度不再等于入口宽度

        result = scan.scanBlueprint(blueprint)

        self.assertEqual(result["status"], "notStackable")
        self.assertEqual(result["issues"][0]["code"], "shapeMismatch")
        self.assertEqual(result["issues"][0]["details"]["outputShape"], [2, 4, 4])

    # --- 验证缺少输出节点判定为不可堆叠 ---
    def test_missing_output_is_not_stackable(self):
        blueprint = self.createStackableBlueprint()
        blueprint["nodes"] = blueprint["nodes"][:2]  # 移除输出节点
        blueprint["edges"] = blueprint["edges"][:1]  # 移除到输出的连线

        result = scan.scanBlueprint(blueprint)

        self.assertEqual(result["status"], "notStackable")
        self.assertEqual(result["issues"][0]["code"], "missingOutput")

    # --- 验证节点执行错误时结论为无法确定 ---
    def test_node_error_makes_result_unknown(self):
        blueprint = self.createStackableBlueprint()
        blueprint["nodes"][1]["data"]["params"]["in_features"] = 99  # 线性层与输入末维不匹配

        result = scan.scanBlueprint(blueprint)

        self.assertEqual(result["status"], "notStackable")  # 出口收不到值同样是结构问题
        codes = [issue["code"] for issue in result["issues"]]
        self.assertIn("nodeExecutionFailed", codes)
        self.assertIn("outputNotReached", codes)

    # --- 验证编译失败直接判定不可堆叠 ---
    def test_compile_failure_is_not_stackable(self):
        blueprint = self.createStackableBlueprint()
        blueprint["nodes"][0]["data"]["params"]["out_shape"] = {"expr": "MISSING"}  # 未定义变量导致编译失败

        result = scan.scanBlueprint(blueprint)

        self.assertEqual(result["status"], "notStackable")
        self.assertEqual(result["issues"][0]["code"], "invalidExpression")

    # --- 验证变量蓝图扫描结果跟随变量值 ---
    def test_scan_uses_variables_for_shapes(self):
        blueprint = self.createStackableBlueprint()
        blueprint["variables"] = [{"id": "var-d", "name": "D", "value": 8}]  # 特征宽度变量
        blueprint["nodes"][0]["data"]["params"]["out_shape"] = {"expr": "2, 4, D"}  # 输入形状引用变量
        blueprint["nodes"][1]["data"]["params"]["in_features"] = {"expr": "D"}  # 线性层入口跟随D
        blueprint["nodes"][1]["data"]["params"]["out_features"] = {"expr": "D"}  # 线性层出口跟随D

        result = scan.scanBlueprint(blueprint)

        self.assertEqual(result["status"], "stackable")
        self.assertEqual(result["outputShapes"]["output-1"], [2, 4, 8])


if __name__ == "__main__":
    unittest.main()
