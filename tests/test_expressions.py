import asyncio  # 异步运行能力，用于验证变量蓝图端到端执行
import unittest  # 单元测试能力，用于锁定变量表达式契约

import expressions  # 变量表达式解析能力，是本组测试主体
import engine  # 蓝图执行能力，用于验证变量在编译链路中消解
from blueprints import linearBlueprint  # 共享最小蓝图，减少重复图噪声


class ExpressionParsingTest(unittest.TestCase):
    # --- 验证数字与变量的基础求值 ---
    def test_resolves_numbers_variables_and_arithmetic(self):
        variables = {"B": 2, "S": 4, "D": 8, "HEADS": 2}  # 模拟蓝图变量表

        self.assertEqual(expressions.resolveText("16", variables), 16)
        self.assertEqual(expressions.resolveText("D", variables), 8)
        self.assertEqual(expressions.resolveText("D*4", variables), 32)
        self.assertEqual(expressions.resolveText("D//HEADS", variables), 4)
        self.assertEqual(expressions.resolveText("(D+8)*2", variables), 32)
        self.assertEqual(expressions.resolveText("-D+10", variables), 2)

    # --- 验证逗号序列和方括号形状写法 ---
    def test_resolves_shape_sequences(self):
        variables = {"B": 2, "S": 4, "D": 8}  # 形状三变量契约

        self.assertEqual(expressions.resolveText("B, S, D", variables), [2, 4, 8])
        self.assertEqual(expressions.resolveText("[B, S, D*2]", variables), [2, 4, 16])
        self.assertEqual(expressions.resolveText("B, 4, D//2", variables), [2, 4, 4])

    # --- 验证列表变量整体引用与展开 ---
    def test_list_variable_expands_into_sequence(self):
        variables = {"SHAPE": [2, 4, 8], "D": 8}  # 列表变量表达完整形状

        self.assertEqual(expressions.resolveText("SHAPE", variables), [2, 4, 8])
        self.assertEqual(expressions.resolveText("SHAPE, D", variables), [2, 4, 8, 8])

    # --- 验证整除结果收敛为整数供层宽使用 ---
    def test_division_result_becomes_int_when_whole(self):
        value = expressions.resolveText("D/2", {"D": 8})  # 真除法得到4.0

        self.assertEqual(value, 4)
        self.assertIsInstance(value, int)

    # --- 验证非法输入都被明确拒绝 ---
    def test_rejects_invalid_expressions(self):
        variables = {"D": 8}  # 单变量即可覆盖所有拒绝路径

        with self.assertRaisesRegex(ValueError, "未定义的变量"):
            expressions.resolveText("MISSING", variables)
        with self.assertRaisesRegex(ValueError, "无法识别的字符"):
            expressions.resolveText("D@2", variables)
        with self.assertRaisesRegex(ValueError, "除以零"):
            expressions.resolveText("D/0", variables)
        with self.assertRaisesRegex(ValueError, "缺少右括号"):
            expressions.resolveText("(D+1", variables)
        with self.assertRaisesRegex(ValueError, "不能为空"):
            expressions.resolveText("  ", variables)
        with self.assertRaisesRegex(ValueError, "列表变量不能参与"):
            expressions.resolveText("SHAPE*2", {"SHAPE": [2, 4]})
        with self.assertRaisesRegex(ValueError, "多余内容"):
            expressions.resolveText("D 2", variables)

    # --- 验证代码注入类文本无法通过词法层 ---
    def test_rejects_code_like_text(self):
        with self.assertRaises(ValueError):
            expressions.resolveText("__import__('os')", {})  # 引号和点号都不是合法字符
        with self.assertRaises(ValueError):
            expressions.resolveText("D.shape", {"D": 8})  # 属性访问被词法层拒绝


class VariablesMapTest(unittest.TestCase):
    # --- 验证合法变量列表转换 ---
    def test_builds_map_from_valid_list(self):
        variables = expressions.getVariablesMap([
            {"id": "v1", "name": "B", "value": 2},
            {"id": "v2", "name": "D", "value": 8.0},
            {"id": "v3", "name": "SHAPE", "value": [2, 4.0, 8]},
        ])  # 覆盖整数、整值浮点和列表三种值

        self.assertEqual(variables, {"B": 2, "D": 8, "SHAPE": [2, 4, 8]})
        self.assertIsInstance(variables["D"], int)

    # --- 验证非法名称和值都被拒绝 ---
    def test_rejects_invalid_names_and_values(self):
        with self.assertRaisesRegex(ValueError, "变量名无效"):
            expressions.getVariablesMap([{"name": "1D", "value": 2}])
        with self.assertRaisesRegex(ValueError, "变量名重复"):
            expressions.getVariablesMap([{"name": "D", "value": 2}, {"name": "D", "value": 3}])
        with self.assertRaisesRegex(ValueError, "必须是数字"):
            expressions.getVariablesMap([{"name": "FLAG", "value": True}])
        with self.assertRaisesRegex(ValueError, "全部是数字"):
            expressions.getVariablesMap([{"name": "BAD", "value": [2, "x"]}])


class ResolveNodeParamsTest(unittest.TestCase):
    # --- 验证参数对象中的表达式被消解为具体值 ---
    def test_resolves_expression_objects_inside_params(self):
        params = {
            "out_shape": {"label": "输出形状", "type": "list", "value": {"expr": "B, S, D"}},
            "in_features": {"label": "输入特征", "type": "int", "value": {"expr": "D"}},
            "bias": {"label": "偏置", "type": "bool", "value": True},
            "mixed": {"label": "混合列表", "type": "list", "value": [1, {"expr": "D*2"}]},
        }  # 覆盖表达式、普通值和混合列表

        resolved = expressions.resolveNodeParams(params, {"B": 2, "S": 4, "D": 8})

        self.assertEqual(resolved["out_shape"], [2, 4, 8])
        self.assertEqual(resolved["in_features"], 8)
        self.assertEqual(resolved["bias"], True)
        self.assertEqual(resolved["mixed"], [1, 16])


class VariableBlueprintExecutionTest(unittest.TestCase):
    # --- 构造使用变量的完整蓝图 ---
    def createVariableBlueprint(self):
        blueprint = linearBlueprint()  # 复用输入-线性-输出结构
        blueprint["variables"] = [
            {"id": "var-b", "name": "B", "value": 2},
            {"id": "var-s", "name": "S", "value": 4},
            {"id": "var-d", "name": "D", "value": 8},
        ]  # 蓝图级变量表
        blueprint["nodes"][0]["data"]["params"]["out_shape"] = {"expr": "B, S, D"}  # 输入形状引用变量
        blueprint["nodes"][1]["data"]["params"]["in_features"] = {"expr": "D"}  # 线性层输入宽度跟随D
        blueprint["nodes"][1]["data"]["params"]["out_features"] = {"expr": "D"}  # 线性层输出宽度跟随D，保持Block同形
        return blueprint  # 返回可直接编译的变量蓝图

    # --- 验证变量蓝图编译并保持形状 ---
    def test_variable_blueprint_compiles_and_keeps_shape(self):
        model = engine.compileBlueprint(self.createVariableBlueprint())  # 编译期完成全部变量消解

        outputs = model()  # 随机输入执行一轮

        self.assertEqual(list(outputs["output-1"].shape), [2, 4, 8])  # 输出形状与输入变量形状一致

    # --- 验证修改变量值即改变整图形状 ---
    def test_changing_variable_value_changes_all_references(self):
        blueprint = self.createVariableBlueprint()
        blueprint["variables"][2]["value"] = 16  # 只改D一个变量

        model = engine.compileBlueprint(blueprint)
        outputs = model()

        self.assertEqual(list(outputs["output-1"].shape), [2, 4, 16])  # 输入形状与线性层宽度同时跟随

    # --- 验证未定义变量在编译期结构化失败 ---
    def test_undefined_variable_fails_with_structured_error(self):
        blueprint = self.createVariableBlueprint()
        blueprint["nodes"][1]["data"]["params"]["in_features"] = {"expr": "MISSING"}  # 引用不存在的变量

        with self.assertRaises(engine.BlueprintError) as context:
            engine.compileBlueprint(blueprint)

        self.assertEqual(context.exception.code, "invalidExpression")
        self.assertEqual(context.exception.details.get("nodeId"), "linear-1")

    # --- 验证非法变量定义在编译期整体失败 ---
    def test_invalid_variable_definition_fails_before_node_creation(self):
        blueprint = self.createVariableBlueprint()
        blueprint["variables"].append({"id": "var-bad", "name": "9X", "value": 1})  # 非法变量名

        with self.assertRaises(engine.BlueprintError) as context:
            engine.compileBlueprint(blueprint)

        self.assertEqual(context.exception.code, "invalidVariable")

    # --- 验证run链路也能消费变量蓝图 ---
    def test_run_executes_variable_blueprint(self):
        reportedIDs = []  # 收集逐节点反馈

        async def onMessage(nodeID, _report):
            reportedIDs.append(nodeID)

        async def onError(_nodeID, error):
            self.fail(error)  # 合法变量蓝图不应进入错误路径

        result = asyncio.run(engine.run(self.createVariableBlueprint(), onMessage, onError))

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(reportedIDs, ["input.with.dot", "linear-1", "output-1"])


if __name__ == "__main__":
    unittest.main()
