import json
import torch
from engine import BlueprintEngine

# 1. 读取蓝图
with open('test_blueprint.json', 'r', encoding='utf-8') as f:
    blueprint_data = json.load(f)

print("✅ 蓝图加载成功")
print(f"节点数量: {len(blueprint_data['nodes'])}")
print(f"连接数量: {len(blueprint_data['edges'])}")

# 2. 初始化引擎
engine = BlueprintEngine(blueprint_data)
print("✅ 引擎初始化成功")

# 3. 随机输入数据
# 假设 input_1 的输出维度是 [1, 10]
dummy_input = torch.randn(1, 10)
initial_data = {
    "input_1": {"out": dummy_input}
}

print(f"📥 输入数据形状: {dummy_input.shape}")

# 4. 执行
try:
    results = engine.execute(initial_data)
    print("✅ 执行成功！")

    # 5. 验证结果
    for node_id, res in results.items():
        if res is None:
            print(f"Node {node_id}: Output is None")
            continue
        for port, val in res.items():
            if hasattr(val, 'shape'):
                print(f"Node {node_id} ({port}): Shape = {val.shape}")
            else:
                print(f"Node {node_id} ({port}): Value = {val}")

except Exception as e:
    print(f"❌ 执行失败: {e}")
    import traceback
    traceback.print_exc()
