"""
MLP蓝图测试脚本

测试完整的MLP网络：Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax -> Output
"""

import json
import torch
from engine import BlueprintEngine

# 1. 读取MLP蓝图
with open('test_mlp_blueprint.json', 'r', encoding='utf-8') as f:
    blueprint_data = json.load(f)

print("=" * 60)
print("     MLP 蓝图端到端测试")
print("=" * 60)
print()
print(f"✅ 蓝图加载成功")
print(f"   节点数量: {len(blueprint_data['nodes'])}")
print(f"   连接数量: {len(blueprint_data['edges'])}")

# 2. 初始化引擎
engine = BlueprintEngine(blueprint_data)
print("✅ 引擎初始化成功")

# 3. 准备输入数据（模拟MNIST：批次大小1，特征784）
dummy_input = torch.randn(1, 784)
initial_data = {
    "node-1": {"out": dummy_input}
}

print(f"📥 输入数据形状: {dummy_input.shape}")
print()

# 4. 执行蓝图
print("🔄 开始执行蓝图...")
print("-" * 60)

try:
    results = engine.execute(initial_data)
    print("-" * 60)
    print("✅ 执行成功！")
    print()

    # 5. 打印每个节点的输出形状
    print("📊 各节点输出：")
    for node_id in sorted(results.keys(), key=lambda x: int(x.split('-')[1])):
        res = results[node_id]
        if res is None:
            print(f"   {node_id}: None")
            continue
        for port, val in res.items():
            if hasattr(val, 'shape'):
                print(f"   {node_id} [{port}]: shape = {list(val.shape)}")
            else:
                print(f"   {node_id} [{port}]: value = {val}")

    # 6. 验证最终输出
    final_output = results.get("node-8")
    if final_output and "out" in final_output:
        out_tensor = final_output["out"]
        print()
        print("🎯 最终输出（Softmax概率分布）：")
        print(f"   形状: {list(out_tensor.shape)}")
        print(f"   概率和: {out_tensor.sum().item():.4f} (应该≈1.0)")
        print(f"   最大概率类别: {out_tensor.argmax().item()}")

except Exception as e:
    print(f"❌ 执行失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("     测试完成")
print("=" * 60)
