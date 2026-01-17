"""
快速测试脚本：验证重构后的代码是否正常工作
"""

import torch
from engine import BlueprintEngine
from registry import Registry

def test_registry():
    """测试注册表加载"""
    print("=" * 50)
    print("测试1: 注册表加载")
    print("=" * 50)
    
    registry = Registry()
    registry.load_nodes()
    
    print(f"✅ 成功加载 {len(registry)} 个节点")
    print(f"   分类: {registry.list_categories()}")
    print(f"   节点数: {len(registry.list_nodes())}")
    print()


def test_simple_blueprint():
    """测试简单蓝图执行"""
    print("=" * 50)
    print("测试2: 简单蓝图执行 (input -> linear -> relu -> output)")
    print("=" * 50)
    
    blueprint = {
        'nodes': [
            {'id': 'n1', 'data': {'nodeKey': 'input', 'params': {'输出维度': {'default': [1, 10]}}}},
            {'id': 'n2', 'data': {'nodeKey': 'linear', 'params': {'in_features': {'default': 10}, 'out_features': {'default': 5}}}},
            {'id': 'n3', 'data': {'nodeKey': 'relu', 'params': {}}},
            {'id': 'n4', 'data': {'nodeKey': 'output', 'params': {}}}
        ],
        'edges': [
            {'source': 'n1', 'target': 'n2', 'sourceHandle': 'out', 'targetHandle': 'x'},
            {'source': 'n2', 'target': 'n3', 'sourceHandle': 'out', 'targetHandle': 'x'},
            {'source': 'n3', 'target': 'n4', 'sourceHandle': 'result', 'targetHandle': 'in'}
        ]
    }
    
    initial_inputs = {
        'n1': {'out': torch.randn(1, 10)}
    }
    
    engine = BlueprintEngine(blueprint)
    results = engine.execute(initial_inputs)
    
    print(f"✅ 执行成功！")
    print(f"   输入形状: {initial_inputs['n1']['out'].shape}")
    output_data = results['n4']
    if output_data:
        out_key = list(output_data.keys())[0] if output_data else None
        if out_key:
            print(f"   输出形状: {output_data[out_key].shape}")
    print()


def test_math_nodes():
    """测试数学运算节点"""
    print("=" * 50)
    print("测试3: 数学运算节点 (x + y)")
    print("=" * 50)
    
    blueprint = {
        'nodes': [
            {'id': 'x', 'data': {'nodeKey': 'input', 'params': {}}},
            {'id': 'y', 'data': {'nodeKey': 'input', 'params': {}}},
            {'id': 'add', 'data': {'nodeKey': 'add', 'params': {}}},
            {'id': 'out', 'data': {'nodeKey': 'output', 'params': {}}}
        ],
        'edges': [
            {'source': 'x', 'target': 'add', 'sourceHandle': 'out', 'targetHandle': 'x'},
            {'source': 'y', 'target': 'add', 'sourceHandle': 'out', 'targetHandle': 'y'},
            {'source': 'add', 'target': 'out', 'sourceHandle': 'result', 'targetHandle': 'in'}
        ]
    }
    
    x_val = torch.tensor([1.0, 2.0, 3.0])
    y_val = torch.tensor([4.0, 5.0, 6.0])
    
    initial_inputs = {
        'x': {'out': x_val},
        'y': {'out': y_val}
    }
    
    engine = BlueprintEngine(blueprint)
    results = engine.execute(initial_inputs)
    
    # output节点的输出端口是'out'
    output = results['out']['out']
    expected = torch.tensor([5.0, 7.0, 9.0])
    
    if torch.allclose(output, expected):
        print(f"✅ 加法计算正确！")
        print(f"   x = {x_val.tolist()}")
        print(f"   y = {y_val.tolist()}")
        print(f"   x + y = {output.tolist()}")
    else:
        print(f"❌ 加法计算错误！")
        print(f"   期望: {expected.tolist()}")
        print(f"   实际: {output.tolist()}")
    print()


def test_utils():
    """测试工具函数"""
    print("=" * 50)
    print("测试4: 工具函数")
    print("=" * 50)
    
    from utils import (
        extract_single_input,
        ensure_tensor,
        serialize_tensor,
        deserialize_tensor,
        safe_get,
        coerce_type
    )
    
    # 测试 extract_single_input
    inputs = {"x": torch.tensor([1, 2, 3])}
    x = extract_single_input(inputs, "x")
    print(f"✅ extract_single_input: {x.tolist()}")
    
    # 测试 ensure_tensor
    t = ensure_tensor([1, 2, 3], torch.float32)
    print(f"✅ ensure_tensor: {t.tolist()}, dtype={t.dtype}")
    
    # 测试序列化
    tensor = torch.tensor([1.0, 2.0, 3.0])
    serialized = serialize_tensor(tensor)
    restored = deserialize_tensor(serialized)
    print(f"✅ serialize/deserialize: {restored.tolist()}")
    
    # 测试 safe_get
    data = {"a": {"b": {"c": 42}}}
    value = safe_get(data, "a", "b", "c", default=0)
    print(f"✅ safe_get: {value}")
    
    # 测试 coerce_type
    num = coerce_type("42", "number")
    print(f"✅ coerce_type: '42' -> {num} ({type(num).__name__})")
    
    print()


def main():
    """运行所有测试"""
    print("\n🔧 开始测试重构后的代码...\n")
    
    try:
        test_registry()
        test_simple_blueprint()
        test_math_nodes()
        test_utils()
        
        print("=" * 50)
        print("🎉 所有测试通过！")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
