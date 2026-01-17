# 炼丹蓝图 - 后端服务

基于 PyTorch 的可视化深度学习模型构建平台后端服务。

## 项目结构

```
backend/
├── main.py              # 主程序入口
├── engine.py            # 蓝图执行引擎
├── ws_server.py         # WebSocket 服务器
├── registry.py          # 节点注册表
├── loader.py            # 节点动态加载器
├── decorators.py        # 节点注册装饰器
├── utils/               # 工具函数包
│   ├── __init__.py      # 统一导出
│   ├── tensor.py        # 张量操作工具
│   ├── serialization.py # 序列化工具
│   ├── validation.py    # 参数验证工具
│   ├── safe.py          # 容错处理工具
│   └── graph.py         # 图算法工具
├── nodes/               # 节点定义
│   ├── __init__.py      # 节点工厂函数
│   ├── base.py          # 基础节点（输入/输出）
│   ├── activations.py   # 激活函数节点
│   ├── layers.py        # 神经网络层节点
│   ├── losses.py        # 损失函数节点
│   └── math.py          # 数学运算节点
└── tests/               # 测试文件
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 启动服务器

```bash
uv run python main.py
# 或指定端口
uv run python main.py --host 0.0.0.0 --port 8765
```

### 3. 生成节点注册表

```bash
uv run python registry.py
```

## 核心模块说明

### 引擎 (engine.py)

蓝图执行引擎，负责：
- 解析蓝图结构
- 拓扑排序确定执行顺序
- 按顺序执行节点计算
- 管理层实例缓存

```python
from engine import BlueprintEngine

engine = BlueprintEngine(blueprint_data)
results = engine.execute(initial_inputs)
```

### 工具函数 (utils/)

提供多种通用工具：

```python
from utils import (
    # 张量工具
    extract_single_input,
    ensure_tensor,
    get_shape,
    
    # 序列化
    serialize_tensor,
    deserialize_tensor,
    
    # 验证
    validate_params,
    coerce_type,
    
    # 容错
    safe_call,
    safe_get,
    
    # 图算法
    topological_sort,
)
```

### 节点工厂 (nodes/__init__.py)

提供便捷的节点创建函数：

```python
from nodes import (
    create_activation_node,     # 激活函数节点
    create_module_node,         # nn.Module 节点
    create_binary_op_node,      # 二元运算节点
    create_loss_node,           # 损失函数节点
    create_passthrough_node,    # 透传节点
)
```

## 创建新节点

### 简单激活函数

```python
from decorators import category, node
from nodes import create_activation_node
import torch.nn.functional as F

@node(
    opcode="my_activation",
    name="我的激活函数",
    ports={"in": ["x"], "out": ["result"]},
    params={}
)
def my_activation_node():
    return create_activation_node(F.relu)
```

### 带参数的层

```python
from nodes import create_module_node
import torch.nn as nn

@node(
    opcode="my_layer",
    name="我的层",
    ports={"in": ["x"], "out": ["out"]},
    params={"hidden_size": 64}
)
def my_layer_node():
    return create_module_node(
        nn.Linear,
        build_args=lambda p: (128, p["hidden_size"]),
    )
```

## WebSocket 协议

### 请求格式

```json
// 获取节点注册表
{"type": "get_registry", "id": "req-1"}

// 运行蓝图
{"type": "run_blueprint", "id": "req-2", "data": {"blueprint": {...}, "inputs": {...}}}
```

### 响应格式

```json
// 注册表
{"type": "registry", "id": "req-1", "data": {...}}

// 节点执行结果
{"type": "node_result", "id": "req-2", "data": {"nodeId": "n1", "output": {...}}}

// 执行完成
{"type": "execution_complete", "id": "req-2", "data": {"success": true, "results": {...}}}

// 错误
{"type": "error", "id": "req-2", "data": {"message": "错误信息"}}
```

## 设计原则

1. **语义化函数分离** - 每个函数只做一件事
2. **容错处理** - 异常不会中断整个执行流程
3. **松散耦合** - 模块之间通过接口通信
4. **高可扩展性** - 新节点只需添加装饰器即可注册

## 测试

```bash
# 运行测试
uv run python test_refactor.py

# 运行 MLP 测试
uv run python test_mlp.py
```

## 许可证

MIT License
