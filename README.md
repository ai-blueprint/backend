# 炼丹蓝图后端

后端负责读取蓝图、创建 PyTorch 节点、执行 Tensor 数据流，并通过 WebSocket 把结果返回给前端。

## 开始使用

需要安装：

- Python `3.12+`
- [uv](https://docs.astral.sh/uv/)

安装依赖：

```sh
uv sync
```

启动 WebSocket 服务：

```sh
uv run python main.py
```

服务默认地址：

```text
ws://127.0.0.1:8765
```

## 后端职责

```text
前端蓝图
→ WebSocket 请求
→ 蓝图校验
→ 节点注册表
→ PyTorch 执行
→ Tensor 和错误反馈
```

当前支持：

- 动态节点注册
- 拓扑顺序执行
- Tensor 逐节点反馈
- 单节点错误反馈
- 变量表达式
- 普通前向传播
- 单步训练和连续训练
- Loss、权重和梯度反馈
- SGD、Adam 和梯度裁剪

## 节点开发

节点放在 `nodes/` 目录中。普通节点使用统一定义：

```python
@node(
    opcode="example",
    label="示例",
    ports={"input": {"x": "输入"}, "output": {"out": "输出"}},
    params={},
    description="说明这个节点做什么",
)
class ExampleNode(BaseNode):
    def compute(self, input):
        return {"out": input.get("x")}
```

节点使用 PyTorch Tensor 作为数据。节点输出必须是端口字典，例如：

```python
return {"out": result}
```

不要在节点里加入具体模型的完整结构。RWKV、Transformer 等结构应由普通节点连接出来。

## 注册表同步

后端注册表是节点定义的来源。修改节点后运行：

```sh
uv run python export_registry.py
```

它会更新前端离线注册表：

```text
../frontend/src/constants/registry.json
```

## WebSocket 协议

请求格式：

```json
{"type":"runBlueprint","id":"request-id","data":{}}
```

当前主要请求：

| 请求 | 用途 |
|---|---|
| `getRegistry` | 获取节点注册表 |
| `runBlueprint` | 执行一次蓝图前向传播 |
| `trainStep` | 执行一次训练、反向传播和参数更新 |
| `resetTraining` | 清除训练模型和优化器状态 |

Tensor 结果包含形状、类型、设备、数值和是否截断等信息。

## 测试

```sh
uv run python -m unittest discover -s tests -p "test_*.py"
```

后端测试覆盖节点默认执行、注册表同步、蓝图执行、训练、变量表达式和 WebSocket 协议。

## 许可证

本项目采用 GNU AGPL v3 许可证。
