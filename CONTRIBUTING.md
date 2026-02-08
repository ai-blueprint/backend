# 贡献指南 (Contributing Guide)

欢迎来到 **炼丹蓝图 (AI Blueprint)** 后端项目！我们非常高兴你愿意为这个项目贡献代码。

为了保持代码库的清晰、易读和易于维护，我们制定了一套独特的开发规范。在提交 Pull Request 之前，请务必仔细阅读并遵守以下指南。

## 🧠 核心开发哲学

本项目遵循以下核心原则，旨在让代码像积木一样直观，让任何具备基础编程知识的人都能看懂。

1.  **面向理解编程 (Programming for Understanding)**
    *   代码逻辑必须符合人类直觉。
    *   **极致的注释要求**：每一行代码（除了极其显而易见的）都必须包含“大白话”尾随注释，解释这一行在做什么。
    *   **文档化**：每个函数/方法开头必须注明具体用法（Usage）和调用示例（Example）。

2.  **命令式与积木化 (Imperative & Block-like)**
    *   避免复杂的函数式编程技巧或深层嵌套。
    *   代码应该像搭积木一样，一行接一行，逻辑线性向下流动。
    *   **减少缩进**：如果能用 `if ... return` 提前结束，就不要用 `else` 包裹大段代码。

3.  **Postel’s Law**
    *   **严于律己，宽以待人**：对传入的参数要自动适配（鲁棒性），输出的结果要严格规范。
    *   方法设计要灵活，把复杂度留在内部，把方便留给调用者。

## 📝 代码风格规范

请严格遵守以下代码风格。如果不符合规范，你的 PR 可能会被要求修改。

### 1. 命名规范
*   **驼峰命名法 (CamelCase)**：变量名、函数名全部使用驼峰命名（例如 `sendMessage`, `nodeMap`）。
*   **类名**：使用帕斯卡命名法（例如 `BaseNode`, `InputNode`）。
*   **语义化**：名字要简洁且符合语境，拒绝不明觉厉的缩写。

### 2. 注释规范 (最重要！)
*   **尾随注释**：代码行的末尾必须加上注释，用大白话解释这一行代码的意图。
*   **函数文档**：尽量包含 `用法` 和 `示例`，除非真的显而易见。

**✅ 正确示例：**
```python
def createNode(opcode, nodeId, params):
    """
    根据opcode创建节点实例

    用法：
        instance = createNode("math_add", "node_1", {"val": 1})
    """
    if opcode not in nodes:  # 如果注册表中找不到这个操作码
        raise ValueError(f"未知节点: {opcode}")  # 报错提示

    cls = nodes[opcode]["cls"]  # 从字典中取出对应的类
    return cls(nodeId, params)  # 实例化并返回
```

**❌ 错误示例：**
```python
# 创建节点
def create_node(opcode, id, p):
    if opcode in nodes:
        return nodes[opcode]['cls'](id, p)
    else:
        raise ValueError
```

### 3. 结构规范
*   **扁平化**：尽量减少 `if/for` 的嵌套层级。

## 🛠️ 如何开发新节点

这是贡献代码最常见的方式。你只需要在 `nodes/` 目录下创建一个新的 `.py` 文件即可。

### 开发步骤

1.  **新建文件**：在 `nodes/` 下新建文件，例如 `nodes/my_model.py`。
2.  **导入依赖**：引入必要的 `torch` 和 `registry` 模块。
3.  **注册分类**（可选）：如果你需要一个新的分类，使用 `@category`。
4.  **定义节点**：
    *   使用 `@node` 装饰器定义元数据（opcode, label, ports, params）。
    *   继承 `BaseNode` 类。
    *   实现 `compute(self, input)` 方法。

### 节点模板

```python
import torch
from registry import node, BaseNode

# 注册节点
@node(
    opcode="my_custom_op",          # 唯一标识符，前端通过它识别
    label="我的自定义节点",          # 前端显示的名称
    ports={
        "input": {"x": "输入数据"},  # 输入端口定义
        "output": {"y": "输出结果"}  # 输出端口定义
    },
    params={                        # 可调参数定义
        "threshold": {"label": "阈值", "type": "float", "value": 0.5}
    },
    description="这是一个示例节点的描述"
)
class MyCustomNode(BaseNode):
    def compute(self, input):
        """
        执行计算逻辑
        """
        x = input.get("x")  # 获取输入端口 'x' 的数据
        threshold = self.params["threshold"]["value"]  # 获取参数 'threshold' 的值
        
        # 执行核心逻辑 (这里必须兼容各种输入情况)
        if x is None:  # 如果输入为空
            return {"y": None}  # 返回空结果
            
        result = x > threshold  # 简单的逻辑运算
        
        return {"y": result}  # 返回字典，key 必须对应 output 端口
```

## 📮 提交 Pull Request

1.  **Fork** 本仓库。
2.  创建一个新的分支 (`git checkout -b feat/new-node-xxx`)。
3.  提交你的更改，确保遵守了上述的代码风格。
4.  推送到你的分支 (`git push origin feat/new-node-xxx`)。
5.  创建一个 Pull Request，描述你添加的功能或修复的问题。

---

再次感谢你的贡献！让我们一起把炼丹蓝图变得更好！
