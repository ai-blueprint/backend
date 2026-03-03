# 贡献指南 (Contributing Guide)

欢迎来到 **炼丹蓝图 (AI Blueprint)** 后端项目！我们非常高兴你愿意为这个项目贡献代码。

为了保持代码库的清晰、易读和易于维护，我们制定了一套独特的开发规范。在提交 Pull Request 之前，请务必仔细阅读并遵守以下指南。

## 🧠 核心开发哲学

本项目遵循 **面向理解编程 (UOP)** 原则，旨在让代码像积木一样直观，让任何具备基础编程知识的人都能看懂。

1.  **代码逻辑必须符合人类直觉**
    *   代码阅读顺序等于执行顺序，不跳跃不倒叙
    *   避免复杂的函数式编程技巧或深层嵌套
    *   减少缩进：如果能用 `if ... return` 提前结束，就不要用 `else` 包裹大段代码

2.  **极致的注释要求**
    *   尽量为每一行有效业务代码末尾添加中文尾随注释
    *   注释解释业务意图，不复述语法（写"跳到下一页"而不是"把 x 加 1"）
    *   反直觉的代码必须注释为什么这么做
    *   简单直观的赋值或调用如果命名已足够清晰可以省略注释

3.  **文档化**
    *   每个函数/方法开头必须注明用法（Usage）和调用示例（Example）
    *   模块文件开头用文档字符串说明文件用途

4.  **Postel's Law（宽容原则）**
    *   **严于律己，宽以待人**：对传入的参数要自动适配（鲁棒性），输出的结果要严格规范
    *   方法设计要灵活，把复杂度留在内部，把方便留给调用者

## 📝 代码风格规范

请严格遵守以下代码风格。如果不符合规范，你的 PR 可能会被要求修改。

### 1. 命名规范

| 类型         | 命名方式        | 示例                                  | 说明         |
| ------------ | --------------- | ------------------------------------- | ------------ |
| **类名**     | PascalCase      | `InputNode`, `BaseNode`               | 首字母大写   |
| **变量名**   | camelCase       | `nodeMap`, `inputValues`              | 首字母小写   |
| **函数名**   | camelCase       | `createNode`, `getAllForFrontend`     | 动词 + 名词  |
| **常量**     | camelCase       | `categoriesOrder`, `hiddenCategories` | 禁止全大写   |
| **布尔值**   | is/has/can 开头 | `isReady`, `hasPermission`            | 禁止否定式   |
| **集合**     | 复数形式        | `nodes`, `edges`, `clients`           | 表示多个元素 |
| **回调函数** | on 开头         | `onMessage`, `onError`                | 事件回调     |

**❌ 禁止的命名：**
- 全大写常量：`MAX_COUNT` ❌ → `maxCount` ✅
- 蛇形命名：`user_name` ❌ → `userName` ✅
- 自定义缩写：`usr`, `mgr`, `cnt`, `evt`, `cb` ❌
- 否定式布尔：`isNotDisabled` ❌ → `isActive` ✅

### 2. 注释规范（最重要！）

**尾随注释格式：**
```python
# 正确示例
nodes = blueprint.get("nodes", [])  # 从蓝图中提取节点列表
edges = blueprint.get("edges", [])  # 从蓝图中提取边列表

sortedIds = sort.topoSort(nodes, edges)  # 调用拓扑排序，得到执行顺序
print(f"拓扑排序结果：{sortedIds}")  # 打印排序结果用于调试

nodeMap = {}  # 创建节点 id 到节点数据的映射字典
for node in nodes:  # 遍历所有节点
    nodeId = node.get("id", "")  # 获取节点 id
    nodeMap[nodeId] = node  # 存入映射字典方便后续查找
```

**函数文档格式：**
```python
async def run(blueprint, onMessage, onError):  # 异步运行蓝图的主函数
    """
    运行蓝图

    用法：
        await run(blueprint, onMessage, onError)

    示例：
        await run(
            {"nodes": [...], "edges": [...]},  # 蓝图数据
            async def(nodeId, result): pass,   # 节点完成回调
            async def(nodeId, error): pass     # 节点错误回调
        )
    """
```

**❌ 错误的注释：**
```python
# 错误：复述语法
x = x + 1  # 把 x 加 1

# 错误：没有注释
nodes = blueprint.get("nodes", [])
for node in nodes:
    nodeId = node.get("id", "")
    nodeMap[nodeId] = node
```

### 3. 结构规范

**扁平化代码结构：**
```python
# ✅ 正确：使用卫语句提前返回
def createNode(opcode, nodeId, params):
    if opcode not in nodes:  # 如果注册表中找不到这个操作码
        raise ValueError(f"未知节点：{opcode}")  # 报错提示
    
    cls = nodes[opcode]["cls"]  # 从字典中取出对应的类
    return cls(nodeId, params)  # 实例化并返回

# ❌ 错误：深层嵌套
def createNode(opcode, nodeId, params):
    if opcode in nodes:
        cls = nodes[opcode]["cls"]
        return cls(nodeId, params)
    else:
        raise ValueError(f"未知节点：{opcode}")
```

**代码分层：**
- **编排层**：负责调度业务步骤，读起来像流程清单（如 `engine.py` 的 `run` 函数）
- **算子层**：负责执行单一具体操作（如各个节点的 `compute` 方法）
- **算子内部禁止调用其它业务算子**

### 4. 视觉规范

**缩进层级：**
- 缩进不超过三层，超过就提前 `return` 或抽取函数
- 相关代码紧挨，不相关代码空一行
- 同级并列逻辑保持视觉对齐


## 🛠️ 如何开发新节点

这是贡献代码最常见的方式。你只需要在 `nodes/` 目录下创建一个新的 `.py` 文件即可。

### 开发步骤

1.  **新建文件**：在 `nodes/` 下新建文件，例如 `nodes/my_model.py`
2.  **导入依赖**：引入必要的 `torch` 和 `registry` 模块
3.  **注册分类**（可选）：如果你需要一个新的分类，使用 `category()`
4.  **定义节点**：
    *   使用 `@node` 装饰器定义元数据（opcode, label, ports, params）
    *   继承 `BaseNode` 类
    *   实现 `build(self)` 方法（可选，用于初始化层）
    *   实现 `compute(self, input)` 方法（必须）

### 节点模板

```python
"""
nodes/my_node.py - 我的自定义节点
"""

import torch  # 导入 torch 用于张量操作
from registry import category, node, BaseNode  # 从 registry 导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册自定义分类
    id="my_category",  # 分类唯一标识
    label="我的分类",  # 分类显示名称
    color="#8B92E5",  # 分类颜色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================

@node(  # 注册节点
    opcode="my_custom_op",  # 节点操作码，唯一标识
    label="我的自定义节点",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入数据"},  # 输入端口定义
        "output": {"y": "输出结果"}  # 输出端口定义
    },
    params={  # 可调参数定义
        "threshold": {"label": "阈值", "type": "float", "value": 0.5, "range": [0, 1]}
    },
    description="这是一个示例节点的描述"  # 节点描述
)
class MyCustomNode(BaseNode):  # 继承 BaseNode
    """
    我的自定义节点类
    
    用法：
        输入 x: shape=[batch, features]
        输出 y: shape=[batch, features]
    """
    
    def build(self):  # 初始化方法（可选）
        # 在这里初始化 PyTorch 层
        pass
    
    def compute(self, input):  # 计算方法（必须实现）
        x = input.get("x")  # 获取输入端口 'x' 的数据
        threshold = self.params["threshold"]["value"]  # 获取参数 'threshold' 的值
        
        if x is None:  # 如果输入为空
            return {"y": None}  # 返回空结果
        
        result = x > threshold  # 简单的逻辑运算
        
        return {"y": result}  # 返回字典，key 必须对应 output 端口
```

### 参数类型参考

| 类型    | 说明   | 示例                                                                                          |
| ------- | ------ | --------------------------------------------------------------------------------------------- |
| `int`   | 整数   | `{"type": "int", "value": 256, "range": [0, 1024]}`                                           |
| `float` | 浮点数 | `{"type": "float", "value": 0.5, "range": [0, 1]}`                                            |
| `bool`  | 布尔值 | `{"type": "bool", "value": True}`                                                             |
| `str`   | 字符串 | `{"type": "str", "value": "默认字符串"}`                                                      |
| `list`  | 列表   | `{"type": "list", "value": [1, 2, 3]}`                                                        |
| `enum`  | 选项   | `{"type": "enum", "value": "option1", "options": {"option1": "选项 1", "option2": "选项 2"}}` |

## 📮 提交 Pull Request

1.  **Fork** 本仓库
2.  创建一个新的分支 (`git checkout -b feat/new-node-xxx`)
3.  提交你的更改，确保遵守了上述的代码风格
4.  推送到你的分支 (`git push origin feat/new-node-xxx`)
5.  创建一个 Pull Request，描述你添加的功能或修复的问题

### 提交前检查清单

- [ ] 每一行有效代码都有中文尾随注释
- [ ] 函数有文档字符串（用法和示例）
- [ ] 命名符合驼峰规范
- [ ] 代码扁平化，缩进不超过三层
- [ ] 新节点文件放在 `nodes/` 目录下
- [ ] 使用 `@node` 和 `@category` 装饰器注册

---

再次感谢你的贡献！让我们一起把炼丹蓝图变得更好！
