# 炼丹蓝图 - 后端项目完整开发文档

## 📋 项目概述

这是一个基于 **可视化节点编程** 的深度学习训练框架后端系统。前端通过拖拽节点构建训练蓝图，后端负责解析并执行这些蓝图。

### 核心特性

- **WebSocket 实时通信**：前后端通过 WebSocket 保持长连接，实时传输节点执行状态
- **动态节点注册**：使用装饰器模式自动注册节点，无需手动维护节点列表
- **拓扑排序执行**：自动分析节点依赖关系，按正确顺序执行
- **PyTorch 集成**：节点基于 `nn.Module`，可直接使用 PyTorch 的所有功能
- **命令式编程风格**：代码像积木一样清晰，每行都有详细注释

### 技术栈

- **Python 3.12+**：主要开发语言
- **PyTorch 2.9+**：深度学习框架
- **WebSockets 16.0+**：实时通信
- **uv**：现代化的 Python 包管理工具

## 🏗️ 项目架构

```
backend/
├── main.py              # 入口文件，启动服务
├── server.py            # WebSocket 服务器，处理前后端通信
├── registry.py          # 节点注册表，管理所有节点定义
├── loader.py            # 动态加载器，自动导入节点模块
├── engine.py            # 蓝图执行引擎，核心执行逻辑
├── sort.py              # 拓扑排序算法，确定执行顺序
├── nodes/               # 节点定义目录
│   ├── __init__.py      # 包初始化文件
│   ├── base.py          # 基础节点（输入/输出/调试）
│   └── example.py       # 示例节点（展示如何定义节点）
├── pyproject.toml       # 项目配置和依赖
└── README.md            # 本文档
```

### 数据流向

```
前端拖拽节点 
    ↓
通过 WebSocket 发送蓝图数据
    ↓
server.py 接收消息
    ↓
engine.py 解析蓝图
    ↓
sort.py 拓扑排序
    ↓
按顺序执行每个节点
    ↓
实时回传执行结果
    ↓
前端显示执行状态
```

### 核心概念

#### 1. 节点（Node）
- 每个节点是一个独立的计算单元
- 继承自 `BaseNode`（实际上是 `nn.Module`）
- 必须实现 `compute(input)` 方法
- 可选实现 `build()` 方法用于初始化层

#### 2. 蓝图（Blueprint）
- 由前端构建的节点图
- 包含 `nodes` 数组和 `edges` 数组
- `nodes`：节点列表，每个节点有 id、data（包含 opcode 和 params）
- `edges`：连接关系，定义数据流向

#### 3. 端口（Port）
- 输入端口（input）：接收上游节点的数据
- 输出端口（output）：向下游节点传递数据
- 端口通过字典的键来标识，如 `{"in": value}`

#### 4. 参数（Params）
- 节点的配置参数
- 支持类型：int、float、bool、str、list、enum
- 在前端可视化配置，传递给后端执行

---

## 🚀 环境配置和启动

### 前置要求

- **Python 3.12.12+**
- **uv 包管理器**（推荐）或 pip

### 安装步骤

#### 方法一：使用 uv（推荐）

```bash
# 1. 安装 uv（如果还没安装）
# Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 进入项目目录
cd backend

# 3. uv 会自动创建虚拟环境并安装依赖
uv sync
```

#### 方法二：使用 pip

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. 安装依赖
pip install torch torchvision torchaudio websockets
```

### 启动服务

```bash
# 使用 uv 启动（推荐）
uv run python main.py

# 或者激活虚拟环境后直接运行
python main.py
```

启动成功后会看到：
```
已加载节点模块: nodes/base.py
已加载节点模块: nodes/example.py
WebSocket服务启动中... ws://localhost:8765
WebSocket服务已启动: ws://localhost:8765
```

### 配置说明

#### 修改服务器地址和端口

编辑 [`main.py`](main.py:7)：

```python
# 默认配置
server.start()  # localhost:8765

# 自定义配置
server.start("0.0.0.0", 9000)  # 监听所有网卡，端口9000
```

#### 依赖说明

[`pyproject.toml`](pyproject.toml:1) 中定义了所有依赖：

```toml
[project]
requires-python = ">=3.12.12"
dependencies = [
    "torch>=2.9.1",        # PyTorch 核心库
    "torchaudio>=2.9.1",   # 音频处理（预留）
    "torchvision>=0.24.1", # 视觉处理（预留）
    "websockets>=16.0",    # WebSocket 通信
]

[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"  # 使用清华镜像加速
```

---

## 📚 核心模块详解

### 1. server.py - WebSocket 服务器

**职责**：处理前后端的实时通信，接收前端请求并返回执行结果。

#### 全局变量

```python
clients = set()  # 已连接的前端客户端集合，用set存储方便增删
```

#### 核心函数

##### `sendMessage(ws, type, id, data)`

发送消息给前端的统一接口。

```python
async def sendMessage(ws, type, id, data):
    """
    参数说明：
        ws: WebSocket连接对象
        type: 消息类型，如 "getRegistry"、"nodeResult"
        id: 消息ID，用于前端匹配请求和响应
        data: 消息数据，具体内容根据type不同而不同
    
    调用示例：
        await sendMessage(websocket, "getRegistry", "req1", nodesData)
        await sendMessage(websocket, "nodeResult", "req2", {"nodeId": "n1", "result": {...}})
    """
    msg = {}  # 创建空字典准备装消息
    msg["type"] = type  # 消息类型
    msg["id"] = id  # 消息ID
    msg["data"] = data  # 消息数据
    print(f"发送给前端消息: {type} {data}")  # 打印日志
    text = json.dumps(msg)  # 把字典转成JSON字符串
    await ws.send(text)  # 通过WebSocket发送给前端
```

**消息格式**：
```json
{
  "type": "消息类型",
  "id": "消息ID",
  "data": { /* 具体数据 */ }
}
```

##### `sendError(ws, type, id, error)`

发送错误消息给前端。

```python
async def sendError(ws, type, id, error):
    """
    参数说明：
        error: 错误信息字符串或字典
    
    调用示例：
        await sendError(ws, "runBlueprint", "msg123", "节点执行失败")
    """
    msg = {}  # 创建空字典准备装错误消息
    msg["type"] = type  # 消息类型
    msg["id"] = id  # 消息ID
    msg["error"] = error  # 错误信息
    text = json.dumps(msg)  # 把字典转成JSON字符串
    await ws.send(text)  # 通过WebSocket发送给前端
```

**错误消息格式**：
```json
{
  "type": "消息类型",
  "id": "消息ID",
  "error": "错误信息"
}
```

##### `handleMessage(ws, message)`

处理前端发来的消息，这是消息路由的核心。

```python
async def handleMessage(ws, message):
    """
    支持的消息类型：
        1. getRegistry - 获取节点注册表
        2. runBlueprint - 运行蓝图
    """
    data = json.loads(message)  # 把JSON字符串解析成字典
    msg_type = data.get("type", "")  # 提取消息类型，默认空字符串
    id = data.get("id", "")  # 提取消息ID，默认空字符串

    if msg_type == "getRegistry":  # 如果是请求节点注册表
        result = registry.getAllForFrontend()  # 调用registry获取前端格式的节点数据
        await sendMessage(ws, msg_type, id, result)  # 发送响应给前端
        return  # 处理完毕，返回

    elif msg_type == "runBlueprint":  # 如果是请求运行蓝图
        blueprint = data["data"].get("blueprint")  # 提取蓝图数据
        print(f"收到运行蓝图请求: {blueprint}")  # 打印日志

        async def onMessage(nodeId, result):  # 定义节点执行完成的回调
            await sendMessage(ws, "nodeResult", id, {"nodeId": nodeId, "result": result})

        async def onError(nodeId, error):  # 定义节点执行出错的回调
            await sendError(ws, "nodeError", id, {"nodeId": nodeId, "error": error})

        await engine.run(blueprint, onMessage, onError)  # 调用引擎运行蓝图
        await sendMessage(ws, "blueprintComplete", id, {})  # 发送蓝图执行完成消息
        return  # 处理完毕，返回

    else:  # 如果是未知消息类型
        await sendError(ws, "unknown", id, f"未知消息类型：{msg_type}")
        return
```

**前端请求示例**：

1. 获取节点注册表：
```json
{
  "type": "getRegistry",
  "id": "req_001"
}
```

2. 运行蓝图：
```json
{
  "type": "runBlueprint",
  "id": "req_002",
  "data": {
    "blueprint": {
      "nodes": [...],
      "edges": [...]
    }
  }
}
```

##### `handleConnection(ws)`

处理 WebSocket 连接的生命周期。

```python
async def handleConnection(ws):
    """
    连接生命周期：
        1. 连接建立 -> 加入clients集合
        2. 循环接收消息 -> 调用handleMessage处理
        3. 连接断开 -> 从clients移除
    """
    clients.add(ws)  # 将新连接的前端加入clients集合
    print(f"前端已连接，当前连接数: {len(clients)}")  # 打印连接信息

    try:  # 尝试接收消息
        async for message in ws:  # 循环接收前端发来的消息
            await handleMessage(ws, message)  # 调用handleMessage处理每条消息
    except websockets.exceptions.ConnectionClosed:  # 如果连接断开
        pass  # 忽略断开异常，正常退出循环
    finally:  # 无论如何都要执行的清理
        clients.discard(ws)  # 从clients集合中移除这个连接
        print(f"前端已断开，当前连接数: {len(clients)}")  # 打印断开信息
```

##### `start(host, port)`

启动 WebSocket 服务器。

```python
def start(host="localhost", port=8765):
    """
    参数说明：
        host: 监听地址，默认 "localhost"
        port: 监听端口，默认 8765
    
    调用示例：
        server.start()  # 使用默认参数
        server.start("0.0.0.0", 9000)  # 监听所有网卡，端口9000
    """
    print(f"WebSocket服务启动中... ws://{host}:{port}")  # 打印启动信息

    async def main():  # 定义异步主函数
        async with websockets.serve(handleConnection, host, port):  # 创建WebSocket服务器
            print(f"WebSocket服务已启动: ws://{host}:{port}")  # 打印启动成功信息
            await asyncio.Future()  # 保持运行，永不结束

    asyncio.run(main())  # 运行异步主函数
```

#### 消息流程图

```
前端发送消息
    ↓
handleConnection 接收
    ↓
handleMessage 路由
    ↓
├─ getRegistry → registry.getAllForFrontend() → sendMessage
└─ runBlueprint → engine.run() → 实时回调 → sendMessage/sendError
```

---

### 2. registry.py - 节点注册表

**职责**：管理所有节点的定义，提供装饰器注册节点，创建节点实例。

#### 全局变量

```python
nodes = {}       # 节点定义字典，格式：{opcode: node}
categories = {}  # 分类定义字典，格式：{id: category}
```

#### 数据结构

**分类（Category）结构**：
```python
{
    "label": "分类显示名称",
    "color": "#8B92E5",  # 分类颜色
    "icon": "",          # 分类图标（base64）
    "nodes": []          # 该分类下的节点opcode列表
}
```

**节点（Node）结构**：
```python
{
    "opcode": "节点操作码",
    "label": "节点显示名称",
    "ports": {
        "input": {"in1": "输入1", "in2": "输入2"},
        "output": {"out": "输出"}
    },
    "params": {
        "param1": {
            "label": "参数显示名称",
            "type": "int",  # int/float/bool/str/list/enum
            "value": 默认值,
            "range": [最小值, 最大值],  # 可选
            "options": {}  # enum类型必需
        }
    },
    "cls": NodeClass  # 节点类（不发给前端）
}
```

#### 核心函数

##### `registerCategory(id, label, color, icon)`

注册一个节点分类。

```python
def registerCategory(id, label, color, icon):
    """
    参数说明：
        id: 分类唯一标识
        label: 分类显示名称
        color: 分类颜色（十六进制）
        icon: 分类图标（base64字符串）
    
    调用示例：
        registerCategory("basic", "基础", "#8B92E5", "")
    """
    category = {}  # 创建空字典
    category["label"] = label  # 设置显示名称
    category["color"] = color  # 设置颜色
    category["icon"] = icon  # 设置图标
    category["nodes"] = []  # 初始化节点列表为空
    categories[id] = category  # 存入全局分类字典
```

##### `registerNode(opcode, label, ports, params, cls)`

注册一个节点。

```python
def registerNode(opcode, label, ports, params, cls):
    """
    参数说明：
        opcode: 节点操作码（唯一标识）
        label: 节点显示名称
        ports: 端口定义字典
        params: 参数定义字典
        cls: 节点类（继承自BaseNode）
    
    调用示例：
        registerNode("input", "输入", {...}, {...}, InputNode)
    """
    node = {}  # 创建空字典
    node["opcode"] = opcode  # 设置操作码
    node["label"] = label  # 设置显示名称
    node["ports"] = ports  # 设置端口定义
    node["params"] = params  # 设置参数定义
    node["cls"] = cls  # 保存节点类
    nodes[opcode] = node  # 存入全局节点字典
    categories[list(categories.keys())[-1]]["nodes"].append(opcode)  # 加入最后一个分类
```

**重要**：节点会自动加入最后注册的分类，所以要先调用 `category()` 再调用 `@node`。

##### `getAllForFrontend()`

获取前端格式的节点注册表。

```python
def getAllForFrontend():
    """
    返回格式：
    {
        "categories": {分类字典},
        "nodes": {节点字典（去掉cls字段）}
    }
    
    调用示例：
        data = registry.getAllForFrontend()
        # 发送给前端用于渲染节点面板
    """
    result = {"categories": categories, "nodes": {}}  # 创建结果字典
    for opcode, node in nodes.items():  # 遍历所有节点
        result["nodes"][opcode] = {k: v for k, v in node.items() if k != "cls"}  # 去掉cls字段
    return result  # 返回前端格式数据
```

**为什么要去掉 cls**：Python 类对象无法序列化为 JSON，前端也不需要这个信息。

##### `createNode(opcode, nodeId, params)`

根据 opcode 创建节点实例。

```python
def createNode(opcode, nodeId, params):
    """
    参数说明：
        opcode: 节点操作码
        nodeId: 节点实例ID（前端生成的唯一ID）
        params: 节点参数字典
    
    返回：
        节点实例（BaseNode子类）
    
    调用示例：
        instance = registry.createNode("input", "node_123", {"out_shape": [2, 4, 10]})
    """
    if opcode not in nodes:  # 检查opcode是否已注册
        raise ValueError(f"未知节点: {opcode}")  # 抛出异常
    cls = nodes[opcode]["cls"]  # 获取节点类
    return cls(nodeId, params)  # 创建并返回节点实例
```

##### `category(id, label, color, icon)` - 装饰器

注册分类的装饰器（实际上不装饰任何东西，只是调用注册函数）。

```python
def category(id="", label="", color="#8992eb", icon=""):
    """
    使用示例：
        category(
            id="basic",
            label="基础",
            color="#8B92E5",
            icon=""
        )
    """
    registerCategory(id, label, color, icon)  # 直接调用注册函数
```

##### `node(opcode, label, ports, params)` - 装饰器

注册节点的装饰器。

```python
def node(opcode="", label="", ports={}, params={}):
    """
    使用示例：
        @node(
            opcode="input",
            label="输入",
            ports={"input": {}, "output": {"out": ""}},
            params={"out_shape": {"label": "输出维度", "type": "list", "value": [2, 4, 10]}}
        )
        class InputNode(BaseNode):
            def compute(self, input):
                return {"out": torch.rand(*self.params["out_shape"])}
    """
    def decorator(cls):  # 装饰器函数
        registerNode(opcode, label, ports, params, cls)  # 注册节点
        return cls  # 返回原类（不修改）
    return decorator  # 返回装饰器函数
```

#### BaseNode 基类

所有节点的基类，继承自 `nn.Module`。

```python
class BaseNode(nn.Module):
    """
    所有节点必须继承此类
    
    生命周期：
        1. __init__ - 初始化，保存nodeId和params
        2. build - 构建层（可选重写）
        3. compute - 执行计算（必须重写）
        4. forward - PyTorch调用入口（已实现，无需重写）
    """
    
    def __init__(self, nodeId, params):
        """
        参数说明：
            nodeId: 节点实例ID
            params: 节点参数字典
        """
        super().__init__()  # 调用父类nn.Module的初始化
        self.nodeId = nodeId  # 保存节点ID
        self.params = params  # 保存参数字典
        self.build()  # 调用build方法

    def build(self):
        """
        构建层的方法，子类可重写
        
        使用示例：
            def build(self):
                self.linear = nn.Linear(256, 128)
                self.relu = nn.ReLU()
        """
        pass  # 默认什么都不做

    def compute(self, input):
        """
        计算方法，子类必须实现
        
        参数说明：
            input: 输入字典，格式：{"端口名": 值}
        
        返回：
            输出字典，格式：{"端口名": 值}
        
        使用示例：
            def compute(self, input):
                x = input.get("x")
                y = input.get("y")
                result = self.linear(x + y)
                return {"out": result}
        """
        raise NotImplementedError("必须实现compute")  # 抛出异常

    def forward(self, input):
        """
        PyTorch的forward方法，已实现，无需重写
        
        调用compute并返回结果
        """
        out = self.compute(input)  # 调用compute方法
        # 占位，到时候做值存储和转发操作
        return out  # 返回输出结果
```

#### 使用流程

```
1. 定义分类
   category(id="basic", label="基础", ...)
   
2. 定义节点
   @node(opcode="input", label="输入", ...)
   class InputNode(BaseNode):
       def compute(self, input):
           ...
   
3. loader.py 自动导入节点模块
   装饰器自动执行，节点注册到全局字典
   
4. 前端请求节点列表
   getAllForFrontend() 返回所有节点定义
   5. 执行时创建实例
      createNode(opcode, nodeId, params)
   ```
   
   ---
   
   ### 3. loader.py - 动态加载器
   
   **职责**：自动扫描并导入 nodes 目录下的所有节点模块。
   
   #### 核心函数
   
   ##### `importModule(filepath)`
   
   动态导入指定路径的 Python 模块。
   
   ```python
   def importModule(filepath):
       """
       参数说明：
           filepath: 相对路径，如 "nodes/example.py"
       
       工作原理：
           1. 把路径转换成模块名格式
           2. 使用 importlib 动态导入
       
       调用示例：
           importModule("nodes/example.py")
           # 会被转换成 nodes.example 模块并导入
       """
       relative = filepath.replace("\\", "/")  # 把Windows路径的反斜杠替换成正斜杠
       noExt = relative.replace(".py", "")  # 去掉.py后缀
       moduleName = noExt.replace("/", ".")  # 把路径分隔符替换成点号，变成模块名格式
       importlib.import_module(moduleName)  # 使用importlib动态导入这个模块
   ```
   
   **路径转换示例**：
   ```
   nodes\example.py  →  nodes/example.py  →  nodes/example  →  nodes.example
   ```
   
   ##### `loadAll(folder)`
   
   加载指定文件夹下的所有节点模块。
   
   ```python
   def loadAll(folder="nodes"):
       """
       参数说明：
           folder: 节点文件夹路径，默认 "nodes"
       
       工作流程：
           1. 遍历文件夹下所有文件
           2. 跳过 __pycache__ 和 __init__.py
           3. 只处理 .py 文件
           4. 动态导入每个模块
       
       调用示例：
           loadAll()  # 自动加载 nodes/*.py
           loadAll("custom_nodes")  # 加载自定义文件夹
       """
       nodesDir = os.path.join(os.path.dirname(__file__), folder)  # 获取nodes文件夹的绝对路径
   
       for filename in os.listdir(nodesDir):  # 遍历nodes文件夹下的所有文件
           if filename == "__pycache__":  # 如果是__pycache__文件夹
               continue  # 跳过，不处理
   
           if filename == "__init__.py":  # 如果是__init__.py文件
               continue  # 跳过，不处理
   
           if not filename.endswith(".py"):  # 如果不是.py文件
               continue  # 跳过，不处理
   
           filepath = os.path.join(folder, filename)  # 拼接相对路径，比如nodes/math.py
           importModule(filepath)  # 动态导入这个模块
           print(f"已加载节点模块: {filepath}")  # 打印加载信息
   ```
   
   #### 工作原理
   
   ```
   启动时（engine.py 导入时）
       ↓
   loader.loadAll() 被调用
       ↓
   遍历 nodes/ 目录
       ↓
   导入每个 .py 文件
       ↓
   @category 和 @node 装饰器自动执行
       ↓
   节点自动注册到 registry
   ```
   
   **关键点**：
   - 装饰器在模块导入时就会执行
   - 不需要手动调用注册函数
   - 新增节点文件后重启服务即可自动加载
   
   ---
   
   ### 4. sort.py - 拓扑排序
   
   **职责**：根据节点依赖关系确定执行顺序，检测循环依赖。
   
   #### 核心函数
   
   ##### `topoSort(nodes, edges)`
   
   对节点进行拓扑排序。
   
   ```python
   def topoSort(nodes, edges):
       """
       参数说明：
           nodes: 节点列表，格式：[{"id": "node1"}, {"id": "node2"}, ...]
           edges: 边列表，格式：[{"source": "node1", "target": "node2"}, ...]
       
       返回：
           排序后的节点ID列表，格式：["node1", "node2", "node3"]
       
       异常：
           如果存在循环依赖，抛出 Exception
       
       调用示例：
           nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
           edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
           result = topoSort(nodes, edges)  # 返回 ["a", "b", "c"]
       """
   ```
   
   #### 算法详解
   
   **步骤1：初始化数据结构**
   
   ```python
   inDegree = {}  # 入度表，记录每个节点有多少个前置节点
   adjacency = {}  # 邻接表，记录每个节点指向哪些后继节点
   
   for node in nodes:  # 遍历所有节点
       nodeId = node.get("id", "")  # 获取节点id
       inDegree[nodeId] = 0  # 初始化入度为0
       adjacency[nodeId] = []  # 初始化邻接列表为空
   ```
   
   **步骤2：构建图结构**
   
   ```python
   for edge in edges:  # 遍历所有边
       source = edge.get("source", "")  # 获取边的源节点
       target = edge.get("target", "")  # 获取边的目标节点
   
       if source not in adjacency:  # 如果源节点不在邻接表中
           continue  # 跳过这条边
   
       if target not in inDegree:  # 如果目标节点不在入度表中
           continue  # 跳过这条边
   
       adjacency[source].append(target)  # 把目标节点加入源节点的邻接列表
       inDegree[target] = inDegree[target] + 1  # 目标节点的入度加1
   ```
   
   **步骤3：Kahn算法（BFS拓扑排序）**
   
   ```python
   queue = deque()  # 创建队列，用于BFS
   
   for nodeId in inDegree:  # 遍历所有节点
       if inDegree[nodeId] == 0:  # 如果节点入度为0（没有前置依赖）
           queue.append(nodeId)  # 加入队列
   
   result = []  # 结果列表，存储排序后的节点id
   
   while len(queue) > 0:  # 循环处理队列直到队列为空
       current = queue.popleft()  # 弹出队首节点
       result.append(current)  # 加入结果列表
   
       for neighbor in adjacency[current]:  # 遍历当前节点的所有后继节点
           inDegree[neighbor] = inDegree[neighbor] - 1  # 后继节点入度减1
   
           if inDegree[neighbor] == 0:  # 如果后继节点入度变成0
               queue.append(neighbor)  # 加入队列
   ```
   
   **步骤4：检测循环依赖**
   
   ```python
   if len(result) != len(nodes):  # 如果结果数量不等于节点数量
       raise Exception("存在循环依赖，无法进行拓扑排序")  # 说明有环，抛出异常
   
   return result  # 返回排序结果数组
   ```
   
   #### 算法示例
   
   **示例1：简单链式**
   
   ```
   输入：
   nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
   edges = [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}]
   
   图结构：
   A → B → C
   
   执行过程：
   1. 初始入度：A=0, B=1, C=1
   2. 队列初始：[A]
   3. 处理A：结果=[A]，B入度-1=0，队列=[B]
   4. 处理B：结果=[A,B]，C入度-1=0，队列=[C]
   5. 处理C：结果=[A,B,C]，队列=[]
   
   输出：["A", "B", "C"]
   ```
   
   **示例2：并行分支**
   
   ```
   输入：
   nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}]
   edges = [
       {"source": "A", "target": "C"},
       {"source": "B", "target": "C"},
       {"source": "C", "target": "D"}
   ]
   
   图结构：
   A ↘
       C → D
   B ↗
   
   执行过程：
   1. 初始入度：A=0, B=0, C=2, D=1
   2. 队列初始：[A, B]
   3. 处理A：结果=[A]，C入度-1=1，队列=[B]
   4. 处理B：结果=[A,B]，C入度-1=0，队列=[C]
   5. 处理C：结果=[A,B,C]，D入度-1=0，队列=[D]
   6. 处理D：结果=[A,B,C,D]，队列=[]
   
   输出：["A", "B", "C", "D"] 或 ["B", "A", "C", "D"]
   （A和B可以并行，顺序不固定）
   ```
   
   **示例3：循环依赖（错误）**
   
   ```
   输入：
   nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
   edges = [
       {"source": "A", "target": "B"},
       {"source": "B", "target": "C"},
       {"source": "C", "target": "A"}  # 形成环
   ]
   
   图结构：
   A → B → C → A（环）
   
   执行过程：
   1. 初始入度：A=1, B=1, C=1
   2. 队列初始：[]（没有入度为0的节点）
   3. 循环结束，result=[]
   
   检测：len(result)=0 != len(nodes)=3
   抛出异常："存在循环依赖，无法进行拓扑排序"
   ```
   
   #### 时间复杂度
   
   - **时间复杂度**：O(V + E)，V是节点数，E是边数
   - **空间复杂度**：O(V)，需要存储入度表和邻接表
   
   ---
   
   ### 5. engine.py - 蓝图执行引擎
   
   **职责**：解析蓝图数据，创建节点实例，按拓扑顺序执行所有节点。
   
   #### 初始化
   
   ```python
   loader.loadAll()  # 模块导入时自动加载所有节点
   ```
   
   **重要**：这行代码在模块级别执行，意味着 `import engine` 时就会自动加载所有节点。
   
   #### 核心函数
   
   ##### `run(blueprint, onMessage, onError)`
   
   运行蓝图的主函数。
   
   ```python
   async def run(blueprint, onMessage, onError):
       """
       参数说明：
           blueprint: 蓝图数据字典
               {
                   "nodes": [节点列表],
                   "edges": [边列表]
               }
           onMessage: 节点执行成功的回调函数
               async def(nodeId, result): pass
           onError: 节点执行失败的回调函数
               async def(nodeId, error): pass
       
       调用示例：
           async def onMsg(nodeId, result):
               print(f"节点{nodeId}执行完成: {result}")
           
           async def onErr(nodeId, error):
               print(f"节点{nodeId}执行出错: {error}")
           
           await engine.run(blueprintData, onMsg, onErr)
       """
   ```
   
   #### 执行流程
   
   **阶段0：准备工作**
   
   ```python
   nodes = blueprint.get("nodes", [])  # 从蓝图中提取节点列表
   edges = blueprint.get("edges", [])  # 从蓝图中提取边列表
   
   sortedIds = sort.topoSort(nodes, edges)  # 调用拓扑排序，得到执行顺序
   print(f"拓扑排序结果: {sortedIds}")  # 打印排序结果用于调试
   
   nodeMap = {}  # 创建节点id到节点数据的映射字典
   for node in nodes:  # 遍历所有节点
       nodeId = node.get("id", "")  # 获取节点id
       nodeMap[nodeId] = node  # 存入映射字典方便后续查找
   
   instances = {}  # 存储所有节点的实例，格式：{nodeId: BaseNode实例}
   results = {}  # 存储所有节点的输出结果，格式：{nodeId: {port: value}}
   ```
   
   **阶段1：创建所有节点实例**
   
   ```python
   print("开始创建节点实例...")  # 打印阶段信息
   for nodeId in sortedIds:  # 按拓扑顺序遍历节点id
       node = nodeMap.get(nodeId)  # 根据id获取节点数据
       if node is None:  # 如果找不到节点数据
           await onError(nodeId, f"节点数据不存在: {nodeId}")  # 发送错误回调
           return  # 终止执行
   
       data = node.get("data", {})  # 获取节点的data字段
       opcode = data.get("opcode", "")  # 从data中获取opcode
       params = data.get("params", {})  # 从data中获取params参数字典
   
       if opcode not in registry.nodes:  # 检查opcode是否已注册
           await onError(nodeId, f"未知的节点类型: {opcode}")  # 发送错误回调
           return  # 终止执行
   
       try:  # 尝试创建节点实例
           instance = registry.createNode(opcode, nodeId, params)  # 调用registry创建实例
           instances[nodeId] = instance  # 存入实例字典
           print(f"节点实例创建成功: {nodeId} ({opcode})")  # 打印成功信息
       except Exception as e:  # 如果创建失败
           await onError(nodeId, f"创建节点实例失败: {str(e)}")  # 发送错误回调
           return  # 终止执行
   ```
   
   **为什么分两个阶段**：
   1. 先创建所有实例，确保所有节点都能正常初始化
   2. 再执行计算，避免执行到一半发现后面的节点无法创建
   
   **阶段2：按拓扑顺序执行所有节点**
   
   ```python
   print("开始执行节点...")  # 打印阶段信息
   for nodeId in sortedIds:  # 按拓扑顺序遍历节点id
       instance = instances.get(nodeId)  # 获取当前节点的实例
       if instance is None:  # 如果实例不存在（理论上不会发生）
           await onError(nodeId, f"节点实例不存在: {nodeId}")  # 发送错误回调
           return  # 终止执行
   
       # 收集当前节点的输入
       inputValues = {}  # 创建空字典准备装输入值
       for edge in edges:  # 遍历所有边
           targetId = edge.get("target", "")  # 获取边的目标节点id
           if targetId != nodeId:  # 如果目标不是当前节点
               continue  # 跳过这条边
   
           sourceId = edge.get("source", "")  # 获取源节点id
           sourcePort = edge.get("sourceHandle", "out")  # 获取源端口名，默认out
           targetPort = edge.get("targetHandle", "in")  # 获取目标端口名，默认in
   
           sourceResult = results.get(sourceId, {})  # 获取源节点的输出结果字典
           value = sourceResult.get(sourcePort, None)  # 获取对应端口的值
           inputValues[targetPort] = value  # 存入输入字典
   
       # 执行节点的compute方法
       try:  # 尝试执行计算
           output = instance.compute(inputValues)  # 调用实例的compute方法
           results[nodeId] = output  # 存储输出结果
           await onMessage(nodeId, output)  # 发送成功回调
           print(f"节点执行成功: {nodeId}, 输出: {output}")  # 打印执行结果
       except Exception as e:  # 如果执行出错
           await onError(nodeId, f"执行出错: {str(e)}")  # 发送错误回调
           print(f"节点执行失败: {nodeId}, 错误: {str(e)}")  # 打印错误信息
           return  # 终止执行
   
   print("蓝图执行完成")  # 打印完成信息
   ```
   
   #### 数据流转示例
   
   **蓝图数据**：
   ```json
   {
     "nodes": [
       {"id": "node1", "data": {"opcode": "input", "params": {"out_shape": [2, 4, 10]}}},
       {"id": "node2", "data": {"opcode": "debug", "params": {}}},
       {"id": "node3", "data": {"opcode": "output", "params": {}}}
     ],
     "edges": [
       {"source": "node1", "sourceHandle": "out", "target": "node2", "targetHandle": "x"},
       {"source": "node2", "sourceHandle": "out", "target": "node3", "targetHandle": "in"}
     ]
   }
   ```
   
   **执行过程**：
   
   ```
   1. 拓扑排序：["node1", "node2", "node3"]
   
   2. 创建实例：
      - node1: InputNode(nodeId="node1", params={"out_shape": [2,4,10]})
      - node2: DebugNode(nodeId="node2", params={})
      - node3: OutputNode(nodeId="node3", params={})
   
   3. 执行 node1：
      - 输入：{}（没有输入）
      - 计算：torch.rand(2, 4, 10)
      - 输出：{"out": tensor([2,4,10])}
      - 存储：results["node1"] = {"out": tensor([2,4,10])}
   
   4. 执行 node2：
      - 收集输入：edge: node1.out → node2.x
      - 输入：{"x": tensor([2,4,10])}
      - 计算：打印调试信息，透传
      - 输出：{"out": tensor([2,4,10])}
      - 存储：results["node2"] = {"out": tensor([2,4,10])}
   
   5. 执行 node3：
      - 收集输入：edge: node2.out → node3.in
      - 输入：{"in": tensor([2,4,10])}
      - 计算：打印最终输出
      - 输出：{}（输出节点没有输出）
      - 存储：results["node3"] = {}
   
   6. 完成
   ```
   
   #### 错误处理
   
   引擎在以下情况会终止执行并回调 `onError`：
   
   1. **节点数据不存在**：拓扑排序返回的ID在nodeMap中找不到
   2. **未知节点类型**：opcode未在registry中注册
   3. **创建实例失败**：节点类的 `__init__` 或 `build` 方法抛出异常
   4. **执行失败**：节点的 `compute` 方法抛出异常
   **错误传播**：任何节点出错都会立即终止整个蓝图的执行，不会继续执行后续节点。
   
   ---
   
   ## 🔧 节点开发指南
   
   ### 创建新节点的完整流程
   
   #### 步骤1：选择或创建节点文件
   
   在 `nodes/` 目录下创建或选择一个 `.py` 文件：
   
   ```bash
   # 创建新的节点文件
   touch nodes/my_nodes.py
   ```
   
   #### 步骤2：导入必要的模块
   
   ```python
   """
   nodes/my_nodes.py - 我的自定义节点
   """
   
   import torch  # 导入torch用于张量操作
   import torch.nn as nn  # 导入nn模块用于定义层
   from registry import category, node, BaseNode  # 从registry导入装饰器和基类
   ```
   
   #### 步骤3：注册节点分类
   
   ```python
   # 注册一个新分类（可选，如果已有分类可跳过）
   category(
       id="my_category",  # 分类唯一标识，不能与其他分类重复
       label="我的节点",  # 分类显示名称，前端会显示这个
       color="#FF6B6B",  # 分类颜色，十六进制格式
       icon=""  # 分类图标，base64格式（可选）
   )
   ```
   
   #### 步骤4：定义节点类
   
   ```python
   @node(
       opcode="my_custom_node",  # 节点操作码，全局唯一
       label="我的自定义节点",  # 节点显示名称
       ports={
           "input": {  # 输入端口定义
               "x": "输入X",  # 端口名: 显示文字
               "y": "输入Y"
           },
           "output": {  # 输出端口定义
               "result": "计算结果"
           }
       },
       params={  # 参数定义
           "scale": {
               "label": "缩放因子",  # 参数显示名称
               "type": "float",  # 参数类型
               "value": 1.0,  # 默认值
               "range": [0.1, 10.0]  # 取值范围（可选）
           }
       }
   )
   class MyCustomNode(BaseNode):  # 继承BaseNode
       """
       我的自定义节点
       
       功能：将两个输入相加并乘以缩放因子
       """
       
       def build(self):
           """
           构建层（可选）
           
           在这里初始化需要的神经网络层
           如果不需要层，可以不重写这个方法
           """
           pass  # 这个节点不需要层
       
       def compute(self, input):
           """
           计算方法（必须实现）
           
           参数：
               input: 输入字典，格式：{"端口名": 值}
           
           返回：
               输出字典，格式：{"端口名": 值}
           """
           x = input.get("x")  # 获取输入x
           y = input.get("y")  # 获取输入y
           scale = self.params.get("scale", 1.0)  # 获取缩放因子参数
           
           result = (x + y) * scale  # 执行计算
           
           return {"result": result}  # 返回输出字典
   ```
   
   #### 步骤5：重启服务
   
   ```bash
   # 重启服务，新节点会自动加载
   uv run python main.py
   ```
   
   ### 参数类型详解
   
   #### 1. int - 整数参数
   
   ```python
   "int_param": {
       "label": "整数参数",
       "type": "int",
       "value": 256,  # 默认值
       "range": [1, 1024]  # 可选：最小值和最大值
   }
   ```
   
   **使用场景**：层的维度、批次大小、迭代次数等。
   
   #### 2. float - 浮点数参数
   
   ```python
   "float_param": {
       "label": "浮点数参数",
       "type": "float",
       "value": 0.001,  # 默认值
       "range": [0.0, 1.0]  # 可选：最小值和最大值
   }
   ```
   
   **使用场景**：学习率、dropout率、权重衰减等。
   
   #### 3. bool - 布尔参数
   
   ```python
   "bool_param": {
       "label": "布尔参数",
       "type": "bool",
       "value": True  # 默认值
   }
   ```
   
   **使用场景**：是否使用bias、是否启用dropout等。
   
   #### 4. str - 字符串参数
   
   ```python
   "str_param": {
       "label": "字符串参数",
       "type": "str",
       "value": "默认文本"  # 默认值
   }
   ```
   
   **使用场景**：文件路径、模型名称、激活函数名称等。
   
   #### 5. list - 列表参数
   
   ```python
   "list_param": {
       "label": "列表参数",
       "type": "list",
       "value": [1, 2, 3]  # 默认值
   }
   ```
   
   **使用场景**：张量形状、卷积核大小、多个超参数等。
   
   #### 6. enum - 枚举参数
   
   ```python
   "enum_param": {
       "label": "枚举参数",
       "type": "enum",
       "value": "option1",  # 默认值（必须是options中的一个键）
       "options": {  # 选项字典
           "option1": "选项1",
           "option2": "选项2",
           "option3": "选项3"
       }
   }
   ```
   
   **使用场景**：激活函数选择、优化器类型、损失函数类型等。
   
   ### 常见节点模式
   
   #### 模式1：纯计算节点（无需层）
   
   ```python
   @node(
       opcode="add",
       label="加法",
       ports={"input": {"a": "", "b": ""}, "output": {"out": ""}},
       params={}
   )
   class AddNode(BaseNode):
       def compute(self, input):
           a = input.get("a")  # 获取输入a
           b = input.get("b")  # 获取输入b
           return {"out": a + b}  # 返回相加结果
   ```
   
   #### 模式2：带层的节点
   
   ```python
   @node(
       opcode="linear",
       label="全连接层",
       ports={"input": {"x": ""}, "output": {"out": ""}},
       params={
           "in_features": {"label": "输入维度", "type": "int", "value": 256},
           "out_features": {"label": "输出维度", "type": "int", "value": 128},
           "use_bias": {"label": "使用偏置", "type": "bool", "value": True}
       }
   )
   class LinearNode(BaseNode):
       def build(self):
           in_feat = self.params.get("in_features")  # 获取输入维度
           out_feat = self.params.get("out_features")  # 获取输出维度
           use_bias = self.params.get("use_bias")  # 获取是否使用偏置
           self.linear = nn.Linear(in_feat, out_feat, bias=use_bias)  # 创建线性层
       
       def compute(self, input):
           x = input.get("x")  # 获取输入
           out = self.linear(x)  # 通过线性层
           return {"out": out}  # 返回输出
   ```
   
   #### 模式3：多输入多输出节点
   
   ```python
   @node(
       opcode="split",
       label="分割",
       ports={
           "input": {"x": ""},
           "output": {"first": "前半部分", "second": "后半部分"}
       },
       params={}
   )
   class SplitNode(BaseNode):
       def compute(self, input):
           x = input.get("x")  # 获取输入
           mid = x.shape[-1] // 2  # 计算中点
           first = x[..., :mid]  # 前半部分
           second = x[..., mid:]  # 后半部分
           return {"first": first, "second": second}  # 返回两个输出
   ```
   
   #### 模式4：条件节点（根据参数选择行为）
   
   ```python
   @node(
       opcode="activation",
       label="激活函数",
       ports={"input": {"x": ""}, "output": {"out": ""}},
       params={
           "activation": {
               "label": "激活函数",
               "type": "enum",
               "value": "relu",
               "options": {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh"}
           }
       }
   )
   class ActivationNode(BaseNode):
       def build(self):
           act_type = self.params.get("activation")  # 获取激活函数类型
           if act_type == "relu":  # 如果是ReLU
               self.act = nn.ReLU()  # 创建ReLU
           elif act_type == "sigmoid":  # 如果是Sigmoid
               self.act = nn.Sigmoid()  # 创建Sigmoid
           elif act_type == "tanh":  # 如果是Tanh
               self.act = nn.Tanh()  # 创建Tanh
       
       def compute(self, input):
           x = input.get("x")  # 获取输入
           out = self.act(x)  # 通过激活函数
           return {"out": out}  # 返回输出
   ```
   
   ### 节点开发注意事项
   
   #### ✅ 应该做的
   
   1. **每行代码都加注释**：遵循项目风格，用大白话解释每一行
   2. **参数验证**：在 `build()` 或 `compute()` 中验证参数合法性
   3. **错误处理**：使用 `try-except` 捕获可能的异常
   4. **返回正确格式**：`compute()` 必须返回字典，键是端口名
   5. **使用 `self.params`**：通过 `self.params.get()` 获取参数
   6. **文档注释**：在类和方法开头写清楚用途
   
   #### ❌ 不应该做的
   
   1. **不要修改输入**：输入可能被其他节点使用，应该创建新张量
   2. **不要在 `compute()` 中创建层**：层应该在 `build()` 中创建
   3. **不要使用全局变量**：所有状态应该保存在实例中
   4. **不要假设输入存在**：使用 `input.get()` 而不是 `input[]`
   5. **不要返回 None**：至少返回空字典 `{}`
   6. **不要在节点间共享状态**：每个节点应该是独立的
   
   ---
   
   ## 📝 代码风格规范
   
   本项目遵循严格的代码风格规范，确保代码易于理解和维护。
   
   ### 核心原则
   
   #### 1. 面向理解编程
   
   **代码逻辑要符合人类直觉**
   
   ```python
   # ✅ 好的写法：符合直觉
   result = []  # 创建结果列表
   for item in items:  # 遍历所有项目
       if item > 0:  # 如果项目大于0
           result.append(item)  # 加入结果列表
   
   # ❌ 不好的写法：过于简洁但不直观
   result = [item for item in items if item > 0]
   ```
   
   #### 2. 命令式写法
   
   **采用命令式而不是声明式，像积木一样的代码**
   
   ```python
   # ✅ 好的写法：命令式，一步一步
   data = getData()  # 获取数据
   filtered = filterData(data)  # 过滤数据
   sorted = sortData(filtered)  # 排序数据
   result = formatData(sorted)  # 格式化数据
   
   # ❌ 不好的写法：链式调用
   result = formatData(sortData(filterData(getData())))
   ```
   
   #### 3. 积木式行状写法
   
   **减少嵌套与缩进，每行代码独立完成一个任务**
   
   ```python
   # ✅ 好的写法：扁平化
   if not data:  # 如果没有数据
       return  # 直接返回
   
   if not valid:  # 如果数据无效
       return  # 直接返回
   
   process(data)  # 处理数据
   
   # ❌ 不好的写法：嵌套过深
   if data:
       if valid:
           process(data)
   ```
   
   ### 注释规范
   
   #### 1. 每行代码都需要尾随注释
   
   ```python
   # ✅ 正确示例
   def calculate(x, y):  # 计算两个数的和
       result = x + y  # 执行加法运算
       return result  # 返回计算结果
   
   # ❌ 错误示例：缺少注释
   def calculate(x, y):
       result = x + y
       return result
   ```
   
   #### 2. 注释要用大白话
   
   ```python
   # ✅ 好的注释：大白话，易懂
   nodes = {}  # 节点定义字典，格式：{opcode: node}
   for node in nodes:  # 遍历所有节点
       print(node)  # 打印节点信息
   
   # ❌ 不好的注释：过于技术化
   nodes = {}  # 哈希表存储节点元数据
   for node in nodes:  # 迭代节点集合
       print(node)  # 输出节点对象
   ```
   
   #### 3. 函数开头要标明用法和示例
   
   ```python
   def sendMessage(ws, type, id, data):
       """
       发送消息给前端
       
       用法：
           await sendMessage(ws, "getNodes", "req1", nodesData)
       
       示例：
           await sendMessage(websocket, "getNodes", "req1", nodesData)  # 发送节点数据
           await sendMessage(websocket, "nodeComplete", "req2", result)  # 发送节点执行结果
       """
       msg = {}  # 创建空字典准备装消息
       msg["type"] = type  # 消息类型
       msg["id"] = id  # 消息ID
       msg["data"] = data  # 消息数据
       text = json.dumps(msg)  # 把字典转成JSON字符串
       await ws.send(text)  # 通过WebSocket发送给前端
   ```
   
   ### 命名规范
   
   #### 1. 全部使用驼峰命名法
   
   ```python
   # ✅ 正确：驼峰命名
   nodeId = "node_123"  # 节点ID
   inputValues = {}  # 输入值字典
   sortedIds = []  # 排序后的ID列表
   
   # ❌ 错误：下划线命名
   node_id = "node_123"
   input_values = {}
   sorted_ids = []
   ```
   
   #### 2. 变量名要简洁易懂
   
   ```python
   # ✅ 好的命名：简洁且符合语境
   nodes = []  # 节点列表
   edges = []  # 边列表
   result = {}  # 结果字典
   
   # ❌ 不好的命名：过于冗长或模糊
   listOfAllNodesInTheBlueprint = []
   e = []
   temp = {}
   ```
   
   #### 3. 符合项目语境
   
   ```python
   # ✅ 符合项目语境
   opcode = "input"  # 节点操作码
   blueprint = {}  # 蓝图数据
   registry = {}  # 注册表
   
   # ❌ 不符合项目语境
   code = "input"  # 太模糊
   graph = {}  # 不是项目术语
   catalog = {}  # 不是项目术语
   ```
   
   ### 代码结构规范
   
   #### 1. 函数要短小精悍
   
   ```python
   # ✅ 好的函数：单一职责
   def createNode(opcode, nodeId, params):  # 创建节点实例
       if opcode not in nodes:  # 检查opcode是否已注册
           raise ValueError(f"未知节点: {opcode}")  # 抛出异常
       cls = nodes[opcode]["cls"]  # 获取节点类
       return cls(nodeId, params)  # 创建并返回节点实例
   
   # ❌ 不好的函数：做太多事情
   def processEverything(data):
       # 100行代码做各种事情...
       pass
   ```
   
   #### 2. 避免过度抽象
   
   ```python
   # ✅ 好的写法：直接明了
   if msg_type == "getRegistry":  # 如果是请求节点注册表
       result = registry.getAllForFrontend()  # 调用registry获取数据
       await sendMessage(ws, msg_type, id, result)  # 发送响应
       return  # 处理完毕
   
   # ❌ 不好的写法：过度抽象
   handler = getHandler(msg_type)
   result = handler.process()
   await handler.respond(ws, result)
   ```
   
   #### 3. 使用早返回减少嵌套
   
   ```python
   # ✅ 好的写法：早返回
   def process(data):
       if not data:  # 如果没有数据
           return None  # 直接返回
       
       if not validate(data):  # 如果数据无效
           return None  # 直接返回
       
       return transform(data)  # 处理数据
   
   # ❌ 不好的写法：嵌套
   def process(data):
       if data:
           if validate(data):
               return transform(data)
       return None
   ```
   
   ### Postel's Law 原则
   
   **接受多变，输出保守**
   
   ```python
   # ✅ 好的写法：灵活接受参数
   def start(host="localhost", port=8765):  # 提供默认值
       """
       调用示例：
           start()  # 使用默认参数
           start("0.0.0.0")  # 只指定host
           start(port=9000)  # 只指定port
           start("0.0.0.0", 9000)  # 都指定
       """
       print(f"启动服务: {host}:{port}")  # 打印信息
   
   # ❌ 不好的写法：参数不灵活
   def start(host, port):  # 必须提供所有参数
       print(f"启动服务: {host}:{port}")
   ```
   
   ### 错误处理规范
   
   #### 1. 使用明确的错误信息
   
   ```python
   # ✅ 好的错误信息：具体明确
   if opcode not in nodes:  # 检查opcode是否已注册
       raise ValueError(f"未知节点: {opcode}")  # 包含具体的opcode
   
   # ❌ 不好的错误信息：模糊不清
   if opcode not in nodes:
       raise ValueError("节点不存在")  # 不知道是哪个节点
   ```
   
   #### 2. 在合适的层级处理错误
   
   ```python
   # ✅ 好的错误处理：在调用层处理
   try:  # 尝试创建节点实例
       instance = registry.createNode(opcode, nodeId, params)  # 调用创建
       instances[nodeId] = instance  # 存入实例字典
   except Exception as e:  # 如果创建失败
       await onError(nodeId, f"创建节点实例失败: {str(e)}")  # 发送错误回调
       return  # 终止执行
   
   # ❌ 不好的错误处理：吞掉错误
   try:
       instance = registry.createNode(opcode, nodeId, params)
   except:
       pass  # 什么都不做
   ```
   
   ### 文件组织规范
   
   #### 1. 文件开头要有说明
   
   ```python
   """
   server.py - WebSocket服务器
   
   用法：
       import server
       server.start()  # 使用默认参数启动
       server.start("0.0.0.0", 9000)  # 指定host和port启动
   
   示例：
       server.start()  # 在localhost:8765启动WebSocket服务
   """
   ```
   
   #### 2. 导入顺序
   
   ```python
   # 1. 标准库
   import os  # 操作系统模块
   import json  # JSON库
   
   # 2. 第三方库
   import torch  # PyTorch
   import websockets  # WebSocket库
   
   # 3. 本地模块
   import registry  # 节点注册表
   import engine  # 执行引擎
   ```
   
   #### 3. 全局变量要说明用途
   
   ```python
   clients = set()  # 全局变量：已连接的前端客户端集合，用set存储方便增删
   nodes = {}  # 全局变量：节点定义字典，格式：{opcode: node}
   ```
   
   ### 代码审查清单
   
   在提交代码前，检查以下项目：
   
   - [ ] 每行代码都有尾随注释
   - [ ] 注释使用大白话，易于理解
   - [ ] 函数开头有用法说明和示例
   - [ ] 变量名使用驼峰命名法
   - [ ] 变量名简洁且符合项目语境
   - [ ] 代码采用命令式写法，像积木一样
   - [ ] 减少了嵌套，使用早返回
   - [ ] 函数单一职责，短小精悍
   - [ ] 错误信息明确具体
   - [ ] 文件开头有说明文档
   
   ---
   
   ## 💡 开发示例和最佳实践
   
   ### 完整示例：创建卷积节点
   
   让我们从零开始创建一个完整的卷积节点，展示所有最佳实践。
   
   #### 1. 创建文件 `nodes/conv.py`
   
   ```python
   """
   nodes/conv.py - 卷积层节点
   
   提供常用的卷积操作节点
   """
   
   import torch  # 导入torch用于张量操作
   import torch.nn as nn  # 导入nn模块用于定义层
   from registry import category, node, BaseNode  # 从registry导入装饰器和基类
   
   
   # ==================== 分类定义 ====================
   
   category(  # 注册卷积分类
       id="conv",  # 分类唯一标识
       label="卷积层",  # 分类显示名称
       color="#4ECDC4",  # 分类颜色
       icon="",  # 分类图标
   )
   
   
   # ==================== 节点定义 ====================
   
   
   @node(  # 注册Conv2d节点
       opcode="conv2d",  # 节点操作码
       label="2D卷积",  # 节点显示名称
       ports={  # 端口定义
           "input": {"x": "输入特征图"},  # 输入端口
           "output": {"out": "输出特征图"}  # 输出端口
       },
       params={  # 参数定义
           "in_channels": {
               "label": "输入通道数",
               "type": "int",
               "value": 3,
               "range": [1, 2048]
           },
           "out_channels": {
               "label": "输出通道数",
               "type": "int",
               "value": 64,
               "range": [1, 2048]
           },
           "kernel_size": {
               "label": "卷积核大小",
               "type": "int",
               "value": 3,
               "range": [1, 11]
           },
           "stride": {
               "label": "步长",
               "type": "int",
               "value": 1,
               "range": [1, 10]
           },
           "padding": {
               "label": "填充",
               "type": "int",
               "value": 1,
               "range": [0, 10]
           },
           "use_bias": {
               "label": "使用偏置",
               "type": "bool",
               "value": True
           }
       },
   )
   class Conv2dNode(BaseNode):  # 继承BaseNode
       """
       2D卷积节点
       
       功能：对输入特征图进行2D卷积操作
       
       输入：
           x: 形状为 (batch, in_channels, height, width) 的张量
       
       输出：
           out: 形状为 (batch, out_channels, height', width') 的张量
       """
       
       def build(self):
           """
           构建卷积层
           
           根据参数创建nn.Conv2d层
           """
           inCh = self.params.get("in_channels")  # 获取输入通道数
           outCh = self.params.get("out_channels")  # 获取输出通道数
           kernel = self.params.get("kernel_size")  # 获取卷积核大小
           stride = self.params.get("stride")  # 获取步长
           padding = self.params.get("padding")  # 获取填充
           useBias = self.params.get("use_bias")  # 获取是否使用偏置
           
           self.conv = nn.Conv2d(  # 创建Conv2d层
               in_channels=inCh,  # 输入通道数
               out_channels=outCh,  # 输出通道数
               kernel_size=kernel,  # 卷积核大小
               stride=stride,  # 步长
               padding=padding,  # 填充
               bias=useBias  # 是否使用偏置
           )
       
       def compute(self, input):
           """
           执行卷积计算
           
           参数：
               input: 输入字典，包含 "x" 键
           
           返回：
               输出字典，包含 "out" 键
           """
           x = input.get("x")  # 获取输入特征图
           
           if x is None:  # 如果输入为空
               raise ValueError("输入x不能为空")  # 抛出异常
           
           out = self.conv(x)  # 通过卷积层
           
           return {"out": out}  # 返回输出字典
   ```
   
   #### 2. 重启服务测试
   
   ```bash
   uv run python main.py
   ```
   
   你会看到：
   ```
   已加载节点模块: nodes/base.py
   已加载节点模块: nodes/conv.py
   已加载节点模块: nodes/example.py
   WebSocket服务已启动: ws://localhost:8765
   ```
   
   ### 最佳实践总结
   
   #### 1. 参数设计
   
   **提供合理的默认值**
   
   ```python
   # ✅ 好的参数设计
   params={
       "learning_rate": {
           "label": "学习率",
           "type": "float",
           "value": 0.001,  # 常用的默认值
           "range": [1e-6, 1.0]  # 合理的范围
       }
   }
   
   # ❌ 不好的参数设计
   params={
       "learning_rate": {
           "label": "学习率",
           "type": "float",
           "value": 0,  # 不合理的默认值
           "range": [0, 999999]  # 范围过大
       }
   }
   ```
   
   #### 2. 输入验证
   
   **始终验证输入的有效性**
   
   ```python
   def compute(self, input):
       x = input.get("x")  # 获取输入
       
       # ✅ 验证输入
       if x is None:  # 如果输入为空
           raise ValueError("输入x不能为空")  # 抛出明确的错误
       
       if x.dim() != 4:  # 如果维度不对
           raise ValueError(f"期望4维张量，得到{x.dim()}维")  # 抛出明确的错误
       
       out = self.process(x)  # 处理输入
       return {"out": out}  # 返回输出
   ```
   
   #### 3. 错误信息
   
   **提供有用的调试信息**
   
   ```python
   # ✅ 好的错误信息
   if x.shape[1] != self.params["in_channels"]:
       raise ValueError(
           f"输入通道数不匹配: 期望{self.params['in_channels']}, "
           f"得到{x.shape[1]}"
       )
   
   # ❌ 不好的错误信息
   if x.shape[1] != self.params["in_channels"]:
       raise ValueError("通道数错误")
   ```
   
   #### 4. 性能优化
   
   **避免不必要的计算**
   
   ```python
   # ✅ 好的写法：缓存计算结果
   def build(self):
       self.scale = self.params.get("scale", 1.0)  # 在build中计算一次
   
   def compute(self, input):
       x = input.get("x")
       return {"out": x * self.scale}  # 直接使用缓存的值
   
   # ❌ 不好的写法：重复计算
   def compute(self, input):
       x = input.get("x")
       scale = self.params.get("scale", 1.0)  # 每次都获取
       return {"out": x * scale}
   ```
   
   ### 常见问题和解决方案
   
   #### 问题1：节点没有被加载
   
   **症状**：前端看不到新创建的节点
   
   **解决方案**：
   1. 检查文件是否在 `nodes/` 目录下
   2. 检查文件名是否以 `.py` 结尾
   3. 检查是否有语法错误（查看启动日志）
   4. 重启服务
   
   #### 问题2：节点执行出错
   
   **症状**：前端显示节点执行失败
   
   **解决方案**：
   1. 查看后端控制台的错误信息
   2. 检查输入是否为 None
   3. 检查张量形状是否匹配
   4. 添加更多的输入验证
   
   #### 问题3：参数没有生效
   
   **症状**：修改参数后节点行为没有变化
   
   **解决方案**：
   1. 确保使用 `self.params.get()` 获取参数
   2. 检查参数名是否拼写正确
   3. 检查是否在 `build()` 中缓存了参数值
   
   ### 调试技巧
   
   #### 1. 使用调试节点
   
   在蓝图中插入调试节点查看中间结果：
   
   ```python
   @node(
       opcode="debug",
       label="调试输出",
       ports={"input": {"x": ""}, "output": {"out": ""}},
   )
   class DebugNode(BaseNode):
       def compute(self, input):
           x = input.get("x")
           print(f"调试输出：shape={x.shape}, dtype={x.dtype}")  # 打印形状和类型
           print(f"调试输出：min={x.min()}, max={x.max()}")  # 打印最小最大值
           return {"out": x}  # 透传输入
   ```
   
   #### 2. 添加日志
   
   ```python
   def compute(self, input):
       x = input.get("x")
       print(f"[{self.nodeId}] 输入形状: {x.shape}")  # 打印输入形状
       
       out = self.process(x)
       print(f"[{self.nodeId}] 输出形状: {out.shape}")  # 打印输出形状
       
       return {"out": out}
   ```
   
   ### 下一步学习
   
   1. **阅读现有节点**：查看 [`nodes/base.py`](nodes/base.py:1) 和 [`nodes/example.py`](nodes/example.py:1) 了解更多示例
   2. **创建自己的节点**：从简单的数学运算节点开始
   3. **测试节点**：在前端创建蓝图测试节点功能
   4. **优化性能**：使用 PyTorch 的性能分析工具
   5. **贡献代码**：将有用的节点分享给团队
   
   ### 参考资源
   
   - **PyTorch 文档**：https://pytorch.org/docs/
   - **项目核心文件**：
     - [`server.py`](server.py:1) - WebSocket 服务器
     - [`registry.py`](registry.py:1) - 节点注册表
     - [`engine.py`](engine.py:1) - 执行引擎
     - [`loader.py`](loader.py:1) - 动态加载器
     - [`sort.py`](sort.py:1) - 拓扑排序
   
   ---
   
   ## 🎯 快速开始检查清单
   
   作为新手程序员，在开始开发前请确认：
   
   - [ ] 已安装 Python 3.12+ 和 uv
   - [ ] 已成功启动服务并看到节点加载信息
   - [ ] 已阅读完整的 README 文档
   - [ ] 理解了项目的核心概念（节点、蓝图、端口、参数）
   - [ ] 查看了 [`nodes/base.py`](nodes/base.py:1) 和 [`nodes/example.py`](nodes/example.py:1) 的示例代码
   - [ ] 理解了代码风格规范（每行注释、驼峰命名、命令式写法）
   - [ ] 知道如何创建新节点（定义分类、使用装饰器、实现 compute）
   - [ ] 知道如何调试（查看日志、使用调试节点、添加打印语句）
   
   ## 📞 获取帮助
   
   如果遇到问题：
   
   1. **查看日志**：后端控制台会显示详细的错误信息
   2. **阅读代码**：所有代码都有详细注释，直接阅读源码
   3. **参考示例**：查看 [`nodes/example.py`](nodes/example.py:1) 的完整示例
   4. **检查规范**：确保遵循了代码风格规范
   
   ---
   
   **祝你开发顺利！记住：代码要像积木一样清晰，每行都要有注释，让后来者能轻松理解。** 🚀
   

