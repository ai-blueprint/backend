# 炼丹蓝图·后端 (AI Blueprint Backend)

![Status](https://img.shields.io/badge/Status-Early_Dev-orange) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![License](https://img.shields.io/badge/License-AGPL_v3-green)

> **炼丹蓝图**是一个可视化的 AI 模型构建平台，让复杂的神经网络搭建变得像搭积木一样简单直观。本项目为其后端服务，负责蓝图的解析、调度与执行。

---

## 🚧 当前状态

**【早期开发阶段】**
- ✅ 核心蓝图执行引擎已跑通，支持拓扑排序与数据流转。
- ✅ WebSocket 通信协议已定型，支持前后端实时交互。
- ✅ 基础节点架构已确立，支持动态注册。
- ⚠️ 目前内置节点种类较少，许多 AI 模型组件尚未实现。
- 🤝 **我们非常欢迎社区贡献更多节点和功能！**

---

## ✨ 功能亮点

🔍 **可视化蓝图执行引擎**
内置高效的图执行引擎 (`engine.py`)，能够自动解析前端传递的节点图，通过拓扑排序确定执行顺序，并处理复杂的节点间数据依赖。

🔌 **WebSocket 实时交互**
基于 `websockets` 库构建的异步服务器 (`server.py`)，支持前端实时获取节点注册表 (`getRegistry`)、发送运行请求 (`runBlueprint`) 以及接收实时的执行结果与错误反馈。

🧩 **模块化节点系统**
高度可扩展的节点注册机制 (`registry.py`)。开发者只需使用简单的 `@node` 和 `@category` 装饰器，即可快速将 PyTorch 代码封装为可视化节点，无需侵入核心逻辑。

🔥 **原生 PyTorch 支持**
底层无缝集成 PyTorch，所有节点本质上都是 `nn.Module`，确保了与主流深度学习生态的完美兼容性和高性能。

---

## 🚀 快速开始

### 前置要求
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (推荐) 或 pip

### 安装与运行

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-username/alchemy-blueprint-backend.git
   cd alchemy-blueprint-backend
   ```

2. **安装依赖**
   本项目使用 `uv` 进行包管理，推荐直接使用 `uv` 运行：
   ```bash
   # 如果没有安装 uv
   pip install uv
   ```

3. **启动服务**
   ```bash
   uv run python main.py
   ```
   *或者使用传统 pip 方式：*
   ```bash
   pip install -r requirements.txt  # 需自行导出
   python main.py
   ```

4. **服务状态**
   启动成功后，控制台将显示：
   ```
   WebSocket服务已启动: ws://localhost:8765
   ```

---

## 📜 许可证

本项目采用 **GNU AGPL v3** 许可证。

- ✅ **欢迎贡献**：您可以自由地修改和分发本项目的代码。
- 📢 **源码公开义务**：如果您将本项目用于网络服务（如 SaaS），您必须向用户公开您的修改源码。
- 💼 **商业授权**：如需闭源商业使用或有其他特殊授权需求，请联系我们。

---

## 🤝 贡献方式

我们强烈鼓励通过 **Pull Request** 向本仓库贡献代码，共同完善这个开源项目，而不是创建封闭的商业 fork。

---

## 📬 联系方式 / 社区

- **QQ 交流群**：[1081197052](https://qun.qq.com/universal-share/share?ac=1&authKey=eLkj1QLdUkC2LQAiLKW2tmH87UgnLxrp22jPc0q4vlCPVy84SOSYOR3coq8pNZuB&busi_data=eyJncm91cENvZGUiOiIxMDgxMTk3MDUyIiwidG9rZW4iOiJVbzB3dmJQNVl0cnozaFpKQmYycStPa2k3TEd2ZytIVTRENklkSHptcHhPU1JHK203QzgyNHhOcE9KSldhd1Q2IiwidWluIjoiOTE3ODExNzI2In0%3D&data=zyq7dImFnIpdAx5x2Zs8oKjKC8DAvkepKvOczDGKaOPHRi7YJGAcrwXq-3upjpICMZ1hK13zJ1UT9bzdTO8WpA&svctype=4&tempid=h5_group_info)
