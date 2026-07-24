"""
plugins.py - 可信本地插件发现和重载指令

插件必须位于 backend/plugins/<pluginID>/，包含 manifest.json 和清单指定的 Python 入口。
加载时注册表记录插件所有权；任何节点或分类碰撞都会回滚整个注册表快照。
"""

import copy  # 注册表快照能力，用于失败时完整回滚
import importlib.util  # 文件导入能力，用于执行可信本地插件入口
import json  # 清单解析能力，用于读取插件身份和入口
import re  # 标识校验能力，用于限制插件 ID 和入口路径
import sys  # 模块缓存能力，用于开发重载时替换旧插件模块
from pathlib import Path  # 路径能力，用于约束插件只能来自可信根目录

import registry  # 节点注册数据，插件导入时写入并标注所有权


pluginsRoot = Path(__file__).resolve().parent / "plugins"  # 唯一可信插件根目录
pluginStatuses = {}  # 保存最近一次发现和加载反馈供协议查询
pluginModuleNames = set()  # 记录动态模块名，重载前从缓存中移除


# --- 读取并校验插件清单 ---
def readManifest(pluginPath):
    manifestPath = pluginPath / "manifest.json"  # 每个一级插件目录必须提供固定名称清单
    if not manifestPath.is_file():
        raise ValueError("缺少manifest.json")  # 无清单目录不具备明确身份和入口
    manifest = json.loads(manifestPath.read_text(encoding="utf-8"))  # 只读取本地可信 JSON 文件
    pluginID = manifest.get("id", "")  # 清单 ID 同时作为注册表所有者
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pluginID) or pluginID != pluginPath.name:
        raise ValueError("插件id必须与目录名一致且只包含字母、数字、_或-")  # 阻止模块名和路径注入
    entry = manifest.get("entry", "plugin.py")  # 简单插件默认使用 plugin.py 入口
    entryPath = (pluginPath / entry).resolve()  # 解析入口中的相对路径片段
    if pluginPath.resolve() not in entryPath.parents or entryPath.suffix != ".py" or not entryPath.is_file():
        raise ValueError("插件entry必须是插件目录内存在的.py文件")  # 入口不能越出插件自己的目录
    return manifest, entryPath  # 返回已验证清单和绝对入口


# --- 发现可信根目录中的插件 ---
def discoverPlugins():
    pluginsRoot.mkdir(parents=True, exist_ok=True)  # 首次运行创建固定可信根目录
    discovered = []  # 返回稳定排序的插件状态列表
    for pluginPath in sorted(path for path in pluginsRoot.iterdir() if path.is_dir()):
        try:
            manifest, entryPath = readManifest(pluginPath)  # 读取完整身份和入口
            discovered.append({"id": manifest["id"], "name": manifest.get("name", manifest["id"]), "version": manifest.get("version", "0.0.0"), "enabled": bool(manifest.get("enabled", True)), "entry": str(entryPath.relative_to(pluginPath)), "status": pluginStatuses.get(manifest["id"], {}).get("status", "discovered")})  # 仅暴露相对入口
        except Exception as error:
            discovered.append({"id": pluginPath.name, "enabled": False, "status": "error", "error": str(error)})  # 无效目录仍可见但绝不执行
    return discovered  # 查询动作不修改注册表


# --- 导入一个插件并登记所有权 ---
def loadPlugin(pluginID):
    pluginPath = (pluginsRoot / pluginID).resolve()  # 插件 ID 先转换为可信根目录下路径
    if pluginsRoot.resolve() not in pluginPath.parents:
        raise ValueError("插件路径不合法")  # 防止 API 直接请求目录穿越
    manifest, entryPath = readManifest(pluginPath)  # 清单再次验证目录名和入口
    if not manifest.get("enabled", True):
        pluginStatuses[pluginID] = {"status": "disabled"}  # 禁用插件不执行任何代码
        return pluginStatuses[pluginID]

    moduleName = f"blueprint_plugin_{pluginID}"  # 稳定模块名便于开发重载替换缓存
    sys.modules.pop(moduleName, None)  # 重载前移除旧模块对象
    specification = importlib.util.spec_from_file_location(moduleName, entryPath)  # 从已验证入口构建导入规范
    module = importlib.util.module_from_spec(specification)  # 创建独立插件模块对象
    registry.setRegistrationOwner(f"plugin:{pluginID}")  # 导入期间所有注册项都标记插件所有权
    try:
        sys.modules[moduleName] = module  # 提前写入缓存支持插件内部相对自引用
        specification.loader.exec_module(module)  # 执行用户明确安装的可信本地插件代码
    finally:
        registry.setRegistrationOwner("core")  # 无论成功失败都恢复内置注册身份
    pluginModuleNames.add(moduleName)  # 成功执行后纳入后续重载清理范围
    ownedNodes = sorted(opcode for opcode, owner in registry.nodeOwners.items() if owner == f"plugin:{pluginID}")  # 汇总插件实际注册节点
    pluginStatuses[pluginID] = {"status": "loaded", "nodes": ownedNodes}  # 记录明确加载反馈
    return pluginStatuses[pluginID]


# --- 重载全部启用插件并保护注册表一致性 ---
def reloadPlugins():
    with registry.registryLock:
        registrySnapshot = (copy.copy(registry.nodes), copy.deepcopy(registry.categories), copy.copy(registry.nodeOwners), copy.copy(registry.categoryOwners))  # 插件失败时恢复导入前完整状态
        statusSnapshot = copy.deepcopy(pluginStatuses)  # 可见状态必须和注册表一起回滚
        pluginStatuses.clear()  # 新一轮结果替换旧状态
        try:
            pluginOpcodes = {opcode for opcode, owner in registry.nodeOwners.items() if owner.startswith("plugin:")}  # 找出上一轮全部插件节点
            pluginCategories = {categoryID for categoryID, owner in registry.categoryOwners.items() if owner.startswith("plugin:")}  # 找出上一轮全部插件分类
            for opcode in pluginOpcodes:
                registry.nodes.pop(opcode, None); registry.nodeOwners.pop(opcode, None)  # 移除旧插件节点后再按当前清单重建
            for categoryID in pluginCategories:
                registry.categories.pop(categoryID, None); registry.categoryOwners.pop(categoryID, None)  # 移除旧插件分类避免禁用插件残留
            for categoryData in registry.categories.values():
                categoryData["nodes"] = [opcode for opcode in categoryData.get("nodes", []) if opcode not in pluginOpcodes]  # 清理插件挂到共享分类的旧引用
            for plugin in discoverPlugins():
                if plugin.get("status") == "error":
                    pluginStatuses[plugin["id"]] = {"status": "error", "error": plugin["error"]}  # 无效清单保持隔离并继续其他插件
                    continue
                if plugin.get("enabled"):
                    loadPlugin(plugin["id"])  # 启用插件按目录名稳定顺序导入
                else:
                    pluginStatuses[plugin["id"]] = {"status": "disabled"}  # 禁用状态进入查询反馈
        except Exception as error:
            registry.nodes.clear(); registry.nodes.update(registrySnapshot[0])  # 回滚节点实现和定义
            registry.categories.clear(); registry.categories.update(registrySnapshot[1])  # 回滚分类和节点顺序
            registry.nodeOwners.clear(); registry.nodeOwners.update(registrySnapshot[2])  # 回滚节点所有权
            registry.categoryOwners.clear(); registry.categoryOwners.update(registrySnapshot[3])  # 回滚分类所有权
            pluginStatuses.clear(); pluginStatuses.update(statusSnapshot)  # 回滚插件状态避免显示虚假loaded
            raise ValueError(f"插件重载失败并已回滚: {error}") from error
    return {"status": "complete", "plugins": discoverPlugins()}  # 返回重载后的完整状态
