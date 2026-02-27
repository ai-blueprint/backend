import asyncio  # 异步库，用于合并短时间内的多次文件变更
import copy  # 拷贝库，用于生成可回滚的深拷贝快照
import os  # 操作系统库，用于拼接目录路径

import loader  # 复用已有节点重载能力
import registry  # 复用已有注册表能力


async def runHotReload(broadcast, folder="nodes"):
    from watchfiles import awatch  # 延迟导入，避免不启用时增加启动负担

    nodesDir = os.path.join(os.path.dirname(__file__), folder)  # 计算需要监听的节点目录
    print(f"开始监控节点目录: {nodesDir}")  # 输出监听状态，方便排查

    async for changes in awatch(nodesDir):  # 持续监听目录变化事件
        pyChanges = [item for item in changes if str(item[1]).endswith(".py")]  # 仅处理python文件变化
        if not pyChanges:
            continue  # 非python改动直接忽略

        print(f"检测到节点文件变化: {pyChanges}")  # 输出变化详情
        await asyncio.sleep(0.1)  # 简单防抖，合并极短时间内的多次保存
        nodesSnapshot, categoriesSnapshot = (
            copy.deepcopy(registry.nodes),
            copy.deepcopy(registry.categories),
        )  # 先做深拷贝快照，保证重载失败可以回滚

        try:
            loader.reloadAll(folder)  # 全量重建注册表，复用已有重载逻辑
            print("热重载完成，广播新registry")  # 输出成功日志
            await broadcast("registryUpdated", registry.getAllForFrontend())  # 广播更新后的注册表给前端
        except Exception as error:
            print(f"热重载失败，回滚: {error}")  # 输出失败日志
            registry.nodes.clear()  # 清空当前节点注册表
            registry.nodes.update(nodesSnapshot)  # 回滚节点快照
            registry.categories.clear()  # 清空当前分类注册表
            registry.categories.update(categoriesSnapshot)  # 回滚分类快照
            await broadcast("reloadError", {"error": str(error)})  # 广播热重载失败信息


def mountHotReload(loop, broadcast, folder="nodes"):
    return loop.create_task(runHotReload(broadcast, folder))  # 挂载后台任务，导入并调用即可启用
