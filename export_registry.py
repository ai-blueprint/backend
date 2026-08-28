"""
export_registry.py - 前端离线注册表同步指令

后端节点注册表是唯一数据源，前端registry.json只是离线兜底副本。
每次新增或修改节点后运行本脚本，把最新注册表写入前端常量目录。
用法：uv run python export_registry.py
"""

import json  # 注册表编码能力，用于生成前端可直接导入的JSON文件
from pathlib import Path  # 路径能力，用于定位前端常量目录

import engine  # 导入执行引擎以触发全部内置节点注册  # noqa: F401
import registry  # 节点注册表能力，提供前端所需的完整定义


frontendRegistryPath = Path(__file__).resolve().parent.parent / "frontend" / "src" / "constants" / "registry.json"  # 前端离线注册表的唯一位置


# --- 生成前端注册表文件文本 ---
def getRegistryFileText():
    """
    用法：text = getRegistryFileText()  # 同步脚本和一致性测试共用同一份格式化结果
    """
    return json.dumps(registry.getAllForFrontend(), ensure_ascii=False, indent="\t") + "\n"  # 制表符缩进和结尾换行保持文件格式稳定


# --- 把最新注册表写入前端 ---
def exportRegistry():
    """
    用法：exportRegistry()  # 覆盖写入frontend/src/constants/registry.json
    """
    frontendRegistryPath.write_text(getRegistryFileText(), encoding="utf-8")  # 覆盖旧副本，前端离线数据与后端保持一致
    nodeCount = len(registry.getAllForFrontend()["nodes"])  # 统计本次导出的可见节点数量
    print(f"已同步前端注册表: {frontendRegistryPath}（{nodeCount}个节点）")  # 反馈导出位置和规模


if __name__ == "__main__":  # 直接运行此文件时执行同步
    exportRegistry()  # 导出最新注册表到前端
