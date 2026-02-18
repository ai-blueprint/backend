"""
uv run python main.py 启动整个项目
"""

import server  # 导入服务器模块，里面有启动服务的函数

if __name__ == "__main__":  # 直接运行此文件时启动服务
    server.start()  # 调用server.start()启动WebSocket服务，默认localhost:8765
