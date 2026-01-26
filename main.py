"""
uv run python main.py 启动整个项目
"""

import server  # 导入服务器模块，里面有启动服务的函数

server.start()  # 调用server.start()启动WebSocket服务，默认localhost:8765
