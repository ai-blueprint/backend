"""
main.py - 启动服务入口

用法：
    python main.py
    
示例：
    直接运行即可启动WebSocket服务
"""

import server  # 导入服务器模块，里面有启动服务的函数

server.start()  # 调用server.start()启动WebSocket服务，默认localhost:8765
