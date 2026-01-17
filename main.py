"""
炼丹蓝图后端主程序入口

功能：启动WebSocket服务器，等待前端连接

启动方式：
    uv run python main.py
    或
    uv run python main.py --host 0.0.0.0 --port 8765
"""

import argparse                                                                  # 导入命令行参数解析库
from ws_server import run_server                                                 # 导入WebSocket服务器启动函数


def main():                                                                      # 主函数定义
    """ 解析命令行参数并启动服务器 """                                               # 函数文档字符串
    parser = argparse.ArgumentParser(description="炼丹蓝图后端服务器")               # 创建参数解析器
    parser.add_argument("--host", default="localhost", help="监听地址")            # 添加host参数
    parser.add_argument("--port", type=int, default=8765, help="监听端口")         # 添加port参数
    
    args = parser.parse_args()                                                   # 解析命令行参数
    
    print("=" * 50)                                                              # 打印分隔线
    print("     炼丹蓝图 - 后端服务器")                                             # 打印标题
    print("=" * 50)                                                              # 打印分隔线
    print()                                                                      # 空行
    
    run_server(host=args.host, port=args.port)                                   # 启动WebSocket服务器


if __name__ == "__main__":                                                       # 主程序入口判断
    main()                                                                       # 调用主函数
