"""
炼丹蓝图后端主程序入口
"""

import argparse
import asyncio
import server  # 调用 server.start()


def main():  # 启动服务
    """解析命令行参数并启动服务器"""
    parser = argparse.ArgumentParser(description="炼丹蓝图后端服务器")
    parser.add_argument("--host", default="localhost", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")

    args = parser.parse_args()

    print("=" * 50)
    print("     炼丹蓝图 - 后端服务器")
    print("=" * 50)
    print()

    asyncio.run(server.start(host=args.host, port=args.port))  # 调用 server.start(host: str = "localhost", port: int = 8765)


if __name__ == "__main__":
    main()
