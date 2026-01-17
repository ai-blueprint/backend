"""
WebSocket客户端测试脚本

用于测试后端WebSocket服务器的功能

使用方法：
    1. 先启动服务器：uv run python main.py
    2. 再运行测试：uv run python test_ws_client.py
"""

import asyncio                                                                   # 导入异步IO库
import json                                                                      # 导入JSON处理库
import websockets                                                                # 导入WebSocket库


async def test_client():                                                         # 测试客户端主函数
    """ 测试WebSocket客户端 """                                                    # 函数文档字符串
    uri = "ws://localhost:8765"                                                  # 服务器地址
    
    print("=" * 50)                                                              # 打印分隔线
    print("     WebSocket 客户端测试")                                             # 打印标题
    print("=" * 50)                                                              # 打印分隔线
    
    try:                                                                         # 尝试连接服务器
        async with websockets.connect(uri) as websocket:                         # 建立WebSocket连接
            print(f"✅ 已连接到服务器：{uri}")                                       # 打印连接成功
            
            # 测试1：获取节点注册表
            print("\n📤 测试1：获取节点注册表")                                       # 打印测试信息
            await websocket.send(json.dumps({                                    # 发送请求
                "type": "get_registry",                                          # 请求类型
                "id": "test-1"                                                   # 请求ID
            }))
            
            response = await websocket.recv()                                    # 接收响应
            data = json.loads(response)                                          # 解析JSON
            print(f"📥 收到响应：type={data['type']}")                               # 打印响应类型
            
            if data['type'] == 'registry':                                       # 如果成功获取注册表
                categories = data['data'].get('categories', {})                  # 获取分类
                nodes = data['data'].get('nodes', {})                            # 获取节点
                print(f"   分类数量：{len(categories)}")                            # 打印分类数
                print(f"   节点数量：{len(nodes)}")                                 # 打印节点数
                print(f"   节点列表：{list(nodes.keys())}")                          # 打印节点名
            
            # 测试2：运行蓝图
            print("\n📤 测试2：运行蓝图")                                            # 打印测试信息
            
            # 构建简单的测试蓝图
            test_blueprint = {                                                   # 测试蓝图数据
                "nodes": [                                                       # 节点列表
                    {
                        "id": "node-1",
                        "type": "baseNode",
                        "data": {
                            "nodeKey": "input",                                  # 输入节点
                            "params": {}
                        }
                    },
                    {
                        "id": "node-2",
                        "type": "baseNode",
                        "data": {
                            "nodeKey": "sum",                                    # 求和节点
                            "params": {
                                "dim": {"type": "number", "default": 1},
                                "keepdim": {"type": "boolean", "default": True}
                            }
                        }
                    }
                ],
                "edges": [                                                       # 连线列表
                    {
                        "source": "node-1",
                        "sourceHandle": "out",
                        "target": "node-2",
                        "targetHandle": "x"
                    }
                ]
            }
            
            # 准备输入数据（3x4矩阵）
            test_inputs = {                                                      # 测试输入数据
                "node-1": {
                    "out": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]          # 3x4矩阵
                }
            }
            
            await websocket.send(json.dumps({                                    # 发送运行蓝图请求
                "type": "run_blueprint",                                         # 请求类型
                "id": "test-2",                                                  # 请求ID
                "data": {                                                        # 请求数据
                    "blueprint": test_blueprint,                                 # 蓝图数据
                    "inputs": test_inputs                                        # 输入数据
                }
            }))
            
            # 接收所有节点执行结果
            while True:                                                          # 持续接收消息
                response = await websocket.recv()                                # 接收响应
                data = json.loads(response)                                      # 解析JSON
                
                if data['type'] == 'node_result':                                # 如果是节点结果
                    node_id = data['data']['nodeId']                             # 获取节点ID
                    output = data['data']['output']                              # 获取输出
                    print(f"📥 节点执行完成：{node_id}")                              # 打印节点ID
                    if output:                                                   # 如果有输出
                        for port, val in output.items():                         # 遍历每个端口
                            if isinstance(val, dict) and val.get('type') == 'tensor':
                                print(f"   {port}: shape={val['shape']}")        # 打印张量形状
                            else:
                                print(f"   {port}: {val}")                       # 打印其他值
                
                elif data['type'] == 'execution_complete':                       # 如果执行完成
                    print(f"\n✅ 蓝图执行完成！")                                     # 打印完成信息
                    print(f"   成功: {data['data']['success']}")                   # 打印成功状态
                    break                                                        # 退出循环
                
                elif data['type'] == 'error':                                    # 如果出错
                    print(f"\n❌ 执行出错：{data['data']['message']}")                # 打印错误信息
                    break                                                        # 退出循环
            
            print("\n" + "=" * 50)                                               # 打印分隔线
            print("     测试完成")                                                 # 打印完成信息
            print("=" * 50)                                                      # 打印分隔线
            
    except ConnectionRefusedError:                                               # 连接被拒绝
        print("❌ 连接失败：请确保服务器已启动 (uv run python main.py)")               # 打印错误提示


if __name__ == "__main__":                                                       # 主程序入口
    asyncio.run(test_client())                                                   # 运行测试客户端
