import torch
from registry import Registry

class BlueprintEngine:
    def __init__(self, blueprint_data):
        """
        blueprint_data 格式:
        {
            "nodes": [
                {"id": "n1", "type": "linear", "params": {"输出特征数": 128}},
                ...
            ],
            "edges": [
                {"source": "n1", "sourceHandle": "y", "target": "n2", "targetHandle": "x"}
            ]
        }
        """
        self.registry = Registry()
        self.registry.load_nodes()
        
        self.nodes_data = {n['id']: n for n in blueprint_data.get('nodes', [])}
        self.edges = blueprint_data.get('edges', [])
        
        self.node_layers = {} # 存储 build 后的 pytorch 层
        self.node_funcs = {}  # 存储 (infer, build, compute) 元组
        self.adjacency = {}
        self._build_graph()

    def _build_graph(self):
        for edge in self.edges:
            src = edge['source']
            dst = edge['target']
            if src not in self.adjacency:
                self.adjacency[src] = []
            self.adjacency[src].append(edge)

    def execute(self, initial_inputs: dict):
        """
        initial_inputs: {"node_id": {"port_id": tensor}}
        """
        in_degree = {node_id: 0 for node_id in self.nodes_data}
        for edge in self.edges:
            in_degree[edge['target']] += 1
        
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        results = {} # {node_id: {port_id: tensor}}

        # 预加载所有节点的函数
        for node_id, node_info in self.nodes_data.items():
            opcode = node_info['type']
            node_def = self.registry.get_function(opcode)
            if node_def and 'func' in node_def:
                self.node_funcs[node_id] = node_def['func']()

        while queue:
            curr_id = queue.pop(0)
            node_info = self.nodes_data[curr_id]
            params = node_info.get('params', {})
            
            if curr_id not in self.node_funcs:
                print(f"Warning: Node type {node_info['type']} not found")
                continue

            infer, build, compute = self.node_funcs[curr_id]

            # 准备输入数据
            curr_node_inputs = {}
            if curr_id in initial_inputs:
                curr_node_inputs.update(initial_inputs[curr_id])
            
            # 从前驱节点获取输入
            for edge in self.edges:
                if edge['target'] == curr_id:
                    src_id = edge['source']
                    src_handle = edge['sourceHandle']
                    target_handle = edge['targetHandle']
                    if src_id in results and src_handle in results[src_id]:
                        curr_node_inputs[target_handle] = results[src_id][src_handle]

            # 推断形状并构建层（如果是第一次执行）
            if curr_id not in self.node_layers:
                input_shapes = {k: list(v.shape) if hasattr(v, 'shape') else v for k, v in curr_node_inputs.items()}
                # 适配不同的 build 签名 (有些节点 build 只收 params)
                import inspect
                sig = inspect.signature(build)
                if len(sig.parameters) == 2:
                    self.node_layers[curr_id] = build(input_shapes, params)
                else:
                    self.node_layers[curr_id] = build(params)

            # 执行计算
            # 适配 compute 签名
            # 如果是 input 节点且 compute 返回 None，我们尝试从 initial_inputs 获取数据
            # 适配 compute 签名：有些节点期望 dict，有些期望单个 tensor
            # 检查 compute 的第一个参数名，如果是 'inputs' 且 node_def 里的 ports['in'] 只有一个，
            # 或者根据 example.py 的写法，它直接用了 inputs
            # 实际上 example.py 里的 linear_node compute(inputs, layer) 内部用了 layer(inputs)
            # 而 torch.nn.Linear 期望 tensor。
            # 这是一个不一致的地方，我们需要统一。
            # 按照 nodes/example.py 的逻辑，compute 接收的是 inputs (dict)
            # 但 linear_node 内部实现可能有误：outputs = layer(inputs) 应该 outputs = layer(inputs['x'])
            
            # 为了兼容用户提供的 nodes/example.py，我们在这里做一点 hack
            # 如果 inputs 只有一个 key，且 layer 是 nn.Module，我们尝试传入该 key 的值
            if isinstance(self.node_layers[curr_id], torch.nn.Module) and len(curr_node_inputs) == 1:
                val = list(curr_node_inputs.values())[0]
                output = compute(val, self.node_layers[curr_id])
            else:
                output = compute(curr_node_inputs, self.node_layers[curr_id])
            
            if output is None and curr_id in initial_inputs:
                output = initial_inputs[curr_id]

            # 包装输出为字典（如果不是字典）
            if not isinstance(output, dict) and output is not None:
                # 默认取第一个输出端口名
                opcode = node_info['type']
                node_def = self.registry.get_function(opcode)
                out_ports = node_def.get('ports', {}).get('out', ['out'])
                output = {out_ports[0]: output}
            
            results[curr_id] = output

            # 更新后继节点
            for edge in self.adjacency.get(curr_id, []):
                neighbor = edge['target']
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return results

if __name__ == "__main__":
    # 测试重构后的引擎
    test_blueprint = {
        "nodes": [
            {"id": "input_1", "type": "input", "params": {"输出维度": [1, 10]}},
            {"id": "linear_1", "type": "linear", "params": {"输出特征数": 5, "bias": True}},
            {"id": "output_1", "type": "output", "params": {}}
        ],
        "edges": [
            {"source": "input_1", "sourceHandle": "out", "target": "linear_1", "targetHandle": "x"},
            {"source": "linear_1", "sourceHandle": "y", "target": "output_1", "targetHandle": "in"}
        ]
    }
    
    engine = BlueprintEngine(test_blueprint)
    # 模拟输入数据
    # 注意：input 节点的 compute 返回 None，但我们需要它能传递数据
    # 实际中 input 节点应该负责把 params 里的维度转为 tensor 或者接收外部输入
    initial_data = {
        "input_1": {"out": torch.randn(1, 10)}
    }
    all_results = engine.execute(initial_data)
    for node_id, res in all_results.items():
        if res is None:
            print(f"Node {node_id} Output: None")
            continue
        for port, val in res.items():
            if hasattr(val, 'shape'):
                print(f"Node {node_id} Port {port} Output Shape: {val.shape}")
            else:
                print(f"Node {node_id} Port {port} Output: {val}")
