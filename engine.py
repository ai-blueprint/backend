import torch                                                                    # 导入 PyTorch 库，用于张量计算
from registry import Registry                                                   # 导入节点注册表，用于获取算子定义
from utils import extract_single_input                                          # 导入工具函数

class BlueprintEngine:                                                          # 定义蓝图执行引擎类
    """ 蓝图执行引擎：负责解析蓝图并按顺序执行算子 """                                # 类文档字符串

    def __init__(self, blueprint_data):                                         # 构造函数，初始化引擎
        self.nodes_data = {n['id']: n for n in blueprint_data.get('nodes', [])} # 将节点列表转为字典，方便通过 ID 快速查找
        self.edges = blueprint_data.get('edges', [])                            # 获取蓝图中的所有连线（边）
        self.registry = Registry()                                              # 实例化注册表对象
        self.registry.load_nodes()                                              # 加载所有可用的节点定义到注册表
        self.node_layers = {}                                                   # 缓存：存储每个节点实例化的 PyTorch 层
        self.node_funcs = {}                                                    # 缓存：存储每个节点的 (infer, build, compute) 函数

    def execute(self, initial_inputs):                                          # 执行蓝图的主入口方法
        """ 执行蓝图并返回结果 """                                                 # 方法文档字符串
        execution_order = self._get_execution_order()                           # 第一步：计算节点的执行顺序（拓扑排序）
        results = {}                                                            # 存储每个节点的输出结果，供后续节点使用

        for node_id in execution_order:                                         # 第二步：按计算出的顺序遍历每个节点
            results[node_id] = self._execute_single_node(node_id, results, initial_inputs) # 执行单个节点并保存结果

        return results                                                          # 返回最终的所有节点计算结果

    async def execute_with_callback(self, initial_inputs, on_node_complete):    # 带回调的异步执行方法
        """ 执行蓝图，每个节点执行完成后调用回调函数 """                                 # 方法文档字符串
        execution_order = self._get_execution_order()                           # 第一步：计算节点的执行顺序（拓扑排序）
        results = {}                                                            # 存储每个节点的输出结果，供后续节点使用

        for node_id in execution_order:                                         # 第二步：按计算出的顺序遍历每个节点
            results[node_id] = self._execute_single_node(node_id, results, initial_inputs) # 执行单个节点并保存结果
            if on_node_complete:                                                # 如果提供了回调函数
                await on_node_complete(node_id, results[node_id])               # 调用回调，传递节点ID和执行结果

        return results                                                          # 返回最终的所有节点计算结果

    def _execute_single_node(self, node_id, results, initial_inputs):           # 执行单个节点的内部逻辑
        node_info = self.nodes_data[node_id]                                    # 获取当前节点的配置信息
        node_type = self._get_node_type(node_info)                              # 获取节点类型（从 data.nodeKey 或 type）
        params = self._extract_params(node_info)                                # 提取节点参数配置

        inputs = self._collect_inputs(node_id, results)                         # 第三步：从已有的结果中收集当前节点的输入
        funcs = self._get_node_functions(node_id, node_type)                    # 第四步：获取该节点类型的处理函数集
        
        if funcs is None:                                                       # 如果找不到该算子的定义
            return None                                                         # 直接返回空结果

        infer, build, compute = funcs                                           # 解构出推断、构建、计算三个核心函数
        layer = self._get_or_build_layer(node_id, build, inputs, params)        # 第五步：获取或创建该节点的 PyTorch 层实例

        output = self._run_compute(compute, layer, inputs, node_type)           # 第六步：执行实际的张量计算逻辑（传入节点类型）
        output = self._handle_special_nodes(node_id, output, initial_inputs)    # 第七步：处理特殊节点（如输入节点透传数据）
        output = self._ensure_dict_output(node_id, node_type, output)           # 第八步：确保输出格式统一为字典，方便后续查找

        return output                                                           # 返回该节点的最终计算结果

    def _run_compute(self, compute_func, layer, inputs, node_type):             # 执行计算的辅助方法（引擎统一处理输入格式）
        """
        引擎统一负责输入格式转换，节点只需专注于计算逻辑：
        1. 无输入节点：直接调用 compute(None, layer)
        2. 单输入 + nn.Module：引擎直接调用 layer(x)
        3. 单输入 + 非 nn.Module：引擎解包为张量后传给 compute(x, layer)
        4. 多输入：引擎传入完整字典，compute(inputs, layer) 自行处理
        """
        # 获取节点的输入端口定义
        node_def = self.registry.get_function(node_type)                        # 获取节点定义
        input_ports = node_def.get('ports', {}).get('in', []) if node_def else []  # 获取输入端口列表
        
        # 无输入节点（如 input 节点）：直接调用 compute
        if len(input_ports) == 0:                                               # 如果是无输入节点
            return compute_func(None, layer)                                    # 直接调用 compute
        
        # 单输入情况：使用工具函数智能解包
        if len(input_ports) == 1:                                               # 如果是单输入节点
            first_port = input_ports[0]                                         # 获取端口名
            x = extract_single_input(inputs, first_port)                        # 使用工具函数安全提取张量
            
            if x is None:                                                       # 如果输入为空（连接缺失）
                return None                                                     # 返回 None，跳过计算
            
            # 对于 nn.Module 类型的层，引擎直接调用层计算（节点无需关心）
            if isinstance(layer, torch.nn.Module):                              # 如果是标准 PyTorch 层
                return layer(x)                                                 # 直接调用层，返回结果
            
            # 对于非 nn.Module（如纯函数计算），传入张量给 compute
            return compute_func(x, layer)                                       # 传入张量和层对象
        
        # 多输入情况：传入完整字典，由 compute 函数处理
        return compute_func(inputs, layer)                                      # 传入字典进行多输入计算

    def _get_execution_order(self):                                             # 计算执行顺序的方法（拓扑排序）
        in_degree = {node_id: 0 for node_id in self.nodes_data}                 # 初始化所有节点的入度为 0
        adj = {node_id: [] for node_id in self.nodes_data}                      # 初始化邻接表，记录节点间的指向关系

        for edge in self.edges:                                                 # 遍历蓝图中的所有连线
            src, dst = edge['source'], edge['target']                           # 获取连线的起点和终点 ID
            adj[src].append(dst)                                                # 在邻接表中添加指向关系
            in_degree[dst] += 1                                                 # 终点节点的入度加 1

        queue = [n for n, d in in_degree.items() if d == 0]                     # 找到所有入度为 0 的节点作为起始执行点
        order = []                                                              # 存储排序后的节点 ID 序列

        while queue:                                                            # 当队列不为空时持续处理
            curr = queue.pop(0)                                                 # 从队列头部取出一个节点
            order.append(curr)                                                  # 将其加入执行序列
            for neighbor in adj[curr]:                                          # 遍历该节点指向的所有邻居
                in_degree[neighbor] -= 1                                        # 邻居节点的入度减 1
                if in_degree[neighbor] == 0:                                    # 如果邻居的入度变为 0
                    queue.append(neighbor)                                      # 将其加入待处理队列

        return order                                                            # 返回计算出的线性执行顺序

    def _collect_inputs(self, node_id, results):                                # 收集输入数据的方法
        node_inputs = {}                                                        # 存储收集到的输入字典
        for edge in self.edges:                                                 # 遍历所有连线
            if edge['target'] == node_id:                                       # 如果连线的终点是当前节点
                src_id = edge['source']                                         # 获取起点节点 ID
                src_port = edge['sourceHandle']                                 # 获取起点输出端口名
                dst_port = edge['targetHandle']                                 # 获取当前节点输入端口名
                if src_id in results and results[src_id]:                       # 如果起点已经产生了计算结果
                    node_inputs[dst_port] = results[src_id].get(src_port)       # 将起点的输出值映射到当前节点的输入端口
        return node_inputs                                                      # 返回收集完成的输入字典

    def _get_node_functions(self, node_id, node_type):                          # 获取算子函数的方法
        if node_id not in self.node_funcs:                                      # 如果缓存中没有该节点的函数
            node_def = self.registry.get_function(node_type)                    # 从注册表中查找该类型的定义
            if not node_def:                                                    # 如果找不到定义
                return None                                                     # 返回空
            self.node_funcs[node_id] = node_def['func']()                       # 执行定义函数，获取 (infer, build, compute) 元组
        return self.node_funcs[node_id]                                         # 返回缓存或新获取的函数元组

    def _get_or_build_layer(self, node_id, build_func, inputs, params):         # 获取或构建层的方法
        if node_id in self.node_layers:                                         # 如果该节点已经实例化过层
            return self.node_layers[node_id]                                    # 直接返回缓存的层实例

        input_shapes = {k: list(v.shape) if hasattr(v, 'shape') else v for k, v in inputs.items()} # 计算输入的形状信息
        import inspect                                                          # 导入检查模块，用于分析函数签名
        sig = inspect.signature(build_func)                                     # 获取 build 函数的参数签名
        
        if len(sig.parameters) == 2:                                            # 如果 build 函数需要两个参数 (shapes, params)
            self.node_layers[node_id] = build_func(input_shapes, params)        # 传入形状和参数进行构建
        else:                                                                   # 如果只需要一个参数 (params)
            self.node_layers[node_id] = build_func(params)                      # 只传入参数进行构建
            
        return self.node_layers[node_id]                                        # 返回新创建的层实例

    def _handle_special_nodes(self, node_id, output, initial_inputs):            # 处理特殊节点的方法
        if output is None and node_id in initial_inputs:                        # 如果计算结果为空且存在初始输入（如 input 节点）
            return initial_inputs[node_id]                                      # 直接透传初始输入数据
        return output                                                           # 否则返回正常的计算结果

    def _ensure_dict_output(self, node_id, node_type, output):                  # 统一输出格式的方法
        if output is None or isinstance(output, dict):                          # 如果输出为空或已经是字典格式
            return output                                                       # 直接返回
            
        node_def = self.registry.get_function(node_type)                        # 获取节点定义信息
        out_ports = node_def.get('ports', {}).get('out', ['out'])               # 获取定义的输出端口列表，默认为 ['out']
        return {out_ports[0]: output}                                           # 将单值结果包装成以第一个端口名为键的字典

    def _get_node_type(self, node_info):                                        # 获取节点真实类型的方法（适配新蓝图格式）
        """ 从节点信息中提取真实的 opcode，优先使用 data.nodeKey """               # 方法文档字符串
        data = node_info.get('data', {})                                        # 获取节点的 data 字段
        node_key = data.get('nodeKey')                                          # 尝试获取 nodeKey（新格式）
        if node_key:                                                            # 如果存在 nodeKey
            return node_key                                                     # 返回 nodeKey 作为节点类型
        return node_info.get('type')                                            # 否则回退到旧格式的 type 字段

    def _extract_params(self, node_info):                                       # 提取节点参数的方法（适配新蓝图格式）
        """ 从节点信息中提取参数，处理新格式的 {label, type, default} 结构 """      # 方法文档字符串
        data = node_info.get('data', {})                                        # 获取节点的 data 字段
        raw_params = data.get('params', {})                                     # 获取原始参数字典
        
        # 检查是否为新格式（参数值是包含 default 的对象）
        if not raw_params:                                                      # 如果参数为空
            return node_info.get('params', {})                                  # 回退到旧格式
        
        first_value = next(iter(raw_params.values()), None)                     # 获取第一个参数值
        if isinstance(first_value, dict) and 'default' in first_value:          # 如果是新格式
            result = {}                                                         # 存储提取后的参数
            for key, param_obj in raw_params.items():                           # 遍历所有参数
                default_val = param_obj.get('default')                          # 获取默认值
                param_type = param_obj.get('type', 'string')                    # 获取参数类型
                result[key] = self._convert_param_value(default_val, param_type) # 转换参数值类型
            return result                                                       # 返回提取后的参数字典
        
        return raw_params                                                       # 旧格式直接返回

    def _convert_param_value(self, value, param_type):                          # 转换参数值类型的方法
        """ 根据参数类型转换值 """                                                # 方法文档字符串
        if value == '' or value is None:                                        # 如果值为空
            return None                                                         # 返回 None
        
        if param_type == 'number':                                              # 如果是数字类型
            try:                                                                # 尝试转换
                return float(value) if '.' in str(value) else int(value)        # 根据是否有小数点决定转换类型
            except (ValueError, TypeError):                                     # 转换失败
                return None                                                     # 返回 None
        elif param_type == 'boolean':                                           # 如果是布尔类型
            if isinstance(value, bool):                                         # 如果已经是布尔值
                return value                                                    # 直接返回
            return str(value).lower() in ('true', '1', 'yes')                   # 否则解析字符串
        
        return value                                                            # 其他类型直接返回原值
