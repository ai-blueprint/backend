/**
 * nodeRegistry.js - 节点注册表
 * 
 * 存储所有可用节点的配置信息
 * 实际项目中，这些数据应该从后端API获取
 */


// ========== 节点注册表数据 ==========

export const NODE_REGISTRY = {
  
  /** 节点分类：每个分类包含显示名称、主题色、包含的节点ID列表 */
  categories: {
    node_group1: {
      label: "节点组1",                                                          // 分类名称
      color: "rgb(137, 146, 235)",                                               // 主题色
      nodes: ["node1", "node2"],                                                 // 包含的节点
    },
    node_group2: {
      label: "节点组2",
      color: "rgb(242, 177, 144)",
      nodes: ["node3", "node4"],
    },
  },

  /** 节点配置：每个节点包含显示名称、输入端口、输出端口、参数配置 */
  nodes: {
    node1: {
      label: "节点1",                                                            // 节点名称
      inputs: [{ id: "in", label: "" }],                                         // 输入端口
      outputs: [{ id: "out", label: "" }],                                       // 输出端口
      params: {                                                                  // 参数配置
        param1: { label: "参数1", type: "number", default: 1 },
        param2: { label: "参数2", type: "boolean", default: false },
        param3: { label: "参数3", type: "string", default: "3" },
        param4: { label: "参数4", type: "number", default: 4 },
        param5: { label: "参数5", type: "number", default: 5 },
        param6: { label: "参数6", type: "number", default: 6 },
        param7: { label: "参数7", type: "number", default: 7 },
      },
    },
    node2: {
      label: "节点2",
      inputs: [{ id: "in", label: "输入" }],
      outputs: [{ id: "out", label: "输出" }],
      params: {
        param1: { label: "参数1", type: "number", default: 1 },
        param2: { label: "参数2", type: "number", default: 2 },
      },
    },
    node3: {
      label: "节点3",
      inputs: [{ id: "in", label: "输入" }],
      outputs: [{ id: "out", label: "输出" }],
      params: {
        param1: { label: "参数1", type: "number", default: 1 },
        param2: { label: "参数2", type: "number", default: 2 },
      },
    },
    node4: {
      label: "节点4",
      inputs: [{ id: "in", label: "输入" }],
      outputs: [{ id: "out", label: "输出" }],
      params: {
        param1: { label: "参数1", type: "number", default: 1 },
        param2: { label: "参数2", type: "number", default: 2 },
      },
    },
  },
};


// ========== 辅助函数 ==========

/** 根据节点ID获取节点配置 */
export const getNodeConfig = (nodeKey) => {
  return NODE_REGISTRY.nodes[nodeKey] || {};                                     // 取不到就返回空对象
};

/** 根据节点ID找到它属于哪个分类 */
export const findCategoryByNode = (nodeKey) => {
  const categories = NODE_REGISTRY.categories;                                   // 获取所有分类
  for (const catKey of Object.keys(categories)) {                                // 遍历分类
    if (categories[catKey].nodes.includes(nodeKey)) return catKey;               // 找到就返回分类ID
  }
  return null;                                                                   // 找不到返回 null
};

/** 根据节点ID获取它的主题色 */
export const getNodeColor = (nodeKey) => {
  const catKey = findCategoryByNode(nodeKey);                                    // 第1步：找到分类
  if (!catKey) return undefined;                                                 // 第2步：找不到分类就返回 undefined
  return NODE_REGISTRY.categories[catKey].color;                                 // 第3步：返回分类的颜色
};

/** 获取所有分类（用于渲染节点面板） */
export const getAllCategories = () => {
  return Object.entries(NODE_REGISTRY.categories);                               // 返回 [分类ID, 分类数据] 数组
};
