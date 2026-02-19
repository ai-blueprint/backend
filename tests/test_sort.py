import unittest  # 导入单元测试框架
import sort  # 导入待测试的排序模块


class TopoSortValidationTest(unittest.TestCase):  # 定义拓扑排序验证测试类
    def test_topo_sort_allows_invalid_edges_in_non_strict_mode(self):
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "missing", "target": "b"},
            {"source": "a", "target": "missing"},
        ]

        result = sort.topoSort(nodes, edges)

        self.assertEqual(result, ["a", "b"])

    def test_topo_sort_rejects_invalid_source_in_strict_mode(self):
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [{"source": "missing", "target": "b"}]

        with self.assertRaisesRegex(Exception, "源节点不存在"):
            sort.topoSort(nodes, edges, strict=True)

    def test_topo_sort_rejects_duplicate_node_ids(self):
        nodes = [{"id": "dup"}, {"id": "dup"}]
        edges = []

        with self.assertRaisesRegex(Exception, "重复节点ID"):
            sort.topoSort(nodes, edges)

    def test_topo_sort_cycle_error_contains_involved_nodes(self):
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
        ]

        with self.assertRaisesRegex(Exception, "涉及节点: a,b"):
            sort.topoSort(nodes, edges)


if __name__ == "__main__":
    unittest.main()
