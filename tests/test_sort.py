import unittest

import sort


class TopoSortValidationTest(unittest.TestCase):
    def test_topo_sort_allows_invalid_edges_in_non_strict_mode(self):
        nodes = [{"id": "a", "label": "NodeA"}, {"id": "b", "label": "NodeB"}]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "missing", "target": "b"},
            {"source": "a", "target": "missing"},
        ]

        result = sort.topoSort(nodes, edges)

        self.assertEqual(result, ["a", "b"])

    def test_topo_sort_rejects_invalid_source_in_strict_mode(self):
        nodes = [{"id": "a", "label": "NodeA"}, {"id": "b", "label": "NodeB"}]
        edges = [{"source": "missing", "target": "b"}]

        with self.assertRaisesRegex(Exception, "target label: NodeB"):
            sort.topoSort(nodes, edges, strict=True)

    def test_topo_sort_rejects_duplicate_node_ids(self):
        nodes = [{"id": "dup", "label": "First"}, {"id": "dup", "label": "Second"}]
        edges = []

        with self.assertRaisesRegex(Exception, "label: Second"):
            sort.topoSort(nodes, edges)

    def test_topo_sort_cycle_error_contains_involved_nodes(self):
        nodes = [{"id": "a", "label": "NodeA"}, {"id": "b", "label": "NodeB"}]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
        ]

        with self.assertRaisesRegex(Exception, "涉及节点: a\\(NodeA\\),b\\(NodeB\\)"):
            sort.topoSort(nodes, edges)


if __name__ == "__main__":
    unittest.main()
