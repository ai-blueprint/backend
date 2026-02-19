import tempfile
import unittest
from pathlib import Path
from unittest import mock

import loader


class LoaderTest(unittest.TestCase):
    def test_load_all_loads_nested_python_modules(self):
        with tempfile.TemporaryDirectory() as tempDir:
            tempPath = Path(tempDir)
            nodesDir = tempPath / "nodes"
            nestedDir = nodesDir / "nested"
            nestedDir.mkdir(parents=True)

            (nodesDir / "__init__.py").write_text("", encoding="utf-8")
            (nodesDir / "math_node.py").write_text("# node", encoding="utf-8")
            (nestedDir / "activation_node.py").write_text("# nested node", encoding="utf-8")

            with mock.patch("loader.os.path.dirname", return_value=tempDir):
                with mock.patch("loader.importModule") as importModuleMock:
                    loader.loadAll()

            imported = [call.args[0].replace("\\", "/") for call in importModuleMock.call_args_list]
            self.assertEqual(imported, ["nodes/math_node.py", "nodes/nested/activation_node.py"])

    def test_load_all_skips_non_python_init_and_pycache(self):
        with tempfile.TemporaryDirectory() as tempDir:
            tempPath = Path(tempDir)
            nodesDir = tempPath / "nodes"
            pycacheDir = nodesDir / "__pycache__"
            nestedDir = nodesDir / "nested"
            pycacheDir.mkdir(parents=True)
            nestedDir.mkdir(parents=True)

            (nodesDir / "__init__.py").write_text("", encoding="utf-8")
            (nodesDir / "README.md").write_text("ignore", encoding="utf-8")
            (pycacheDir / "compiled.py").write_text("# ignore", encoding="utf-8")
            (nestedDir / "__init__.py").write_text("", encoding="utf-8")
            (nestedDir / "node.txt").write_text("ignore", encoding="utf-8")
            (nestedDir / "valid_node.py").write_text("# valid", encoding="utf-8")

            with mock.patch("loader.os.path.dirname", return_value=tempDir):
                with mock.patch("loader.importModule") as importModuleMock:
                    loader.loadAll()

            imported = [call.args[0].replace("\\", "/") for call in importModuleMock.call_args_list]
            self.assertEqual(imported, ["nodes/nested/valid_node.py"])


if __name__ == "__main__":
    unittest.main()
