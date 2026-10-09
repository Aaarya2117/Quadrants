import sys
import types
import unittest
from unittest import mock

from modelswap.engine import REAL_MODULE, load_engine
from modelswap.results import EngineError
from modelswap.stub_engine import StubEngine


class LoadEngineTests(unittest.TestCase):
    def test_auto_falls_back_to_stub_when_real_module_missing(self):
        with mock.patch.dict(sys.modules, {REAL_MODULE: None}):
            engine, backend = load_engine("auto")
        self.assertEqual(backend, "stub")
        self.assertIsInstance(engine, StubEngine)

    def test_stub_backend_is_always_stub(self):
        fake = types.ModuleType(REAL_MODULE)
        fake.create_engine = mock.Mock(return_value=object())
        with mock.patch.dict(sys.modules, {REAL_MODULE: fake}):
            engine, backend = load_engine("stub")
        self.assertEqual(backend, "stub")
        self.assertIsInstance(engine, StubEngine)
        fake.create_engine.assert_not_called()

    def test_real_backend_requested_but_missing_raises(self):
        with mock.patch.dict(sys.modules, {REAL_MODULE: None}):
            with self.assertRaises(EngineError):
                load_engine("real")

    def test_auto_uses_real_engine_when_module_provides_it(self):
        sentinel = object()
        fake = types.ModuleType(REAL_MODULE)
        fake.create_engine = mock.Mock(return_value=sentinel)
        with mock.patch.dict(sys.modules, {REAL_MODULE: fake}):
            engine, backend = load_engine("auto", state_file=None)
        self.assertEqual(backend, "real")
        self.assertIs(engine, sentinel)
        fake.create_engine.assert_called_once_with(None, None)

    def test_unknown_backend_raises(self):
        with self.assertRaises(EngineError):
            load_engine("gpu")


if __name__ == "__main__":
    unittest.main()
