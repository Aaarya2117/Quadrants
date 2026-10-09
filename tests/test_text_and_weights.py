import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, RobertaConfig, RobertaForSequenceClassification

from modelswap import weights
from modelswap.render import render_weights
from modelswap.results import WeightInfo, WeightReport
from modelswap.text import read_text_meta, save_text_checkpoint, verify_text_pair


def tiny_bert(seed=0):
    torch.manual_seed(seed)
    return BertForSequenceClassification(BertConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, num_labels=2))


def tiny_roberta(seed=0):
    torch.manual_seed(seed)
    return RobertaForSequenceClassification(RobertaConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, num_labels=2, max_position_embeddings=64))


class TextCheckpointTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_meta_round_trip(self):
        path = self.dir / "bert.pt"
        save_text_checkpoint(tiny_bert(), "bert-base-uncased", path)
        meta = read_text_meta(path)
        self.assertEqual(meta["architecture"], "bert_seqcls")
        self.assertEqual(meta["num_labels"], 2)
        self.assertEqual(meta["base_model"], "bert-base-uncased")
        self.assertNotIn("state_dict", meta)

    def test_verify_refuses_cross_architecture_by_default(self):
        bert, roberta = self.dir / "bert.pt", self.dir / "roberta.pt"
        save_text_checkpoint(tiny_bert(), "bert-base-uncased", bert)
        save_text_checkpoint(tiny_roberta(), "roberta-base", roberta)
        ok, message = verify_text_pair(str(roberta), str(bert), {})
        self.assertFalse(ok)
        self.assertIn("architecture differs", message)

    def test_verify_allows_cross_architecture_when_entry_allows_it(self):
        bert, roberta = self.dir / "bert.pt", self.dir / "roberta.pt"
        save_text_checkpoint(tiny_bert(), "bert-base-uncased", bert)
        save_text_checkpoint(tiny_roberta(), "roberta-base", roberta)
        ok, message = verify_text_pair(str(roberta), str(bert), {"cross_architecture": True})
        self.assertTrue(ok)
        self.assertIn("bert_seqcls -> roberta_seqcls", message)

    def test_verify_refuses_different_num_labels(self):
        bert, bert3 = self.dir / "bert.pt", self.dir / "bert3.pt"
        save_text_checkpoint(tiny_bert(), "bert-base-uncased", bert)
        torch.manual_seed(1)
        three = BertForSequenceClassification(BertConfig(
            vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, num_labels=3))
        save_text_checkpoint(three, "bert-base-uncased", bert3)
        ok, message = verify_text_pair(str(bert3), str(bert), {"cross_architecture": True})
        self.assertFalse(ok)
        self.assertIn("num_labels", message)

    def test_verify_refuses_unreadable_candidate(self):
        bert = self.dir / "bert.pt"
        save_text_checkpoint(tiny_bert(), "bert-base-uncased", bert)
        bad = self.dir / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        ok, _ = verify_text_pair(str(bad), str(bert), {})
        self.assertFalse(ok)


class WeightTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _save(self, name, model):
        path = self.dir / name
        save_text_checkpoint(model, "bert-base-uncased", path)
        return str(path)

    def test_hash_is_deterministic_and_changes_with_weights(self):
        a1 = weights.tensor_hash(tiny_bert(0).state_dict())
        a2 = weights.tensor_hash(tiny_bert(0).state_dict())
        b = weights.tensor_hash(tiny_bert(1).state_dict())
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)
        self.assertEqual(len(a1), 16)

    def test_describe_pair_same_architecture_gives_delta(self):
        prev = self._save("prev.pt", tiny_bert(0))
        new = self._save("new.pt", tiny_bert(1))
        report = weights.describe_pair(prev, new)
        self.assertIsNotNone(report.previous)
        self.assertIsNotNone(report.new)
        self.assertNotEqual(report.previous.sha256, report.new.sha256)
        self.assertGreater(report.delta_l2, 0)
        self.assertGreater(report.changed_tensors, 0)
        self.assertIsNone(report.conversion_verified)

    def test_describe_pair_identical_weights_have_zero_delta(self):
        prev = self._save("prev.pt", tiny_bert(0))
        new = self._save("new.pt", tiny_bert(0))
        report = weights.describe_pair(prev, new)
        self.assertEqual(report.delta_l2, 0.0)
        self.assertEqual(report.changed_tensors, 0)

    def test_describe_pair_different_architectures_has_no_delta(self):
        prev = self._save("bert.pt", tiny_bert(0))
        roberta_path = self.dir / "roberta.pt"
        save_text_checkpoint(tiny_roberta(0), "roberta-base", roberta_path)
        report = weights.describe_pair(prev, str(roberta_path))
        self.assertIsNone(report.delta_l2)
        self.assertIsNone(report.changed_tensors)
        self.assertIn("different architectures", report.note)

    def test_describe_pair_reports_unreadable_file(self):
        prev = self._save("prev.pt", tiny_bert(0))
        report = weights.describe_pair(prev, str(self.dir / "missing.pt"))
        self.assertIsNone(report.new)
        self.assertIn("new unreadable", report.note)

    def test_verify_active_passes_when_hash_matches(self):
        new = self._save("new.pt", tiny_bert(1))
        report = weights.describe_pair(None, new)
        self.assertTrue(weights.verify_active(report, new).conversion_verified)

    def test_verify_active_fails_when_active_holds_other_weights(self):
        new = self._save("new.pt", tiny_bert(1))
        other = self._save("other.pt", tiny_bert(0))
        report = weights.describe_pair(None, new)
        verified = weights.verify_active(report, other)
        self.assertFalse(verified.conversion_verified)
        self.assertIn("active hash", verified.note)


class RenderWeightsTests(unittest.TestCase):
    def info(self, path, sha):
        return WeightInfo(path=path, params=1000, tensors=4, size_mb=1.5, l2_norm=12.5, sha256=sha)

    def test_shows_previous_new_delta_and_verified(self):
        report = WeightReport(
            previous=self.info("models/a.pt", "aaaa"),
            new=self.info("models/b.pt", "bbbb"),
            delta_l2=3.25,
            changed_tensors=4,
            conversion_verified=True,
        )
        text = "\n".join(render_weights(report))
        self.assertIn("previous  models/a.pt", text)
        self.assertIn("new       models/b.pt", text)
        self.assertIn("hash aaaa", text)
        self.assertIn("hash bbbb", text)
        self.assertIn("delta     ||W_new - W_old||_F = 3.2500 (4 of 4 tensors changed)", text)
        self.assertIn("conversion VERIFIED", text)

    def test_shows_not_verified_and_cross_architecture(self):
        report = WeightReport(
            previous=self.info("models/a.pt", "aaaa"),
            new=self.info("models/b.pt", "bbbb"),
            delta_l2=None,
            changed_tensors=None,
            conversion_verified=False,
            note="active hash cccc != new bbbb",
        )
        text = "\n".join(render_weights(report))
        self.assertIn("n/a (different architectures", text)
        self.assertIn("conversion NOT VERIFIED: active hash cccc != new bbbb", text)


if __name__ == "__main__":
    unittest.main()
