import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from baseline_checkpoint import BaselineCheckpointCallback, save_baseline_checkpoint


class ToyMKGL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.backbone.requires_grad_(False)
        self.lora_A = torch.nn.Linear(2, 1, bias=False)
        self.context_retriever = torch.nn.Linear(2, 2)
        self.score_retriever = torch.nn.Linear(2, 1)
        self.score_retriever.register_buffer("scaler", torch.tensor(2.0))


class BaselineCheckpointTests(unittest.TestCase):
    def test_all_adapted_weights_and_buffers_round_trip(self):
        model = ToyMKGL()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            save_baseline_checkpoint(model, path, {"base_model": "toy"},
                                     step=7, epoch=1.5, training_completed=True)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["global_step"], 7)
            self.assertTrue(payload["training_completed"])
            self.assertFalse(any("backbone" in key for key in payload["state_dict"]))
            restored = ToyMKGL()
            missing = restored.load_state_dict(payload["state_dict"], strict=False)
            self.assertEqual(set(missing.missing_keys), {"backbone.weight", "backbone.bias"})
            self.assertEqual(missing.unexpected_keys, [])
            for key, value in payload["state_dict"].items():
                torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key])
                self.assertEqual(value.device.type, "cpu")

    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            model = ToyMKGL()
            save_baseline_checkpoint(model, path, {}, step=1)
            with patch("baseline_checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_baseline_checkpoint(model, path, {}, step=2)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["global_step"], 1)
            self.assertEqual(os.listdir(directory), ["checkpoint.pt"])

    def test_refuses_to_omit_trainable_backbone(self):
        model = ToyMKGL()
        model.backbone.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "omit trainable"):
            save_baseline_checkpoint(model, "unused.pt", {})

    def test_periodic_and_best_validation_callbacks(self):
        callback = BaselineCheckpointCallback(ToyMKGL(), "outputs/latest.pt", {}, 5)
        state = SimpleNamespace(global_step=4, epoch=0.5, is_world_process_zero=True)
        with patch("baseline_checkpoint.save_baseline_checkpoint") as save:
            callback.on_step_end(None, state, None)
            save.assert_not_called()
            state.global_step = 5
            callback.on_step_end(None, state, None)
            self.assertEqual(save.call_count, 1)
            callback.on_evaluate(None, state, None, {"eval_mrr": 0.3})
            callback.on_evaluate(None, state, None, {"eval_mrr": 0.2})
            self.assertEqual(save.call_count, 2)
            self.assertEqual(save.call_args.args[1], callback.best_path)
            state.is_world_process_zero = False
            callback.on_epoch_end(None, state, None)
            self.assertEqual(save.call_count, 2)
            state.is_world_process_zero = True
            state.global_step = 0
            callback.on_evaluate(None, state, None, {"eval_mrr": 0.9})
            self.assertEqual(save.call_count, 2)


if __name__ == "__main__":
    unittest.main()
