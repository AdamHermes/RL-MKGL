"""Compact MKGL weights for a frozen conditioner, not an optimizer resume file."""
import os
import tempfile

import torch
from transformers import TrainerCallback


def save_baseline_checkpoint(model, path, metadata, step=0, epoch=0,
                             training_completed=False):
    components = ("lora_", "context_retriever", "score_retriever")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if any(component in name for component in components)
    }
    for component in components:
        if not any(component in name for name in state):
            raise ValueError("Missing checkpoint component: %s" % component)
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and name not in state]
    if missing:
        raise ValueError("Checkpoint would omit trainable parameters: %s" % missing)

    payload = {
        "format": "mkgl_baseline_components_v1",
        "state_dict": state,
        "metadata": metadata,
        "global_step": int(step),
        "epoch": float(epoch or 0),
        "training_completed": bool(training_completed),
    }
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=parent, suffix=".tmp", delete=False) as f:
        temporary_path = f.name
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    print("Saved MKGL checkpoint: %s (step=%d, completed=%s)" %
          (path, step, training_completed), flush=True)


class BaselineCheckpointCallback(TrainerCallback):
    def __init__(self, model, path, metadata, every_steps=500):
        self.mkgl_model = model
        self.path = path
        self.metadata = metadata
        self.every_steps = every_steps
        self.best_mrr = float("-inf")
        self.best_path = os.path.join(os.path.dirname(path), "mkgl_best.pt")

    def _save(self, state, path=None):
        if state.is_world_process_zero:
            save_baseline_checkpoint(
                self.mkgl_model, path or self.path, self.metadata,
                step=state.global_step, epoch=state.epoch,
            )

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps > 0 and state.global_step % self.every_steps == 0:
            self._save(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._save(state)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        mrr = (metrics or {}).get("eval_mrr")
        if state.global_step > 0 and mrr is not None and mrr > self.best_mrr:
            self.best_mrr = mrr
            self._save(state, self.best_path)
