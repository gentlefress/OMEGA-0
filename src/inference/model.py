"""WAM input preprocessing, model inference, and action denormalization.

Learning dependencies are loaded only when constructing WAMInference.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import time

import numpy as np

from omega_real.core.protocol import action_spec


def model_action_spec(model_config):
    """Describe the model's float32 action vector at the Sonic 30 Hz cadence."""
    dimension = model_config["action_dim"]
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Model action_dim must be a positive integer")
    return {"id": f"omega.action{dimension}.v1",
            "fields": {"action": {"dtype": "float32", "shape": [dimension]}},
            "period_ns": 33_333_333}


class WAMInference:
    @classmethod
    def load(cls, **options):
        from .checkpoint import load_checkpoint
        model, metadata = load_checkpoint(**options)
        return cls(model, metadata, device=options.get("device", "cuda"))

    def __init__(self, model, metadata, *, device="cpu", processor=None, tokenizer=None):
        import torch
        from omega.data.normalization import Normalizer
        from transformers import AutoProcessor, T5Tokenizer
        self.model, self.metadata, self.device = model, metadata, torch.device(device)
        self.cfg, self.pre = metadata["model"], metadata["preprocessing"]
        derived = model_action_spec(self.cfg)
        self.spec = action_spec(metadata.get("action_spec", derived))
        field = self.spec.fields.get("action")
        if (set(self.spec.fields) != {"action"} or field.dtype != "float32"
                or field.shape != (self.cfg["action_dim"],)):
            raise ValueError("WAM requires one float32 action field matching model action_dim")
        self.dimension = self.cfg["action_dim"]
        local = self.cfg.get("local_files_only", True)
        self.processor = processor or AutoProcessor.from_pretrained(self.cfg["model_name_or_path"], local_files_only=local)
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained(self.cfg["text_model"], use_fast=False, local_files_only=local)
        self.normalizer = Normalizer(metadata["normalization"]["action"], self.dimension)
        predictor = getattr(model, "predictor", None)
        state_dim = predictor.state_encoder.in_features if predictor is not None else self.cfg.get("ac_vjepa_state_dim") or 45
        self.state_normalizer = Normalizer(metadata["normalization"].get("state", {"mode": "none"}), state_dim)
        self.history = deque(maxlen=int(self.pre.get("history_size", 1)))
        if self.history.maxlen < 1:
            raise ValueError("history_size must be positive")
        self.current_instruction = None
        self.text_inputs = None
        self.reset(None)

    def reset(self, session_id):
        import torch
        self.session_id, self.last_image = session_id, None
        self.current_view = None
        self.history.clear()
        self.previous_actions = None
        self.last_prediction_time = None
        seed = int(self.pre.get("seed", 1039))
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def preprocess(self, observation, *, view="ego"):
        import torch
        from PIL import Image
        from torchvision.transforms import v2
        from qwen_vl_utils import process_vision_info
        if view not in ("ego", "exo"):
            raise ValueError("WAM request view must be 'ego' or 'exo'")
        if getattr(self, "current_view", None) != view:
            self.history.clear()
            self.last_image = None
            self.previous_actions = None
            self.current_view = view
        sample = observation.samples[self.pre.get("image_field", "image")]
        if not sample.valid:
            raise ValueError("WAM image is invalid")
        pixels = np.asarray(sample.value)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("WAM requires RGB uint8 HWC images")
        identity = (sample.stream, sample.sequence, sample.received_ns)
        if identity != self.last_image:
            self.history.append(Image.fromarray(pixels))
            self.last_image = identity
        history = [self.history[0]] * (self.history.maxlen - len(self.history)) + list(self.history)
        valid = torch.tensor([[0.] * (self.history.maxlen - len(self.history)) + [1.] * len(self.history)])
        size = self.cfg.get("ac_vjepa_image_size", 224)
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((size, size)), v2.CenterCrop(size)])
        context = torch.stack([transform(frame) for frame in history]).unsqueeze(0)
        instruction = observation.instruction
        vlm_instruction = instruction.strip().lower() if self.pre.get("lowercase_instruction", True) else instruction.strip()
        current = v2.Resize(tuple(self.pre.get("vlm_size", [270, 480])))(history[-1])
        view_tag = "<EGO_VIEW>" if view == "ego" else "<EXO_VIEW>"
        messages = [{"role": "user", "content": [{"type": "text", "text": view_tag + "\n"}, {"type": "image", "image": current}, {"type": "text", "text": vlm_instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        images, _ = process_vision_info(messages, image_patch_size=16)
        inputs = dict(self.processor(text=text, images=images, padding=True, return_tensors="pt"))
        if instruction != self.current_instruction:
            self.current_instruction = instruction
            token = self.tokenizer(instruction, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
            self.text_inputs = {"input_ids": token["input_ids"],
                                "attention_mask": token["input_ids"].ne(getattr(self.tokenizer, "pad_token_id", 0))}
        token = self.text_inputs
        inputs.update(t5_input_ids=token["input_ids"], t5_attention_mask=token["attention_mask"], states=None,
                      current_frames=context[:, 0], context_frames=context, context_valid_mask=valid,
                      view_token=torch.tensor([1 if view == "ego" else 0], dtype=torch.long))
        state = self._state(observation)
        if state is not None:
            inputs["states"] = torch.from_numpy(state).reshape(1, 1, -1)
            inputs["state_valid_mask"] = torch.ones(1, 1)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def _state(self, observation):
        """Normalize the state vector assembled by the real-world client."""
        sample = observation.samples.get(self.pre.get("state_field", "state"))
        if sample is None:
            if self.pre.get("require_state", False):
                raise ValueError("WAM inference requires measured robot state")
            return None
        if not sample.valid:
            raise ValueError("WAM state input is invalid")
        return self.state_normalizer.normalize(np.asarray(sample.value, np.float32).reshape(-1))

    def predict(self, request):
        import torch
        now = time.monotonic()
        view_changed = getattr(self, "current_view", None) not in (None, request.view)
        previous_time = getattr(self, "last_prediction_time", None)
        idle_reset = previous_time is not None and now - previous_time > float(self.pre.get("idle_reset_seconds", 2.))
        if idle_reset:
            self.reset(request.session_id)
        self.last_prediction_time = now
        inputs = self.preprocess(request.observation, view=request.view)
        horizon = self.cfg["action_chunk_size"]
        noise = torch.randn(1, horizon, self.dimension, device=self.device, generator=self.generator)
        prefix = request.context.get("previous_actions")
        if "inference_start_idx" in request.context:
            start = request.context["inference_start_idx"]
            count = request.context.get("inference_delay", 0)
            if type(start) is not int or type(count) is not int or min(start, count) < 0 or count > horizon:
                raise ValueError("Invalid RTC start index or delay")
            if self.previous_actions is not None and not idle_reset and not view_changed and count:
                if start + count > len(self.previous_actions):
                    raise ValueError("RTC prefix extends beyond the previous prediction")
                noise[:, :count] = torch.from_numpy(self.previous_actions[start:start + count]).to(self.device)
        elif prefix and not idle_reset and not view_changed:
            if set(prefix) != {"action"}:
                raise ValueError("RTC prefix requires one action array")
            values = np.asarray(prefix["action"], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != self.dimension or not np.isfinite(values).all():
                raise ValueError("Invalid RTC action prefix")
            count = min(len(values), horizon, int(request.context.get("inference_delay", 0)))
            if count < 0:
                raise ValueError("RTC inference delay cannot be negative")
            if count:
                noise[:, :count] = torch.from_numpy(self.normalizer.normalize(values[:count])).to(self.device)
        inputs["init_action"] = noise
        precision = self.pre.get("precision", "bfloat16")
        autocast = torch.autocast(self.device.type, dtype=getattr(torch, precision)) if precision != "float32" else nullcontext()
        with torch.inference_mode(), autocast:
            result = self.model.sample(**inputs).pred_actions
        self.previous_actions = result[0].float().cpu().numpy().copy()
        return self.postprocess(result)

    def postprocess(self, result):
        """Return a denormalized float32 array of shape (horizon, action_dim)."""
        values = self.normalizer.denormalize(result[0].float().cpu().numpy())
        if values.shape != (self.cfg["action_chunk_size"], self.dimension):
            raise ValueError("Unexpected model output shape")
        return values.astype(np.float32)
