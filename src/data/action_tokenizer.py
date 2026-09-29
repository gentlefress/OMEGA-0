"""FAST continuous-action tokenization used by Qwen pretraining."""
import numpy as np


class FastActionTokenizer:
    def __init__(self, tokenizer, checkpoint, time_horizon, action_dim, bins=2048, *, local_files_only=True):
        from transformers import AutoProcessor
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=local_files_only)
        self.fast_tokenizer.action_dim = action_dim
        self.fast_tokenizer.time_horizon = time_horizon
        self.n_bins = bins
        if self.fast_tokenizer.vocab_size != bins:
            raise ValueError("FAST vocabulary size differs from action_bins")
        tokens = [f"<|a_{i}|>" for i in range(bins)]
        tokenizer.add_tokens(tokens)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|action_start|>", "<|action_end|>"]},
                                     replace_additional_special_tokens=False)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.action_token_begin_idx = ids[0]
        if ids != list(range(ids[0], ids[0] + bins)):
            raise ValueError("FAST action token IDs must be contiguous")

    def __call__(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.fast_tokenizer.time_horizon, self.fast_tokenizer.action_dim)
        if actions.ndim not in (2, 3) or actions.shape[-2:] != expected or not np.isfinite(actions).all():
            raise ValueError(f"FAST expects finite action chunks of shape {expected}")
        encoded = self.fast_tokenizer(actions)
        if any(not 0 <= index < self.n_bins for chunk in encoded for index in chunk):
            raise ValueError("FAST returned an out-of-vocabulary action token")
        strings = ["".join(f"<|a_{index}|>" for index in chunk) for chunk in encoded]
        return strings[0] if actions.ndim == 2 else strings

    def decode_token_ids_to_actions(self, token_ids):
        tokens = [[min(max(token - self.action_token_begin_idx, 0), self.n_bins - 1) for token in chunk]
                  for chunk in token_ids]
        return self.fast_tokenizer.decode(tokens).astype(np.float32)
