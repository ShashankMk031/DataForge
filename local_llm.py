"""Shared helper for calling a local transformer model."""

from __future__ import annotations

import os
import subprocess
import logging
from typing import Dict, Iterator
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OLLAMA_PREFIX = "ollama://"
LOCAL_MODEL_ID = os.environ.get("LOCAL_LLM_MODEL", f"{OLLAMA_PREFIX}llama2")
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

logger = logging.getLogger(__name__)

_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModelForCausalLM | None = None


def _load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _TOKENIZER, _MODEL
    if _MODEL is None or _TOKENIZER is None:
        tokenizer_kwargs: Dict[str, object] = {
            "use_fast": False,
            "padding_side": "left",
        }
        _TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, **tokenizer_kwargs)

        model_kwargs: Dict[str, object] = {}
        if DEVICE == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["low_cpu_mem_usage"] = True

        _MODEL = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, **model_kwargs)
        _MODEL.eval()
        if DEVICE == "mps":
            _MODEL.to(torch.device("mps"))

    return _TOKENIZER, _MODEL


def _call_ollama(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Call Ollama CLI. Note: Ollama CLI does not support --max-tokens, --temperature, --top-p flags.
    Model parameters should be configured via the model's Modelfile or Ollama HTTP API.
    """
    model_name = LOCAL_MODEL_ID[len(OLLAMA_PREFIX) :]
    cmd = [
        "ollama",
        "run",
        model_name,
        prompt,
    ]

    timeout_seconds = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ollama timed out after {timeout_seconds} seconds while processing prompt"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown"
        raise RuntimeError(f"ollama failed (exit {result.returncode}): {stderr}")

    return result.stdout.strip()


def call_local_llm(
    prompt: str,
    max_new_tokens: int = 1000,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.03,
) -> str:
    """Generate text for the provided prompt using the cached local model."""

    if LOCAL_MODEL_ID.startswith(OLLAMA_PREFIX):
        return _call_ollama(prompt, max_new_tokens, temperature, top_p)

    tokenizer, model = _load_model()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

    # Compute safe token IDs, handling None values
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id
    if eos_id is None:
        eos_id = tokenizer.cls_token_id
    if eos_id is None:
        logger.warning("Using eos_id=0 fallback; tokenizer may be misconfigured.")
        eos_id = 0  # Safe default fallback
    
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            do_sample=True,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded.strip()


def local_model_info() -> str:
    """Return a short description of the model currently in use."""

    if LOCAL_MODEL_ID.startswith(OLLAMA_PREFIX):
        return f"{LOCAL_MODEL_ID} @ ollama"

    tokenizer, model = _load_model()
    return f"{LOCAL_MODEL_ID} @ {DEVICE} ({model.num_parameters():,} params)"


@contextmanager
def with_model_override(model_id: str) -> Iterator[None]:
    """Temporarily override LOCAL_MODEL_ID for scoring with a different model."""
    global LOCAL_MODEL_ID, _MODEL, _TOKENIZER
    original = LOCAL_MODEL_ID
    LOCAL_MODEL_ID = model_id
    _MODEL = None
    _TOKENIZER = None
    try:
        yield
    finally:
        LOCAL_MODEL_ID = original
        _MODEL = None
        _TOKENIZER = None
