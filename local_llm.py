"""Shared helper for calling a local transformer model."""

from __future__ import annotations

import os
import subprocess
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OLLAMA_PREFIX = "ollama://"
LOCAL_MODEL_ID = os.environ.get("LOCAL_LLM_MODEL", f"{OLLAMA_PREFIX}llama2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    return _TOKENIZER, _MODEL


def _call_ollama(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    model_name = LOCAL_MODEL_ID[len(OLLAMA_PREFIX) :]
    cmd = [
        "ollama",
        "run",
        model_name,
        prompt,
        "--max-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

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
        return _call_ollama(prompt, max_new_tokens, temperature, top_p, repetition_penalty)

    tokenizer, model = _load_model()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
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
