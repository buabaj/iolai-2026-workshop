from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL_ID = "."
LOAD_MODE_AWQ = "awq"
LOAD_MODE_BNB = "bnb"

PRIMARY_TOKEN_BASE = 48
PRIMARY_TOKEN_PER_ITEM = 64
PRIMARY_TOKEN_CAP = 768
BASELINE_TOKEN_FLOOR = 1024

RESCUE_TOKEN_BASE = 120
RESCUE_TOKEN_PER_ITEM = 80
RESCUE_TOKEN_CAP = 1024

EXPLAIN_MAX_NEW_TOKENS = 128


def primary_max_new_tokens(n_items: int, strategy: str = "") -> int:
    n = max(1, n_items)
    tokens = min(PRIMARY_TOKEN_CAP, PRIMARY_TOKEN_BASE + PRIMARY_TOKEN_PER_ITEM * n)
    if strategy == "baseline":
        return max(tokens, BASELINE_TOKEN_FLOOR)
    return tokens


def rescue_max_new_tokens(n_items: int) -> int:
    n = max(1, n_items)
    return min(RESCUE_TOKEN_CAP, RESCUE_TOKEN_BASE + RESCUE_TOKEN_PER_ITEM * n)


@dataclass
class ModelBundle:
    tok: Any
    model: Any
    model_id: str


def load_model(
    model_id: str | None = None,
    *,
    offline: bool | None = None,
    load_mode: str | None = None,
) -> ModelBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_id or os.environ.get("IOL_MODEL_ID", DEFAULT_MODEL_ID)
    load_mode = (load_mode or os.environ.get("IOL_LOAD", LOAD_MODE_AWQ)).strip().lower()

    if offline is None:
        offline = model_id == DEFAULT_MODEL_ID or os.environ.get("HF_HUB_OFFLINE") == "1"

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(model_id)

    if load_mode == LOAD_MODE_BNB:
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

    return ModelBundle(tok=tok, model=model, model_id=model_id)


def _prompt_tensors(tok: Any, model: Any, messages: list[dict[str, str]]):
    try:
        encoded = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return {"input_ids": ids.to(model.device)}, ids.shape[-1]

    if hasattr(encoded, "to"):
        encoded = encoded.to(model.device)
        return encoded, encoded["input_ids"].shape[-1]

    if isinstance(encoded, dict):
        moved = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in encoded.items()}
        return moved, moved["input_ids"].shape[-1]

    ids = encoded.to(model.device)
    return {"input_ids": ids}, ids.shape[-1]


def generate(
    bundle: ModelBundle,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 512,
) -> str:
    import torch

    inputs, prompt_len = _prompt_tensors(bundle.tok, bundle.model, messages)
    with torch.no_grad():
        output = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return bundle.tok.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
