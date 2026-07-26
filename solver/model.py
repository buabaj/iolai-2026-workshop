from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL_ID = "."
LOAD_MODE_AWQ = "awq"
LOAD_MODE_BNB = "bnb"


@dataclass
class ModelBundle:
    tok: Any
    model: Any
    model_id: str


@dataclass(frozen=True)
class GenStats:
    prompt_tokens: int
    new_tokens: int
    hit_max_new: bool
    eos_limited: bool


def assert_gpu_resident(bundle: ModelBundle) -> None:
    import torch

    if not torch.cuda.is_available():
        print("warn: CUDA unavailable", flush=True)
        return
    bad = []
    for name, param in bundle.model.named_parameters():
        if not str(param.device).startswith("cuda"):
            bad.append(f"{name}:{param.device}")
            if len(bad) >= 5:
                break
    if bad:
        raise RuntimeError(f"non-CUDA parameters: {bad}")
    print(
        f"gpu ok | VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB",
        flush=True,
    )


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
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    dtype_kwargs = _dtype_kwargs(torch)
    preferred: Any = {"": 0} if torch.cuda.is_available() else "auto"

    def _load(device_map: Any):
        if load_mode == LOAD_MODE_BNB:
            from transformers import BitsAndBytesConfig

            return AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map=device_map,
            ).eval()
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                **dtype_kwargs,
            ).eval()
        except ImportError as exc:
            raise ImportError(
                "AWQ load failed; install gptqmodel/autoawq or use IOL_LOAD=bnb"
            ) from exc

    try:
        model = _load(preferred)
    except ImportError:
        raise
    except Exception as exc:
        if preferred == "auto":
            raise
        print(f"warn: device_map retry auto ({exc})", flush=True)
        model = _load("auto")

    _force_greedy(model)
    return ModelBundle(tok=tok, model=model, model_id=model_id)


def _force_greedy(model: Any) -> None:
    try:
        cfg = model.generation_config
        cfg.do_sample = False
        cfg.repetition_penalty = 1.0
        for key in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(cfg, key):
                setattr(cfg, key, None)
    except Exception:
        pass


def _dtype_kwargs(torch_mod) -> dict:
    try:
        import inspect
        from transformers import AutoModelForCausalLM

        if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
            return {"dtype": torch_mod.float16}
    except Exception:
        pass
    return {"torch_dtype": torch_mod.float16}


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
        moved = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in encoded.items()
        }
        return moved, moved["input_ids"].shape[-1]
    ids = encoded.to(model.device)
    return {"input_ids": ids}, ids.shape[-1]


def _pad_token_id(bundle: ModelBundle) -> int | None:
    if getattr(bundle.tok, "pad_token_id", None) is not None:
        return int(bundle.tok.pad_token_id)
    if getattr(bundle.tok, "eos_token_id", None) is not None:
        return int(bundle.tok.eos_token_id)
    return None


def generate(
    bundle: ModelBundle,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
) -> str:
    text, _ = generate_with_stats(bundle, messages, max_new_tokens=max_new_tokens)
    return text


def generate_with_stats(
    bundle: ModelBundle,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
) -> tuple[str, GenStats]:
    import torch

    _force_greedy(bundle.model)
    inputs, prompt_len = _prompt_tensors(bundle.tok, bundle.model, messages)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.0,
    }
    pad_id = _pad_token_id(bundle)
    if pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    with torch.no_grad():
        output = bundle.model.generate(**inputs, **kwargs)
    new_tokens = int(output.shape[-1] - prompt_len)
    hit_max = new_tokens >= max_new_tokens
    text = bundle.tok.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    return text, GenStats(
        prompt_tokens=int(prompt_len),
        new_tokens=new_tokens,
        hit_max_new=hit_max,
        eos_limited=not hit_max,
    )
