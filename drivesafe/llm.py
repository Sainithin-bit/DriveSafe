from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import transformers

from drivesafe.constants import DEFAULT_MODEL_ID, HF_ADAPTER_REPO, HF_ADAPTER_WEIGHT_FILE


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_llm_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    torch_dtype: str = "bfloat16",
) -> transformers.Pipeline:
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": getattr(__import__("torch"), torch_dtype)},
        device_map="auto",
    )


def run_chat(
    pipeline: transformers.Pipeline,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    generated = outputs[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1]["content"]
    return str(generated)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
