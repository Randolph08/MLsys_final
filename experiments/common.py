"""Shared helpers for experiment scripts."""

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = "/home/hzliu/AD/Homework_haozhe/MLsys_final"
EAGLE_ROOT = os.path.join(PROJECT_ROOT, "EAGLE")


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content. Please ensure that your responses are socially unbiased and "
    "positive in nature.\n\nIf a question does not make any sense, or is not "
    "factually coherent, explain why instead of answering something not "
    "correct. If you don't know the answer to a question, please don't share "
    "false information."
)


TOY_PROMPTS = [
    "The capital of France is",
    "Explain the concept of machine learning in simple terms:",
    "Write a Python function to find the nth Fibonacci number:",
    "What are the main causes of climate change?",
    "Translate the following to French: Hello, how are you?",
]


def ensure_project_paths() -> None:
    """Make local project and EAGLE modules importable."""
    for path in (PROJECT_ROOT, EAGLE_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)


def build_chat_messages(
    user_prompt: str,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Build a single-turn chat message list."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_chat_input(
    tokenizer,
    user_prompt: Optional[str] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True,
    return_prompt: bool = False,
):
    """Tokenize a LLaMA-chat-style input with the model chat template."""
    if messages is None:
        if user_prompt is None:
            raise ValueError("Either user_prompt or messages must be provided.")
        messages = build_chat_messages(user_prompt, system_prompt=system_prompt)

    prompt = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
    if return_prompt:
        return input_ids, prompt
    return input_ids


def load_prompt_records(
    prompt_source: str,
    limit: Optional[int] = None,
    question_begin: Optional[int] = None,
    question_end: Optional[int] = None,
) -> List[Dict]:
    """Load prompts as records with an id and one or more user turns."""
    if prompt_source == "toy":
        records = [
            {"question_id": f"toy-{idx}", "turns": [prompt], "source": "toy"}
            for idx, prompt in enumerate(TOY_PROMPTS)
        ]
    elif prompt_source == "mt_bench":
        records = _load_mt_bench_records(question_begin, question_end)
    else:
        raise ValueError(f"Unsupported prompt_source: {prompt_source}")

    if limit is not None:
        records = records[:limit]
    return records


def trim_generated_ids(
    generated_ids: Sequence[int],
    stop_token_ids: Sequence[Optional[int]],
    max_new_tokens: Optional[int] = None,
) -> List[int]:
    """Trim at the first stop token and optionally cap to max_new_tokens."""
    trimmed = list(generated_ids)
    valid_stops = {int(x) for x in stop_token_ids if x is not None and int(x) >= 0}
    if valid_stops:
        for idx, token_id in enumerate(trimmed):
            if int(token_id) in valid_stops:
                trimmed = trimmed[:idx]
                break
    if max_new_tokens is not None:
        trimmed = trimmed[:max_new_tokens]
    return trimmed


def _load_mt_bench_records(
    question_begin: Optional[int],
    question_end: Optional[int],
) -> List[Dict]:
    question_file = os.path.join(EAGLE_ROOT, "eagle", "data", "mt_bench", "question.jsonl")
    records = []
    with open(question_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                {
                    "question_id": item.get("question_id", f"mt-{len(records)}"),
                    "turns": item["turns"],
                    "category": item.get("category"),
                    "source": "mt_bench",
                }
            )

    begin = question_begin or 0
    end = question_end if question_end is not None else len(records)
    return records[begin:end]

