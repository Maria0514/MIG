from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer

_JSON_TAG_FIELD_RE = re.compile(r"""["']tag["']\s*:\s*["']([^"']+)["']""", re.IGNORECASE)
_TEXT_TAG_FIELD_RE = re.compile(r"""^["']?tag["']?\s*:\s*["']?(.*?)["']?$""", re.IGNORECASE)
_TAG_ALIAS_MAP = {
    "conditional": "conditional statement",
    "conditionals": "conditional statement",
    "conditional statements": "conditional statement",
    "input output": "input/output",
    "input and output": "input/output",
    "i/o": "input/output",
    "io": "input/output",
    "problem solving": "problem solve",
    "problem-solving": "problem solve",
    "programming logic": "program logic",
}


def _normalize_tag_text(tag: str) -> str:
    """Normalize tag text to match repository tag style."""
    tag = tag.strip().lower()
    tag = tag.replace("_", " ").replace("-", " ")
    tag = re.sub(r"\s+", " ", tag)
    return _TAG_ALIAS_MAP.get(tag, tag)


def _dedupe_preserve_order(tags: Sequence[str]) -> list[str]:
    uniq: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        uniq.append(tag)
        seen.add(tag)
    return uniq


def _extract_json_style_tags(raw_text: str) -> list[str]:
    matches = _JSON_TAG_FIELD_RE.findall(raw_text or "")
    tags = [_normalize_tag_text(m) for m in matches if _normalize_tag_text(m)]
    return _dedupe_preserve_order(tags)


def _build_valid_tag_index(valid_tags: Sequence[str]) -> dict[str, str]:
    """Build normalized->canonical mapping for valid tag projection."""
    return {_normalize_tag_text(t): t for t in valid_tags}


def parse_instag_output(raw_text: str) -> list[str]:
    """Parse model output into a normalized tag list."""
    if not raw_text:
        return []

    json_style_tags = _extract_json_style_tags(raw_text)
    if json_style_tags:
        return json_style_tags

    text = raw_text.strip()
    # Common wrappers in LLM outputs.
    text = text.replace("Tags:", "").replace("tags:", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace(";", ",")
    text = text.replace("\n", ",")

    tags: list[str] = []
    for piece in text.split(","):
        tag = piece.strip().strip('"').strip("'").strip()
        if not tag:
            continue
        if "explanation" in tag.lower():
            continue
        tag_field_match = _TEXT_TAG_FIELD_RE.match(tag)
        if tag_field_match:
            tag = tag_field_match.group(1).strip()
        tag = _normalize_tag_text(tag)
        if tag:
            tags.append(tag)

    return _dedupe_preserve_order(tags)


def project_tags_to_valid_set(tags: Sequence[str], valid_tags: Sequence[str]) -> list[str]:
    """Project raw tags to current valid tag space."""
    if not tags:
        return []

    valid_index = _build_valid_tag_index(valid_tags)
    projected: list[str] = []
    seen: set[str] = set()
    for t in tags:
        key = _normalize_tag_text(t)
        canonical = valid_index.get(key)
        if canonical is None or canonical in seen:
            continue
        projected.append(canonical)
        seen.add(canonical)
    return projected


def build_instag_prompt(query_text: str) -> str:
    """Vicuna-style prompt for InsTagger query tagging."""
    query_text = (query_text or "").strip()
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "USER: Tag the following instruction query with concise semantic intent tags. "
        "Output tags only, as a comma-separated list in lowercase.\n"
        f"Query: {query_text}\n"
        "ASSISTANT:"
    )


@dataclass
class TaggedQuery:
    query_id: str
    query_text: str
    raw_tags: list[str]
    valid_tags: list[str]
    raw_generation: str


class HFInsTagger:
    """InsTagger backend using local Hugging Face checkpoint."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device_map: str = "auto",
        torch_dtype: str | None = "auto",
        max_new_tokens: int = 96,
        do_sample: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        model_kwargs: dict[str, Any] = {"device_map": device_map}
        if torch_dtype is not None and torch_dtype != "auto":
            import torch
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        elif torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def generate(self, prompts: Sequence[str]) -> list[str]:
        if not prompts:
            return []

        inputs = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        model_device = self.model.device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=0.0 if not self.do_sample else 0.7,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[:, input_len:]
        texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]


class OpenAICompatInsTagger:
    """InsTagger backend via OpenAI-compatible chat/completions endpoint."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        api_mode: str = "auto",
        max_tokens: int = 96,
        timeout_sec: int = 60,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.api_mode = api_mode
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url=url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _generate_chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        res = self._post_json("/chat/completions", payload)
        return str(
            res.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        ).strip()

    def _generate_completion(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        res = self._post_json("/completions", payload)
        return str(res.get("choices", [{}])[0].get("text", "")).strip()

    def generate(self, prompts: Sequence[str]) -> list[str]:
        out: list[str] = []
        for prompt in prompts:
            if self.api_mode == "chat":
                out.append(self._generate_chat(prompt))
                continue
            if self.api_mode == "completion":
                out.append(self._generate_completion(prompt))
                continue

            # auto: try chat first, then fallback to completion when template is missing
            try:
                out.append(self._generate_chat(prompt))
            except Exception as e:
                msg = str(e).lower()
                if "chat template" in msg or "badrequesterror" in msg or "400" in msg:
                    out.append(self._generate_completion(prompt))
                else:
                    raise
        return out


def tag_queries(
    query_records: Iterable[dict[str, Any]],
    *,
    valid_tags: Sequence[str],
    tagger: HFInsTagger | OpenAICompatInsTagger,
    text_key: str = "prompt",
    id_key: str = "id",
    batch_size: int = 8,
) -> list[TaggedQuery]:
    """Tag query records and project tags to valid tag space."""
    records = list(query_records)
    if not records:
        return []

    prompts = [build_instag_prompt(str(r.get(text_key, "") or "")) for r in records]

    generations: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        generations.extend(tagger.generate(batch))

    results: list[TaggedQuery] = []
    for rec, gen in zip(records, generations):
        raw_tags = parse_instag_output(gen)
        valid = project_tags_to_valid_set(raw_tags, valid_tags)
        results.append(
            TaggedQuery(
                query_id=str(rec.get(id_key, "")),
                query_text=str(rec.get(text_key, "") or ""),
                raw_tags=raw_tags,
                valid_tags=valid,
                raw_generation=gen,
            )
        )
    return results
