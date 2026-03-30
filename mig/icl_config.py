import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_PROFILE: dict[str, str] = {
    "system_prompt": (
        "You are a Python coding assistant. Complete the target function correctly. "
        "Output only Python code with no markdown fences and no explanation."
    ),
    "target_instruction": "Please provide the completion for the target function only.",
    "demo_problem_header": "Problem",
    "demo_solution_header": "Reference Solution",
    "target_problem_header": "Target Problem",
}

DEFAULT_GENERATION_PROFILE: dict[str, Any] = {
    "max_tokens": 2048,
}


def load_eval_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    return {}


def _eval_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("icl_eval")
    if isinstance(section, dict):
        return section
    return config


def normalize_method_name(method: str) -> str:
    m = str(method or "").strip().lower()
    if m == "similarity":
        return "sim"
    return m


def normalize_prompt_task_name(prompt_task: str) -> str:
    raw = str(prompt_task or "").strip().lower()
    if not raw:
        return "default"
    return raw.replace("-", "_").replace(" ", "_")


def method_k_from_config(config: dict[str, Any], method: str) -> int | None:
    kb = _eval_section(config).get("k_by_method")
    if not isinstance(kb, dict):
        return None
    raw = kb.get(normalize_method_name(method))
    if isinstance(raw, int) and raw >= 0:
        return raw
    return None


def detect_prompt_task(row: dict[str, Any]) -> str:
    source = row.get("source")
    dataset = ""
    split = ""
    if isinstance(source, dict):
        dataset = str(source.get("dataset", "")).strip().lower()
        split = str(source.get("split", "")).strip().lower()

    tests = row.get("tests")
    has_input_output = isinstance(tests, dict) and isinstance(tests.get("input_output"), dict)
    has_humaneval_test = isinstance(tests, dict) and bool(str(tests.get("test", "")).strip())

    if dataset == "codeparrot/apps" or has_input_output:
        if split:
            return normalize_prompt_task_name(f"apps_{split}")
        return "apps"

    if dataset in {"openai_humaneval", "openai/humaneval"} or bool(row.get("entry_point")) or has_humaneval_test:
        return "humaneval"

    task_type = str(row.get("task_type", "")).strip()
    if task_type:
        return normalize_prompt_task_name(task_type)
    return "default"


def _resolve_profile(
    profiles: Any,
    profile_name: str,
    *,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(defaults)
    if isinstance(profiles, dict):
        default_profile = profiles.get("default")
        if isinstance(default_profile, dict):
            merged.update(default_profile)
        specific_profile = profiles.get(profile_name)
        if isinstance(specific_profile, dict):
            merged.update(specific_profile)
    return merged


def resolve_prompt_profile(config: dict[str, Any], prompt_task: str) -> tuple[str, dict[str, str]]:
    task_name = normalize_prompt_task_name(prompt_task)
    profiles = _eval_section(config).get("prompt_profiles")
    profile = _resolve_profile(
        profiles,
        task_name,
        defaults=DEFAULT_PROMPT_PROFILE,
    )
    return task_name, {
        "system_prompt": str(profile.get("system_prompt", DEFAULT_PROMPT_PROFILE["system_prompt"])),
        "target_instruction": str(profile.get("target_instruction", DEFAULT_PROMPT_PROFILE["target_instruction"])),
        "demo_problem_header": str(profile.get("demo_problem_header", DEFAULT_PROMPT_PROFILE["demo_problem_header"])),
        "demo_solution_header": str(profile.get("demo_solution_header", DEFAULT_PROMPT_PROFILE["demo_solution_header"])),
        "target_problem_header": str(profile.get("target_problem_header", DEFAULT_PROMPT_PROFILE["target_problem_header"])),
    }


def resolve_generation_profile(config: dict[str, Any], prompt_task: str) -> tuple[str, dict[str, Any]]:
    task_name = normalize_prompt_task_name(prompt_task)
    profiles = _eval_section(config).get("generation_profiles")
    profile = _resolve_profile(
        profiles,
        task_name,
        defaults=DEFAULT_GENERATION_PROFILE,
    )
    max_tokens = profile.get("max_tokens", DEFAULT_GENERATION_PROFILE["max_tokens"])
    if isinstance(max_tokens, float):
        max_tokens = int(max_tokens)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = int(DEFAULT_GENERATION_PROFILE["max_tokens"])
    return task_name, {
        "max_tokens": max_tokens,
    }
