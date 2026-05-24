"""SDPO-style helpers built from existing MILES extension points.

This example reuses the on-policy-distillation loss path: a teacher endpoint
scores the sampled response tokens under a reprompted context, and MILES uses
teacher_log_probs - student_log_probs as token-level advantages.

It intentionally does not implement SDPO's full-logit/top-k KL or EMA teacher;
those require actor/teacher forward support inside the training backend.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.utils.types import Sample


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


@lru_cache(maxsize=4)
def _get_tokenizer(hf_checkpoint: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)


async def reward_func(args, sample: Sample, **kwargs):
    """Math reward with lightweight feedback for SDPO reprompting.

    The returned dict works with `--reward-key score`.
    """
    reward = _compute_math_reward(sample)

    feedback = reward.get("feedback", "")
    if not feedback and float(reward.get("score", 0.0)) < _env_float("SDPO_SUCCESS_REWARD_THRESHOLD", 0.5):
        pred = reward.get("pred")
        if pred:
            feedback = f"Your previous answer was parsed as {pred}, but it did not match the expected answer."
        else:
            feedback = "Your previous answer was missing or had the wrong format. Put the final answer in \\boxed{}."
    reward["feedback"] = feedback
    return reward


def _compute_math_reward(sample: Sample) -> dict[str, Any]:
    backend = os.environ.get("SDPO_REWARD_BACKEND", "auto").strip().lower()
    if backend in {"auto", "math_verify", "math-verify"}:
        reward = _compute_math_verify_reward(sample.response, sample.label)
        if reward is not None:
            return reward
        if backend in {"math_verify", "math-verify"}:
            return _incorrect_reward("", "math_verify is not installed or could not verify the answer.")

    try:
        from miles.rollout.rm_hub.math_dapo_utils import compute_score

        reward = compute_score(sample.response, sample.label)
        if not isinstance(reward, dict):
            return {"score": float(reward), "acc": float(reward), "pred": ""}
        return reward
    except Exception as exc:
        return _incorrect_reward("", f"Reward verification failed for label {sample.label!r}: {exc}")


def _compute_math_verify_reward(response: str, label: Any) -> dict[str, Any] | None:
    pred = _extract_boxed(response)
    if pred is None:
        return _incorrect_reward("", "Your answer had the wrong format. Put the final answer in \\boxed{}.")

    gold = str(label).strip()
    if _normalize_answer_text(pred) == _normalize_answer_text(gold):
        return {"score": 1.0, "acc": True, "pred": pred}

    try:
        from math_verify import parse as mv_parse
        from math_verify import verify as mv_verify
    except Exception:
        return None

    try:
        correct = bool(mv_verify(mv_parse(gold), mv_parse(pred)))
    except Exception:
        correct = False

    if correct:
        return {"score": 1.0, "acc": True, "pred": pred}
    return _incorrect_reward(pred, "")


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx + len("\\boxed{")
    depth = 1
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[idx + len("\\boxed{") : i].strip()
        i += 1
    return None


def _normalize_answer_text(value: Any) -> str:
    text = str(value).strip()
    for before, after in (
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\left", ""),
        ("\\right", ""),
        (" ", ""),
        (",", ""),
        ("$", ""),
    ):
        text = text.replace(before, after)
    return text


def _incorrect_reward(pred: str, feedback: str) -> dict[str, Any]:
    return {
        "score": -1.0,
        "acc": False,
        "pred": pred,
        "feedback": feedback,
    }


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Attach teacher logprobs computed from SDPO-style reprompts."""
    if not samples:
        return [], []

    scores = [_score(args, sample) for sample in samples]
    groups = _group_samples(samples)
    score_by_identity = {id(sample): score for sample, score in zip(samples, scores, strict=False)}
    solutions_by_group = _collect_solutions(groups, score_by_identity)

    teacher_url = _teacher_url(args)
    tokenizer = _get_tokenizer(args.hf_checkpoint)
    payloads: list[dict[str, Any] | None] = []

    for sample in samples:
        teacher_prompt = _build_teacher_prompt(args, tokenizer, sample, solutions_by_group)
        if teacher_prompt is None or sample.response_length == 0:
            payloads.append(None)
            sample.teacher_log_probs = _zeros(sample.response_length)
            sample.loss_mask = [0] * sample.response_length
            continue

        response_ids = sample.tokens[-sample.response_length :]
        input_ids = teacher_prompt + response_ids
        payloads.append(
            {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 0,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            }
        )

    teacher_responses = _fetch_teacher_logprobs(teacher_url, payloads)
    for sample, payload, response in zip(samples, payloads, teacher_responses, strict=False):
        if payload is None:
            continue
        sample.teacher_log_probs = _extract_response_logprobs(response, sample.response_length)

    return scores, scores


def _score(args, sample: Sample) -> float:
    reward = sample.reward
    if isinstance(reward, dict):
        key = args.reward_key or "score"
        return float(reward.get(key, 0.0))
    return float(reward)


def _sample_group_key(sample: Sample) -> int:
    if sample.group_index is not None:
        return sample.group_index
    if sample.index is not None:
        return sample.index
    return id(sample)


def _group_samples(samples: list[Sample]) -> dict[int, list[Sample]]:
    groups: dict[int, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(_sample_group_key(sample), []).append(sample)
    return groups


def _collect_solutions(
    groups: dict[int, list[Sample]], score_by_identity: dict[int, float]
) -> dict[int, list[tuple[int, int | None, str]]]:
    threshold = _env_float("SDPO_SUCCESS_REWARD_THRESHOLD", 0.5)
    remove_thinking = _env_bool("SDPO_REMOVE_THINKING_FROM_DEMONSTRATION", True)

    solutions_by_group: dict[int, list[tuple[int, int | None, str]]] = {}
    for group_index, group in groups.items():
        solutions = []
        for sample in group:
            if score_by_identity.get(id(sample), 0.0) < threshold:
                continue
            solution = sample.response
            if remove_thinking:
                solution = _THINK_RE.sub("", solution).strip()
            if solution:
                solutions.append((id(sample), sample.index, solution))
        solutions_by_group[group_index] = solutions
    return solutions_by_group


def _build_teacher_prompt(
    args,
    tokenizer,
    sample: Sample,
    solutions_by_group: dict[int, list[tuple[int, int | None, str]]],
) -> list[int] | None:
    group_index = _sample_group_key(sample)
    solution = _pick_solution(sample, solutions_by_group.get(group_index, []))
    feedback = _feedback(sample)

    include_feedback = _env_bool("SDPO_INCLUDE_ENVIRONMENT_FEEDBACK", True)
    feedback_only_without_solution = _env_bool("SDPO_ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION", True)
    use_feedback = bool(feedback) and include_feedback and (solution is None or not feedback_only_without_solution)

    if solution is None and not use_feedback:
        return None

    solution_section = ""
    if solution is not None:
        solution_section = f"\n\nCorrect solution:\n\n{solution}"

    feedback_section = ""
    if use_feedback:
        feedback_section = f"\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback}"

    reprompt_text = (
        f"{_prompt_text(sample)}{solution_section}{feedback_section}\n\n"
        "Correctly solve the original question."
    )

    prompt_ids = _encode_teacher_prompt(args, tokenizer, sample, reprompt_text)
    max_reprompt_len = _env_int("SDPO_MAX_REPROMPT_LEN", 10240)
    if len(prompt_ids) > max_reprompt_len:
        truncation = os.environ.get("SDPO_REPROMPT_TRUNCATION", "right").strip().lower()
        if truncation == "right":
            prompt_ids = prompt_ids[:max_reprompt_len]
        elif truncation == "left":
            prompt_ids = prompt_ids[-max_reprompt_len:]
        elif truncation == "error":
            raise ValueError(f"Teacher prompt has {len(prompt_ids)} tokens, above SDPO_MAX_REPROMPT_LEN")
        else:
            raise ValueError(f"Unknown SDPO_REPROMPT_TRUNCATION={truncation}")
    return prompt_ids


def _pick_solution(sample: Sample, group_solutions: list[tuple[int, int | None, str]]) -> str | None:
    dont_self_success = _env_bool("SDPO_DONT_REPROMPT_ON_SELF_SUCCESS", True)
    for sample_id, sample_index, solution in group_solutions:
        if dont_self_success and (sample_id == id(sample) or (sample_index is not None and sample_index == sample.index)):
            continue
        return solution
    if not dont_self_success and group_solutions:
        return group_solutions[0][2]
    return None


def _feedback(sample: Sample) -> str:
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    metadata_feedback = sample.metadata.get("feedback") if sample.metadata else None
    feedback = reward.get("feedback") or metadata_feedback
    return str(feedback).strip() if feedback else ""


def _prompt_text(sample: Sample) -> str:
    metadata_prompt = _metadata_prompt_text(sample)
    if metadata_prompt is not None:
        return metadata_prompt
    if isinstance(sample.prompt, list) and sample.prompt:
        last = sample.prompt[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(sample.prompt)


def _metadata_prompt_text(sample: Sample) -> str | None:
    if sample.metadata:
        for key in ("raw_prompt", "question", "problem"):
            if key in sample.metadata and sample.metadata[key]:
                return str(sample.metadata[key])
    return None


def _encode_teacher_prompt(args, tokenizer, sample: Sample, reprompt_text: str) -> list[int]:
    apply_chat_template = bool(getattr(args, "apply_chat_template", False))
    has_metadata_prompt = _metadata_prompt_text(sample) is not None
    prompt_is_already_formatted = (
        isinstance(sample.prompt, str) and ("<|im_start|>" in sample.prompt) and not has_metadata_prompt
    )
    if apply_chat_template and not prompt_is_already_formatted:
        kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": reprompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        return list(ids)
    return tokenizer.encode(reprompt_text, add_special_tokens=False)


def _teacher_url(args) -> str:
    rm_url = getattr(args, "rm_url", None)
    if rm_url:
        return rm_url
    router_ip = getattr(args, "sglang_router_ip", None)
    router_port = getattr(args, "sglang_router_port", None)
    if router_ip is None or router_port is None:
        raise ValueError("Set --rm-url or let MILES initialize --sglang-router-ip/--sglang-router-port.")
    return f"http://{router_ip}:{router_port}/generate"


def _fetch_teacher_logprobs(url: str, payloads: list[dict[str, Any] | None]) -> list[dict[str, Any] | None]:
    return [_post_json(url, payload, i) if payload is not None else None for i, payload in enumerate(payloads)]


def _post_json(url: str, payload: dict[str, Any], index: int):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    timeout = _env_int("SDPO_TEACHER_TIMEOUT_SECS", 14400)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Teacher request {index} failed with HTTP {exc.code}: {detail[:1000]}") from exc


def _zeros(length: int):
    import torch

    return torch.zeros(length, dtype=torch.float32)


def _extract_response_logprobs(response: dict[str, Any] | None, response_length: int):
    import torch

    if response is None:
        return torch.zeros(response_length, dtype=torch.float32)
    token_logprobs = response["meta_info"]["input_token_logprobs"][1:]
    values = [float(item[0]) for item in token_logprobs]
    if len(values) < response_length:
        raise ValueError(f"Teacher returned {len(values)} logprobs for response length {response_length}")
    return torch.tensor(values[-response_length:], dtype=torch.float32)
