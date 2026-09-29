"""Execute a narrowly transformed, SHA-verified frozen oracle source for 50 questions.

The original source remains byte-for-byte frozen. All substitutions are checked
for exactly one match, and the transformed source SHA is recorded by run.py.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/scratch_lacot_rollout.py"
PIN = "276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc"
DERIVED_PIN = "4e02a2c38994a9e322520cada5bd0a1ca9215ccc0a4a0a7000909298435f7e3e"


def derived_source():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError("Frozen rollout source differs from pinned code hash")
    source = raw.decode("utf-8")
    replacements = (
        (
            "        for sd_ in range(SEEDS):\n            if oracle_draw is not None:",
            "        for sd_ in range(SEEDS):\n"
            "            if oracle_draw is not None and (task, sd_) not in _n64_question_set:\n"
            "                continue\n"
            "            if oracle_draw is not None:",
        ),
        (
            "if len(_rows) != N_TASKS * SEEDS or _rate !=",
            "if len(_rows) != len(_n64_questions) or _rate !=",
        ),
        (
            "    _oracle_result = write_result(_oracle_dst, _oracle_meta, ORACLE_COLLECTOR)",
            "    _oracle_meta['sampled_questions'] = "
            "[{'task': t, 'episode': e} for t, e in _n64_questions]\n"
            "    _oracle_meta['question_seed'] = 20260929\n"
            "    _oracle_result = write_result(_oracle_dst, _oracle_meta, ORACLE_COLLECTOR)",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"N64 transform anchor count {source.count(old)}: {old[:60]}")
        source = source.replace(old, new)
    return source


def source_hash():
    actual = hashlib.sha256(derived_source().encode("utf-8")).hexdigest()
    if actual != DERIVED_PIN:
        raise ValueError("N64 derived source differs from preregistered SHA")
    return actual


def main():
    questions = json.loads(os.environ["LACOT_UCONTRAST2_QUESTIONS_JSON"])
    pairs = [(row["task"], row["episode"]) for row in questions]
    if (len(pairs) != 50 or len(set(pairs)) != 50
            or any(t not in range(1, 6) or e not in range(40) for t, e in pairs)):
        raise ValueError("Invalid N64 question list")
    source_hash()
    source = derived_source()
    scope = {"__name__": "__main__", "__file__": str(SOURCE),
             "_n64_questions": pairs, "_n64_question_set": set(pairs)}
    exec(compile(source, str(SOURCE) + "#ucontrast2-n64", "exec"), scope)


if __name__ == "__main__":
    main()
