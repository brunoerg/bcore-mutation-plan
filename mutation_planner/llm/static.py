"""Deterministic baseline "provider": builds a plan from the planning view with fixed rules.

No model, no network, no subprocess. It receives exactly the same view an LLM would and its
output goes through the same validator, so its plans are directly comparable to LLM plans.
It makes no semantic judgement: it cannot tell a behavioural change from a mechanical one.

Rules (version ``STATIC_RULES_VERSION``):

* Eligible file: a production file with at least one detected changed function.
  Files whose changes are all outside functions (declarations, globals, macros) are skipped.
* ``mutate`` is true iff at least one file is eligible.
* ``functions``: every detected changed function; ``changed_lines``: their changed ranges.
* ``priority``:
    - high   if the path is in a consensus/validation-critical area (``CRITICAL_AREAS``);
    - medium if >= ``MEDIUM_MIN_CHANGED_LINES`` changed lines or >= ``MEDIUM_MIN_FUNCTIONS``
      changed functions, or a changed function is an RPC handler;
    - low    otherwise.
* Tests, from the ranked candidates, up to the per-file budget:
    1. direct candidates (confidence ``high``);
    2. call_path / rpc_call_path candidates (confidence ``medium``);
    3. module_area candidates (confidence ``low``), at most ``MAX_MODULE_AREA`` and only
       when steps 1-2 found fewer than ``MODULE_AREA_FILL_BELOW`` tests;
    name_similarity-only candidates are never selected. A suite is skipped when one of its
    cases is already selected (and vice versa).
"""

from __future__ import annotations

from typing import Any

STATIC_RULES_VERSION = "1"

CRITICAL_AREAS = (
    "src/consensus/",
    "src/script/",
    "src/primitives/",
    "src/kernel/",
    "src/policy/",
    "src/validation.",
    "src/validationinterface.",
    "src/net_processing.",
    "src/txmempool.",
    "src/coins.",
    "src/pow.",
    "src/chain.",
    "src/wallet/coinselection.",
    "src/wallet/spend.",
    "src/wallet/feebumper.",
)
MEDIUM_MIN_CHANGED_LINES = 20
MEDIUM_MIN_FUNCTIONS = 3
MAX_MODULE_AREA = 2
MODULE_AREA_FILL_BELOW = 3

_CALL_PATHS = {"call_path", "rpc_call_path"}


def _n_lines(ranges: list[dict[str, int]]) -> int:
    return sum(r["end"] - r["start"] + 1 for r in ranges)


def _priority(path: str, functions: list[dict[str, Any]]) -> tuple[str, str]:
    if path.startswith(CRITICAL_AREAS):
        return "high", "file is in a consensus/validation-critical area"
    n = sum(_n_lines(fn.get("changed_lines", [])) for fn in functions)
    if any(fn.get("rpc_name") for fn in functions):
        return "medium", "a changed function is an RPC handler"
    if n >= MEDIUM_MIN_CHANGED_LINES or len(functions) >= MEDIUM_MIN_FUNCTIONS:
        return "medium", f"{n} changed lines in {len(functions)} function(s)"
    return "low", f"small change ({n} changed lines in {len(functions)} function(s))"


def _relationships(cand: dict[str, Any]) -> list[str]:
    rels = [e["relationship"] for e in cand["evidence"]] + cand.get("unlisted_relationships", [])
    return list(dict.fromkeys(rels))


def _overlaps(selected: set[str], tid: str) -> bool:
    return any(s.startswith(tid + "/") or tid.startswith(s + "/") for s in selected)


def _select_tests(candidates: list[dict[str, Any]], budget: int) -> dict[str, list[dict[str, Any]]]:
    tiers: list[tuple[str, list[dict[str, Any]]]] = [
        ("high", [c for c in candidates if c["strength"] == "direct"]),
        ("medium", [c for c in candidates if c["strength"] == "inferred" and _CALL_PATHS & set(_relationships(c))]),
    ]
    out: dict[str, list[dict[str, Any]]] = {"unit": [], "functional": [], "fuzz": []}
    selected: set[str] = set()

    def take(conf: str, cand: dict[str, Any]) -> bool:
        tid = cand["test_id"]
        if len(selected) >= budget or tid in selected or _overlaps(selected, tid) or not cand["evidence"]:
            return False
        selected.add(tid)
        out[tid.split(":", 1)[0]].append({
            "id": tid,
            "reason": f"static baseline: {', '.join(_relationships(cand))} (score {cand['score']})",
            "confidence": conf,
            "evidence": [e["id"] for e in cand["evidence"]],
        })
        return True

    for conf, tier in tiers:
        for cand in tier:  # candidates arrive ranked by score
            take(conf, cand)
    if len(selected) < MODULE_AREA_FILL_BELOW:
        area = [c for c in candidates if c["strength"] == "inferred" and _relationships(c) == ["module_area"]]
        added = 0
        for cand in area:
            if added >= MAX_MODULE_AREA:
                break
            added += take("low", cand)
    return out


def static_plan(view: dict[str, Any]) -> dict[str, Any]:
    budget = int(view.get("budget", {}).get("max_tests_per_file") or 10)
    candidates = view.get("candidate_tests", {})
    files: list[dict[str, Any]] = []
    skipped: list[str] = []
    for src in view.get("production_files", []):
        fns = src.get("functions", [])
        if not fns:
            skipped.append(src["path"])
            continue
        priority, why = _priority(src["path"], fns)
        tests = _select_tests(candidates.get(src["path"], []), budget)
        files.append({
            "path": src["path"],
            "priority": priority,
            "reason": f"static baseline: {len(fns)} changed function(s); priority {priority}: {why}",
            "functions": [fn["name"] for fn in fns],
            "changed_lines": [r for fn in fns for r in fn.get("changed_lines", [])],
            "tests": tests,
            "uncertainty": "Deterministic rules only: whether the change is behavioural or mechanical was not judged.",
        })
    warnings = [f"static baseline plan (rules v{STATIC_RULES_VERSION}); no LLM judgement was applied"]
    if skipped:
        warnings.append("static baseline skipped production files with no detected changed function: " + ", ".join(skipped))
    if files:
        decision = {"mutate": True, "reason": f"{len(files)} production file(s) have changed functions."}
    elif view.get("production_files"):
        decision = {"mutate": False, "reason": "Production files changed, but no changed function was detected."}
    else:
        decision = {"mutate": False, "reason": "No production code changed."}
    return {"decision": decision, "files": files, "warnings": warnings}


class StaticProvider:
    name = "static"

    def __init__(self, model: str | None = None):
        self.model = None  # no model; --model is ignored

    def generate_plan(self, system_prompt: str, context: dict[str, Any], output_schema: dict[str, Any]) -> dict[str, Any]:
        return static_plan(context)
