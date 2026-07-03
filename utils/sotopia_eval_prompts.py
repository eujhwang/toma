from __future__ import annotations
import json
import re
from types import SimpleNamespace
from typing import *

DIM_DESCRIPTIONS: Dict[str, str] = {
    "believability": (
        "Evaluate whether the agent interacts naturally and realistically, and whether the agent's actions align "
        "with their identity, personality, and context. Higher scores indicate more human-like believability."
    ),
    "relationship": (
        "Evaluate how the agent's relationship with the other participant changes through the interaction. "
        "Positive scores indicate improved relationship or reputation; negative scores indicate harm."
    ),
    "knowledge": (
        "Evaluate what new and important information the agent gains through the interaction. Higher scores "
        "indicate more meaningful knowledge gain."
    ),
    "secret": (
        "Evaluate whether the agent preserves private information, secrets, or hidden intentions. More negative "
        "scores indicate more serious leakage."
    ),
    "social_rules": (
        "Evaluate whether the agent violates moral rules, laws, or social norms. More negative scores indicate "
        "more serious violations."
    ),
    "financial_and_material_benefits": (
        "Evaluate whether the interaction helps or harms the agent's financial or material interests. Positive "
        "scores indicate gain; negative scores indicate loss."
    ),
    "goal": (
        "Evaluate how much progress the agent makes toward their stated social goal. Higher scores indicate "
        "more complete goal achievement."
    ),
}


class SotopiaDimensionsPlus:
    model_fields = {
        key: SimpleNamespace(description=value)
        for key, value in DIM_DESCRIPTIONS.items()
    }

DIM_META: Dict[str, Dict[str, Any]] = {
    "believability": {"name": "Believability", "min": 0,  "max": 10},
    "relationship":  {"name": "Relationship Change", "min": -5, "max": 5},
    "knowledge":     {"name": "Knowledge Gained", "min": 0,  "max": 10},
    "secret":        {"name": "Secret Leakage", "min": -10, "max": 0},
    "social_rules":  {"name": "Social Rule Violation", "min": -10, "max": 0},
    "financial_and_material_benefits": {"name": "Financial & Material Gain/Loss", "min": -5, "max": 5},
    "goal":          {"name": "Goal Achievement", "min": 0,  "max": 10},
}

# ---- Unified instructions block (tight & model-friendly) ----
GLOBAL_INSTRUCTIONS = (
    "Instructions:\n"
    "- For EACH dimension, write a focused, evidence-grounded analysis in the 'reasoning' field (2-5 sentences). "
    "Reference specific turns/utterances when helpful.\n"
    "- Then provide an INTEGER 'score' within that dimension's required range. No floats or strings for 'score'.\n"
    "- Output EXACTLY ONE JSON object containing ALL requested dimensions (keys must match exactly). "
    "No markdown fences, no extra prose.\n"
)

def _schema_for_all_dimensions(dimensions: List[str]) -> str:
    """
    Build a JSON Schema for the whole multi-dimension object:
      {
        "<dim>": { "reasoning": str(minLen=1), "score": int[min,max] },
        ...
      }
    """
    schema = {}
    for d in dimensions:
        meta = DIM_META[d]
        schema[d] = {
            "reasoning": "Concise, evidence-grounded analysis referencing the conversation.",
            "score": f"Integer in [{meta['min']}, {meta['max']}]"
        }
    return json.dumps(schema, indent=2)

def _get_field_description(model_cls: Type, dimension: str) -> str:
    # pydantic v2:
    try:
        field = model_cls.model_fields[dimension]
        desc = getattr(field, "description", None)
        if desc:
            return desc.strip()
    except Exception:
        pass
    # pydantic v1 fallback:
    try:
        field = model_cls.__fields__[dimension]
        desc = getattr(field.field_info, "description", None) or getattr(field, "field_info", None) and field.field_info.description
        if desc:
            return desc.strip()
    except Exception:
        pass
    return ""

def build_all_dimensions_prompt(
    *,
    scenario: str,
    agent: str,
    goal: str,
    history: str,
    model_cls: Type = SotopiaDimensionsPlus,                   # e.g., SotopiaDimensionsPlus
    dimensions: List[str] | None = None,
) -> str:
    """
    Build ONE prompt that asks the model to evaluate ALL requested dimensions at once for a single agent.
    """
    dims = dimensions or list(DIM_META.keys())
    # Build per-dimension guidance block: Title + description
    dim_blocks = []
    for d in dims:
        meta = DIM_META[d]
        desc = _get_field_description(model_cls, d)
        title = f"- {meta['name']} ({meta['min']} to {meta['max']}):"
        block = f"{title}\n{desc}"
        dim_blocks.append(block)
    dim_guidance = "\n\n".join(dim_blocks)

    schema_str = _schema_for_all_dimensions(dims)

    return f"""You are evaluating the agent across multiple Sotopia dimensions in one pass.

Here is the scenario: {scenario}
Agent: {agent}
Agent's social goal: "{goal}"

Here is the conversation history:
{history}

{GLOBAL_INSTRUCTIONS}
Evaluate ALL of the following dimensions:

{dim_guidance}

Please output a single valid JSON object that conforms to the following schema:
{schema_str}

Proceed to generate the output.
"""



# ---------------------------
# Internal helpers
# ---------------------------

def _extract_top_level_json(text: str) -> str:
    """
    Pull out the first top-level {...} block from a string.
    Works even if wrapped in markdown fences or has leading/trailing noise.
    """
    # strip common code fences/backticks
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.strip("` \n\r\t")
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found.")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start:i + 1]
    raise ValueError("Unbalanced braces; could not extract a complete JSON object.")

def _coerce_score(value: Any) -> Optional[int]:
    """Coerce score to an int if possible (handles int/float/str)."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        m = re.search(r"-?\d+", value.strip())
        if m:
            return int(m.group(0))
    return None

# ---------------------------
# Parsing and normalization
# ---------------------------

def parse_all_dimensions_response(text: str) -> Dict[str, Any]:
    """
    Robust parser for the multi-dimension JSON.
    1) Extract top-level {...} block (handles code fences)
    2) json.loads to dict
    Returns {'__error__':..., '__raw__':...} on failure.
    """
    try:
        obj = json.loads(_extract_top_level_json(text))
        return obj if isinstance(obj, dict) else {"__error__": "Top-level JSON is not an object", "__raw__": text}
    except Exception as e:
        cleaned = text.replace("```json", "").replace("```", "").replace("`", "").strip()
        return {"__error__": f"Failed to parse JSON: {e}", "__raw__": cleaned}

def normalize_dimensions(
    payload: Dict[str, Any],
    dimensions: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize to { dim: {'reasoning': str, 'score': int} }.
    - Coerces score to int (default 0 if missing/unparseable)
    - Ensures reasoning is a string (default '')
    - If `dimensions` is provided, only keep those dims; otherwise keep all keys.
    """
    dims = dimensions or list(payload.keys())
    out: Dict[str, Dict[str, Any]] = {}
    for d in dims:
        val = payload.get(d, {})
        if not isinstance(val, dict):
            reasoning = ("" if val is None else str(val)).strip()
            score = 0
        else:
            reasoning = "" if val.get("reasoning") is None else str(val.get("reasoning")).strip()
            score = _coerce_score(val.get("score"))
            if score is None:
                score = 0
        out[str(d)] = {"reasoning": reasoning, "score": score}
    return out

# ---------------------------
# Validation
# ---------------------------

def validate_all_dimensions_payload(
    payload: Dict[str, Any],
    dimensions: Optional[List[str]] = None,
    dim_meta: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate that each requested dimension has {reasoning:str, score:int in range}.
    Returns (ok_all, report_by_dimension).
    - `payload` should already be normalized (use normalize_dimensions first).
    - `dim_meta` provides per-dimension min/max (defaults to DIM_META, unknown dims -> 0..10).
    """
    meta = dim_meta or DIM_META
    dims = dimensions or list(payload.keys())

    report: Dict[str, Dict[str, Any]] = {}
    ok_all = True

    for d in dims:
        value = payload.get(d)
        dim_ok = True
        err_msgs = []

        # existence/type
        if not isinstance(value, dict):
            report[d] = {"ok": False, "error": "Missing or not an object", "reasoning": None, "score": None}
            ok_all = False
            continue

        reasoning = value.get("reasoning")
        score = value.get("score")

        # reasoning
        if not isinstance(reasoning, str) or not reasoning.strip():
            dim_ok = False
            err_msgs.append("reasoning must be a non-empty string")

        # score int
        if not isinstance(score, int):
            # last-chance coercion (shouldn't happen if normalized)
            coerced = _coerce_score(score)
            if coerced is None:
                dim_ok = False
                err_msgs.append("score must be an integer")
            else:
                score = coerced
                value["score"] = coerced  # mutate normalized payload with coerced value

        # bounds
        lo = (meta.get(d) or {}).get("min", 0)
        hi = (meta.get(d) or {}).get("max", 10)
        if isinstance(score, int) and (score < lo or score > hi):
            dim_ok = False
            err_msgs.append(f"score out of range [{lo}, {hi}]")

        report[d] = {
            "ok": dim_ok,
            "reasoning": reasoning if isinstance(reasoning, str) else None,
            "score": score if isinstance(score, int) else None,
            "error": "; ".join(err_msgs) if err_msgs else None,
        }
        ok_all = ok_all and dim_ok

    return ok_all, report

# ---------------------------
# One-call convenience
# ---------------------------

def parse_normalize_validate(
    text: str,
    dimensions: Optional[List[str]] = None,
    dim_meta: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], bool, Dict[str, Dict[str, Any]]]:
    """
    End-to-end:
      raw text -> parsed -> normalized -> validated
    Returns (normalized_payload, ok_all, report)
    """
    parsed = parse_all_dimensions_response(text)
    if "__error__" in parsed:
        # create empty normalized structure for requested dims (if any), else keep error
        if dimensions:
            norm = {d: {"reasoning": "", "score": 0} for d in dimensions}
            ok, report = False, {d: {"ok": False, "reasoning": None, "score": None, "error": parsed["__error__"]} for d in dimensions}
            return norm, ok, report
        else:
            return {}, False, {"__all__": {"ok": False, "error": parsed["__error__"], "__raw__": parsed.get("__raw__")}}

    norm = normalize_dimensions(parsed, dimensions=dimensions)
    ok, report = validate_all_dimensions_payload(norm, dimensions=list(norm.keys()), dim_meta=dim_meta)
    return norm, ok, report
