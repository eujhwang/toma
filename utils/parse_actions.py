# parser_actions.py
from __future__ import annotations
import json, ast, re
from dataclasses import dataclass
from typing import Optional, List, Dict

ALLOWED_ACTIONS = {
    "none",
    "speak",
    "non-verbal communication",
    "action",
}

# common aliases / typos mapped to allowed set
ACTION_ALIASES = {
    "say": "speak",
    "speak": "speak",
    "talk": "speak",
    "reply": "speak",
    "respond": "speak",

    "leave": "none",
    "end": "none",
    "stop": "none",
    "quit": "none",
    "none": "none",

    "nonverbal": "non-verbal communication",
    "non verbal": "non-verbal communication",
    "non-verbal": "non-verbal communication",
    "nonverbal communication": "non-verbal communication",
    "non verbal communication": "non-verbal communication",
    "non-verbal communication": "non-verbal communication",

    "action": "action",
    "act": "action",
}


@dataclass
class ParsedAction:
    action_type: str
    argument: str
    raw_json: str
    repaired: bool
    errors: List[str]
    # NEW: optional mental state, empty string when not provided
    mental_state: str = ""


def _strip_code_fences(text: str) -> str:
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def _find_first_json_object(text: str) -> Optional[str]:
    """
    Returns the substring of the first balanced {...} JSON object.
    Ignores braces inside double-quoted strings.
    """
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return text[start:i + 1]
    return None


def _de_smart_quotes(s: str) -> str:
    return (
        s.replace("“", '"').replace("”", '"').replace("‟", '"')
         .replace("’", "'").replace("‚", "'").replace("‘", "'")
    )


def _remove_trailing_commas(s: str) -> str:
    # remove trailing commas before } or ]
    return re.sub(r",(\s*[}\]])", r"\1", s)


def _json_loads_lenient(s: str) -> Optional[Dict]:
    """Try json.loads with a few safe normalizations."""
    try:
        return json.loads(s)
    except Exception:
        pass

    s = _de_smart_quotes(s)
    s = _remove_trailing_commas(s)

    # If it looks like python-ish dict with single quotes, try converting:
    if '"' not in s and "'" in s:
        s_try = s.replace("'", '"')
        try:
            return json.loads(s_try)
        except Exception:
            pass

    # Last resort: literal_eval on a python-ish dict
    # convert JSON booleans/null to Python
    s_py = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
    s_py = re.sub(r"\bfalse\b", "False", s_py, flags=re.IGNORECASE)
    s_py = re.sub(r"\bnull\b", "None", s_py, flags=re.IGNORECASE)
    try:
        obj = ast.literal_eval(s_py)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def _normalize_action_type(a: Optional[str]) -> str:
    if not a:
        return "speak"
    a = a.strip().lower()
    a = ACTION_ALIASES.get(a, a)
    if a not in ALLOWED_ACTIONS:
        # default to "speak" if it's unrecognized
        a = "speak"
    return a


def parse_action(output_text: str) -> ParsedAction:
    """
    Parse model output that should contain a single JSON object:
      {"action_type":"...", "argument":"...", "mental_state":"...?"}
    Returns a normalized ParsedAction (with safe defaults if needed).
    - "mental_state" is optional; when absent, it's returned as an empty string.
    """
    errors: List[str] = []
    txt = _strip_code_fences(output_text)

    json_str = _find_first_json_object(txt)
    if json_str is None:
        errors.append("no_json_object_found")
        # try whole text as-is
        json_str = txt

    data = _json_loads_lenient(json_str)
    repaired = False

    if data is None:
        errors.append("json_parse_failed")
        # fallback minimal result
        return ParsedAction(
            action_type="speak",
            argument=txt.strip(),
            raw_json=json_str,
            repaired=True,
            errors=errors,
            mental_state="",  # nothing recoverable
        )

    # normalize fields
    a_type = _normalize_action_type(str(data.get("action_type", "speak")))
    arg = data.get("argument", "")
    if not isinstance(arg, str):
        arg = str(arg)
    arg = arg.replace("\n", " ").strip()

    # optional mental_state
    mental_state = data.get("mental_state", "")
    if not isinstance(mental_state, str):
        mental_state = str(mental_state)
    # keep it single-line to play nice with logs/parsers
    mental_state = mental_state.replace("\n", " ").strip()

    if a_type == "none":
        arg = ""  # enforce empty argument on "none"

    # basic length clamp (optional) for argument only (kept from original)
    # if len(arg) > 200:
    #     arg = arg[:200].rstrip()
    #     repaired = True

    return ParsedAction(
        action_type=a_type,
        argument=arg,
        raw_json=json_str,
        repaired=repaired or ("action_type" not in data) or ("argument" not in data),
        errors=errors,
        mental_state=mental_state,
    )

def is_valid_action(parsed: ParsedAction, *, strict: bool = True) -> bool:
    """
    strict=True -> reject if parsing required 'repairs' even if we recovered.
    We also reject if JSON wasn't found / failed, or if argument is empty
    for non-'none' actions.
    """
    if any(e in parsed.errors for e in ("json_parse_failed", "no_json_object_found")):
        return False
    if strict and parsed.repaired:
        return False
    if parsed.action_type != "none" and not parsed.argument:
        return False
    return True