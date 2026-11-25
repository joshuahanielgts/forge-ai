"""Tool registry for ForgeAI. The factory attaches only the tools a brief needs.

Stdlib only, so it runs anywhere. Every tool returns a dict and never raises:
the agent runner gets {"ok": False, "error": ...} instead of a crash.
"""
import ast
import csv
import json
import operator
import urllib.parse
import urllib.request
from datetime import date

TIMEOUT_S = 8


def csv_lookup(path: str, key_col: str, key: str, return_cols: list | None = None) -> dict:
    """Exact, case-insensitive row lookup in a CSV file."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or key_col not in reader.fieldnames:
                return {"ok": False, "error": f"column '{key_col}' not found"}
            matches = []
            for row in reader:
                if str(row.get(key_col, "")).strip().lower() == key.strip().lower():
                    matches.append({k: row[k] for k in (return_cols or row.keys())})
            return {"ok": True, "rows": matches, "count": len(matches)}
    except OSError as e:
        return {"ok": False, "error": str(e)}


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / ** % are allowed")


def calculator(expression: str) -> dict:
    """Arithmetic only. No names, no calls: nothing reaches eval()."""
    try:
        value = _safe_eval(ast.parse(expression, mode="eval"))
        return {"ok": True, "value": value}
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return {"ok": False, "error": str(e)}


def current_date() -> dict:
    return {"ok": True, "date": date.today().isoformat()}


def weather(latitude: float, longitude: float) -> dict:
    """Keyless call to open-meteo. Needs outbound HTTPS; fails cleanly offline."""
    qs = urllib.parse.urlencode({"latitude": latitude, "longitude": longitude,
                                 "current": "temperature_2m,precipitation"})
    try:
        with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{qs}",
                                    timeout=TIMEOUT_S) as r:
            return {"ok": True, "current": json.load(r).get("current", {})}
    except Exception as e:  # network errors are data for the agent, not crashes
        return {"ok": False, "error": f"weather unavailable: {e}"}


# name -> (callable, JSON-schema-style spec shown to the model)
REGISTRY = {
    "csv_lookup": (csv_lookup, {
        "description": "Look up rows in a CSV by exact key value.",
        "parameters": {"path": "str", "key_col": "str", "key": "str", "return_cols": "list[str]?"}}),
    "calculator": (calculator, {
        "description": "Evaluate arithmetic (+ - * / ** %).",
        "parameters": {"expression": "str"}}),
    "current_date": (current_date, {
        "description": "Return today's date (ISO 8601).",
        "parameters": {}}),
    "weather": (weather, {
        "description": "Current temperature and precipitation at a lat/lon.",
        "parameters": {"latitude": "float", "longitude": "float"}}),
}


def attach(names: list[str]) -> dict:
    """Return only the requested tools. Unknown names are an error, not a silent skip."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown tools: {unknown}")
    return {n: REGISTRY[n] for n in names}
