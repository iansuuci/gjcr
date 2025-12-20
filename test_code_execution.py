#!/usr/bin/env python3
"""Test script to verify code execution functions work correctly."""

import re
import io
import math
from typing import Optional
from contextlib import redirect_stdout

# Copy the relevant functions from grok_experiment.py
def _coerce_int(x) -> Optional[int]:
    """Convert value to integer if possible."""
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            r = round(x)
            return r if abs(x - r) < 1e-9 else None
        return int(str(x).strip())
    except Exception:
        return None

def _normalize_aime_int(x: Optional[int]) -> Optional[int]:
    """Clamp to valid AIME range [0, 999]."""
    if x is None:
        return None
    try:
        xi = int(x)
        if 0 <= xi <= 999:
            return xi
        return None
    except Exception:
        return None

def extract_code_block(text: str) -> Optional[str]:
    """Extract Python code block from text."""
    # Try custom markers first
    m = re.search(r"PYCODE:\s*\n(.*?)\nENDPYCODE", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try standard markdown
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: collect code-like lines
    kws = ("import", "def ", "for ", "while ", "if ", "=", "print", "return")
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if any(k in s for k in kws) or (lines and s):
            lines.append(ln)
    code = "\n".join(lines).strip()
    return code if len(code) >= 4 else None

def execute_python_code(code_text: str) -> Optional[int]:
    """Execute Python code and extract integer answer."""
    code = extract_code_block(code_text)
    if not code:
        # If no code block found, try using the text directly
        code = code_text.strip()
    
    safe_globals = {
        "__builtins__": {
            "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "set": set,
            "tuple": tuple, "sum": sum, "max": max, "min": min,
            "abs": abs, "round": round, "pow": pow, "print": print,
            "enumerate": enumerate, "zip": zip, "sorted": sorted,
            "reversed": reversed, "all": all, "any": any,
            "__import__": __import__,
        },
        "math": math,
        "__name__": "__main__",
    }
    safe_locals = {}
    out_buf = io.StringIO()
    
    try:
        with redirect_stdout(out_buf):
            exec(code, safe_globals, safe_locals)
        
        # Try to extract from printed output
        printed = out_buf.getvalue().strip()
        print(f"Printed output: {repr(printed)}")
        m = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed)
        if m:
            result = _normalize_aime_int(int(m[-1]))
            print(f"Found answer in print output: {result}")
            return result
        
        # Try common variable names
        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                v = _coerce_int(safe_locals[key])
                if v is not None:
                    result = _normalize_aime_int(v)
                    print(f"Found answer in variable '{key}': {result}")
                    return result
        
        # Try executing solve() or main() if present
        for fn_name in ("solve", "main"):
            fn = safe_locals.get(fn_name)
            if callable(fn):
                out_buf2 = io.StringIO()
                with redirect_stdout(out_buf2):
                    ret = fn()
                printed2 = out_buf2.getvalue().strip()
                m2 = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed2)
                if m2:
                    result = _normalize_aime_int(int(m2[-1]))
                    print(f"Found answer from {fn_name}() print: {result}")
                    return result
                v = _coerce_int(ret)
                if v is not None:
                    result = _normalize_aime_int(v)
                    print(f"Found answer from {fn_name}() return: {result}")
                    return result
        
        # Parse from printed output
        nums = re.findall(r"-?\d{1,3}", printed)
        if nums:
            result = _normalize_aime_int(int(nums[-1]))
            print(f"Found answer from printed numbers: {result}")
            return result
        
        print("No answer found")
        return None
    except Exception as e:
        print(f"Error executing code: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test code
test_code = """def convert_to_decimal(s, b):



    val = 0

    for c in s:

        d = int(c)

        val = val * b + d

    return val

answer = 0

for b in range(10, 100):

    n17 = convert_to_decimal('17', b)

    n97 = convert_to_decimal('97', b)

    if n97 % n17 == 0:

        answer += b

print("__ANS__=" + str(answer))"""

if __name__ == "__main__":
    print("Testing code execution function...")
    print("=" * 80)
    print("Input code:")
    print(test_code)
    print("=" * 80)
    
    result = execute_python_code(test_code)
    
    print("=" * 80)
    print(f"Result: {result}")
    print(f"Expected: 70")
    print(f"Match: {'✓' if result == 70 else '✗'}")

