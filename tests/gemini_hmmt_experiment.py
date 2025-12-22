#!/usr/bin/env python3
"""
Gemini Experiment: Can models reason in code? (HMMT Version)

Uses thinking_level="high" for reasoning and thinking_level="low" for non-reasoning.
Install with: pip install google-genai
"""

import re
import io
import math
import json
import traceback
import os
import sys
import time
from typing import Dict, Optional, Union
from contextlib import redirect_stdout
from collections import Counter
from fractions import Fraction

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    print("ERROR: datasets package not found. Install with: pip install datasets")
    raise

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not found.")

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package not found. Install with: pip install google-genai")
    raise

# Load HMMT dataset
load_dataset = hf_load_dataset
HMMT_PROBLEMS = None

dataset_paths = ["FlagEval/HMMT_2025", "FlagEval/HMMT-2025", "Maxwell-Jia/HMMT"]

for dataset_path in dataset_paths:
    for attempt in range(3):
        try:
            print(f"Loading HMMT dataset from {dataset_path}... (attempt {attempt + 1}/3)")
            hmmt_dataset = load_dataset(dataset_path, split="train", trust_remote_code=True)
            HMMT_PROBLEMS = []
            for item in hmmt_dataset:
                HMMT_PROBLEMS.append({
                    "question": item.get("problem", item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "solution": item.get("solution", ""),
                    "id": item.get("id", ""),
                })
            print(f"✓ Loaded {len(HMMT_PROBLEMS)} HMMT problems")
            break
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
    if HMMT_PROBLEMS is not None:
        break

if HMMT_PROBLEMS is None or len(HMMT_PROBLEMS) == 0:
    print("\nUsing embedded sample HMMT problems.\n")
    HMMT_PROBLEMS = [
        {"question": "Find the sum of all positive integers $n$ such that $n^2 + 12n - 2007$ is a perfect square.", "answer": "80"},
        {"question": "Let $a$ and $b$ be positive real numbers such that $a + b = 1$. Find the minimum value of $\\frac{1}{a} + \\frac{4}{b}$.", "answer": "9"},
        {"question": "How many ways are there to arrange the letters in BANANA such that no two adjacent letters are the same?", "answer": "40"},
        {"question": "Find the number of ordered pairs $(a, b)$ of positive integers such that $\\gcd(a, b) = 1$ and $a + b = 100$.", "answer": "40"},
        {"question": "A triangle has sides of length 13, 14, and 15. Find the area of the triangle.", "answer": "84"},
    ]
    print(f"Loaded {len(HMMT_PROBLEMS)} sample problems")


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    answer = answer.strip().replace(",", "")
    answer = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", answer)
    answer = answer.replace("\\", "").replace("$", "")
    if "/" in answer and not any(c.isalpha() for c in answer):
        try:
            parts = answer.split("/")
            if len(parts) == 2:
                frac = Fraction(int(float(parts[0])), int(float(parts[1])))
                answer = f"{frac.numerator}/{frac.denominator}" if frac.denominator != 1 else str(frac.numerator)
        except:
            pass
    return answer.lower().strip()


def answers_match(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    if pred_norm == exp_norm:
        return True
    try:
        pred_val = float(pred_norm.split("/")[0]) / float(pred_norm.split("/")[1]) if "/" in pred_norm else float(pred_norm)
        exp_val = float(exp_norm.split("/")[0]) / float(exp_norm.split("/")[1]) if "/" in exp_norm else float(exp_norm)
        if abs(pred_val - exp_val) < 1e-9:
            return True
    except:
        pass
    return False


def extract_code_block(text: str) -> Optional[str]:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    kws = ("import", "def ", "for ", "while ", "if ", "=", "print", "return")
    lines = [ln for ln in text.splitlines() if any(k in ln.strip() for k in kws)]
    code = "\n".join(lines).strip()
    return code if len(code) >= 4 else None


def extract_comments(text: str) -> str:
    return "\n".join(line.strip()[1:].strip() for line in text.splitlines() 
                     if line.strip().startswith("#") and len(line.strip()) > 1)


def has_executable_code(text: str) -> bool:
    code = extract_code_block(text)
    if not code:
        return False
    kws = ["import", "def ", "for ", "while ", "if ", "=", "print", "return"]
    return any(kw in line for line in code.splitlines() for kw in kws if not line.strip().startswith("#"))


def has_natural_language(text: str) -> bool:
    text_clean = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    lines = [l.strip() for l in text_clean.splitlines() if l.strip() and not l.strip().startswith("#")]
    return len(" ".join(lines)) > 20


def execute_python_code_detailed(code_text: str) -> Dict:
    result = {"answer": None, "code_found": False, "execution_success": False, "extraction_method": "failed", "error": None}
    code = extract_code_block(code_text)
    if not code:
        result["error"] = "no_code_block"
        return result
    result["code_found"] = True
    safe_globals = {
        "__builtins__": {"range": range, "len": len, "int": int, "float": float, "str": str, "list": list,
                        "dict": dict, "set": set, "tuple": tuple, "sum": sum, "max": max, "min": min,
                        "abs": abs, "round": round, "pow": pow, "print": print, "enumerate": enumerate,
                        "zip": zip, "sorted": sorted, "reversed": reversed, "all": all, "any": any, "__import__": __import__},
        "math": math, "Fraction": Fraction, "__name__": "__main__",
    }
    safe_locals = {}
    out_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf):
            exec(code, safe_globals, safe_locals)
        result["execution_success"] = True
        printed = out_buf.getvalue().strip()
        m = re.findall(r"__ANS__\s*=\s*(.+?)(?:\n|$)", printed)
        if m:
            result["answer"] = m[-1].strip()
            result["extraction_method"] = "print_ANS"
            return result
        for key in ("answer", "result", "final_answer", "ans"):
            if key in safe_locals and safe_locals[key] is not None:
                result["answer"] = str(safe_locals[key])
                result["extraction_method"] = f"variable_{key}"
                return result
        if printed:
            result["answer"] = printed.splitlines()[-1].strip()
            result["extraction_method"] = "printed_output"
            return result
        result["error"] = "no_answer_found"
        return result
    except Exception as e:
        result["error"] = str(e)[:100]
        return result


def extract_final_answer_from_text(text: str) -> Optional[str]:
    patterns = [r"\\boxed\{([^}]+)\}", r"Final Answer[:\s]*(.+?)(?:\n|$)", r"The answer is[:\s]*(.+?)(?:\n|$)",
                r"Answer[:\s]*(.+?)(?:\n|$)", r"####\s*(.+?)(?:\n|$)"]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


class ModelRunner:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview"):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        print(f"Initializing Gemini: {model_name}")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print("✓ Gemini initialized\n")

    def generate(self, prompt: str, use_reasoning: bool = True) -> str:
        thinking_level = "high" if use_reasoning else "low"
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=32000,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
                )
            )
            return (response.text or "").strip()
        except Exception as e:
            print(f"ERROR: {e}")
            return ""


def prompt_only_code(problem: str) -> str:
    return f"""Generate ONLY executable Python code. No comments, no markdown.
Problem: {problem}
Print answer as: print("__ANS__=" + str(answer))"""

def prompt_only_comments(problem: str) -> str:
    return f"""Generate ONLY comments (lines starting with #) explaining reasoning.
Problem: {problem}
End with: # Final Answer: <answer>"""

def prompt_both(problem: str) -> str:
    return f"""Generate Python code WITH comments.
Problem: {problem}
Print answer as: print("__ANS__=" + str(answer))"""

def prompt_nothing(problem: str) -> str:
    return f"""Do not use reasoning. Problem: {problem}
Answer:"""

def prompt_cot(problem: str) -> str:
    return f"""Use chain of thought reasoning. Mark final answer with answer: . Problem: {problem}"""


def calculate_adherence_score(response: str, condition: str) -> Dict:
    has_code = has_executable_code(response)
    has_comments = bool(extract_comments(response))
    has_nl = has_natural_language(response)
    if condition == "only_code":
        overall = (1.0 if has_code else 0.0) + (1.0 if not has_comments else 0.0) + (1.0 if not has_nl else 0.0)
        overall /= 3.0
    elif condition == "only_comments":
        overall = (1.0 if has_comments else 0.0) + (1.0 if not has_code else 0.0) + (1.0 if not has_nl else 0.0)
        overall /= 3.0
    elif condition == "both":
        overall = ((1.0 if has_code else 0.0) + (1.0 if has_comments else 0.0)) / 2.0
    else:
        overall = 1.0
    return {"overall_score": overall, "details": {"has_code": has_code, "has_comments": has_comments, "has_nl": has_nl}}


def extract_answer_detailed(response: str, condition: str) -> Dict:
    result = {"answer": None, "extraction_source": "none", "code_details": None}
    if condition in ["only_code", "both"]:
        code_result = execute_python_code_detailed(response)
        result["code_details"] = code_result
        if code_result["answer"]:
            result["answer"] = code_result["answer"]
            result["extraction_source"] = f"code_exec:{code_result['extraction_method']}"
            return result
    if condition == "only_comments":
        ans = extract_final_answer_from_text(extract_comments(response))
        if ans:
            result["answer"] = ans
            result["extraction_source"] = "comments"
            return result
    ans = extract_final_answer_from_text(response)
    if ans:
        result["answer"] = ans
        result["extraction_source"] = "text"
    return result


def evaluate_problem(runner: ModelRunner, prob: Dict, index: int) -> Dict:
    q, true_ans = prob["question"], prob["answer"]
    results = {"problem_index": index, "question": q, "true_answer": true_ans}
    conditions = [("only_code", prompt_only_code, True), ("only_comments", prompt_only_comments, True),
                  ("both", prompt_both, True), ("nothing", prompt_nothing, False), ("cot", prompt_cot, True)]
    for cond, prompt_fn, use_reasoning in conditions:
        prompt = prompt_fn(q)
        print(f"\n  [{cond}] thinking_level: {'high' if use_reasoning else 'low'}")
        response = runner.generate(prompt, use_reasoning=use_reasoning)
        print(f"  Response: {response[:200]}...")
        extraction = extract_answer_detailed(response, cond)
        pred = extraction["answer"]
        is_correct = answers_match(pred, true_ans) if pred else False
        adherence = calculate_adherence_score(response, cond)
        results[f"{cond}_response"] = response
        results[f"{cond}_prediction"] = pred
        results[f"{cond}_correct"] = is_correct
        results[f"{cond}_adherence"] = adherence
        print(f"  Summary: pred={pred}, correct={'✓' if is_correct else '✗'}")
    return results


def main():
    model_name = "gemini-3-flash-preview"
    print("=" * 80)
    print(f"Gemini HMMT Experiment - Model: {model_name}")
    print(f"Dataset: {len(HMMT_PROBLEMS)} problems")
    print("=" * 80)
    
    runner = ModelRunner(model_name=model_name)
    all_results = []
    correct_counts = {"only_code": 0, "only_comments": 0, "both": 0, "nothing": 0, "cot": 0}
    
    for i, prob in enumerate(HMMT_PROBLEMS, 1):
        print(f"\nProblem {i}/{len(HMMT_PROBLEMS)}")
        res = evaluate_problem(runner, prob, i)
        all_results.append(res)
        for cond in correct_counts:
            if res[f"{cond}_correct"]:
                correct_counts[cond] += 1
    
    total = len(HMMT_PROBLEMS)
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    for cond, count in correct_counts.items():
        print(f"  {cond}: {count}/{total} = {100*count/total:.2f}%")
    
    with open("gemini_hmmt_experiment_results.json", "w") as f:
        json.dump({"summary": {"total": total, "correct_counts": correct_counts}, "results": all_results}, f, indent=2)
    print("\nResults saved to gemini_hmmt_experiment_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
