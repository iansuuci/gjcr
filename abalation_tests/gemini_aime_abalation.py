#!/usr/bin/env python3
"""
Gemini ABLATION Experiment: Token-Limited Reasoning

This ablation test uses max_tokens set to a configurable percentage
of the average tokens across ALL conditions from the normal run.

Usage:
    python gemini_aime_abalation.py --percent 10   # 10% of avg tokens (default)
    python gemini_aime_abalation.py --percent 70   # 70% of avg tokens

Average across all conditions: 16,419 tokens
"""

import re
import io
import math
import json
import traceback
import os
import sys
import argparse
from typing import Dict, Optional, List, Tuple
from contextlib import redirect_stdout
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package not found.")
    raise

from datasets.aime_problems import AIME_2024_PROBLEMS
print(f"Loaded {len(AIME_2024_PROBLEMS)} AIME 2024 problems")

# ========================= Token Configuration =========================
# Average across all conditions: (26412 + 21646 + 24077 + 1020 + 8942) / 5 = 16419
AVG_TOKENS = 16419

# ========================= Utilities =========================

def _coerce_int(x) -> Optional[int]:
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
    m = re.search(r"PYCODE:\s*\n(.*?)\nENDPYCODE", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    kws = ("import", "def ", "for ", "while ", "if ", "=", "print", "return")
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if any(k in s for k in kws) or (lines and s):
            lines.append(ln)
    code = "\n".join(lines).strip()
    return code if len(code) >= 4 else None

def extract_comments(text: str) -> str:
    comments = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_text = stripped[1:].strip()
            if comment_text:
                comments.append(comment_text)
    return "\n".join(comments)

def has_executable_code(text: str) -> bool:
    code = extract_code_block(text)
    if not code:
        return False
    executable_keywords = ["import", "def ", "for ", "while ", "if ", "=", "print", "return"]
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if any(kw in stripped for kw in executable_keywords):
                return True
    return False

def has_natural_language(text: str) -> bool:
    text_no_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    text_no_code = re.sub(r"PYCODE:.*?ENDPYCODE", "", text_no_code, flags=re.DOTALL | re.IGNORECASE)
    lines = text_no_code.splitlines()
    non_comment_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if not any(kw in stripped for kw in ["import", "def ", "for ", "while ", "if ", "=", "print", "return"]):
                non_comment_lines.append(stripped)
    natural_text = " ".join(non_comment_lines)
    return len(natural_text.strip()) > 20

def execute_python_code_detailed(code_text: str) -> Dict:
    result = {
        "answer": None,
        "code_found": False,
        "execution_success": False,
        "extraction_method": "failed",
        "error": None,
    }
    
    code = extract_code_block(code_text)
    if not code:
        result["error"] = "no_code_block"
        return result
    
    result["code_found"] = True
    
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
        
        result["execution_success"] = True
        printed = out_buf.getvalue().strip()
        m = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed)
        if m:
            result["answer"] = _normalize_aime_int(int(m[-1]))
            result["extraction_method"] = "print_ANS"
            return result
        
        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                v = _coerce_int(safe_locals[key])
                if v is not None:
                    result["answer"] = _normalize_aime_int(v)
                    result["extraction_method"] = f"variable_{key}"
                    return result
        
        nums = re.findall(r"-?\d{1,3}", printed)
        if nums:
            result["answer"] = _normalize_aime_int(int(nums[-1]))
            result["extraction_method"] = "printed_number"
            return result
        
        result["error"] = "no_answer_found"
        return result
    except Exception as e:
        result["error"] = str(e)[:100]
        return result

def extract_final_answer_from_text(text: str) -> Optional[int]:
    patterns = [
        r"\\boxed\{(\d{1,3})\}",
        r"\*\*Final Answer[:\s]*(-?\d{1,3})\*\*",
        r"Final Answer[:\s]*(-?\d{1,3})",
        r"The final answer is[:\s]*(-?\d{1,3})",
        r"The answer is[:\s]*(-?\d{1,3})",
        r"Answer[:\s]*(-?\d{1,3})",
        r"=\s*(-?\d{1,3})\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            v = _coerce_int(m.group(1))
            return _normalize_aime_int(v)
    
    tail = "\n".join(text.strip().splitlines()[-15:])
    nums = re.findall(r"-?\d{1,3}", tail)
    if nums:
        v = _coerce_int(nums[-1])
        return _normalize_aime_int(v)
    
    return None


# ========================= Model Wrapper =========================

class ModelRunner:
    def __init__(self, api_key: Optional[str] = None, 
                 model_name: str = "gemini-3-flash-preview"):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key is None or api_key == "your_secret_key_here":
                raise ValueError("Gemini API key not provided.")
        
        print(f"Initializing Gemini API client (ABLATION MODE)")
        print(f"  Model: {model_name}")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print("✓ Gemini API client initialized\n")

    def generate(self, prompt: str, use_reasoning: bool = True, max_tokens: int = 1000) -> Tuple[str, Dict]:
        """Generate response with specified max_tokens limit."""
        thinking_level = "high" if use_reasoning else "low"
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                )
            )
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
                token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
                token_usage["total_tokens"] = response.usage_metadata.total_token_count or 0
            
            return (response.text or "").strip(), token_usage
        except Exception as e:
            print(f"ERROR: Failed to generate response: {e}")
            return "", token_usage


# ========================= Prompts =========================

def prompt_only_code(problem: str) -> str:
    return f"""You are solving an math problem. Generate ONLY executable Python code. 
DO NOT include any comments, explanations, or natural language text.
DO NOT use markdown code blocks or any formatting.
Output ONLY the Python code that solves the problem.

Problem: {problem}

Requirements:
- Output ONLY executable Python code
- No comments (no # symbols)
- No explanations before or after code
- No markdown formatting
- The code should compute the answer and print it as: print("__ANS__=" + str(answer))
- Store the final answer in a variable called 'answer'

Generate the code now:"""

def prompt_only_comments(problem: str) -> str:
    return f"""You are solving a math problem. Generate ONLY comments explaining your reasoning.
DO NOT include any executable Python code.
DO NOT use natural language paragraphs.
Output ONLY comment lines (lines starting with #) that explain how to solve the problem.

Problem: {problem}

Requirements:
- Output ONLY comment lines (each line must start with #)
- Explain your reasoning step by step in comments
- Do NOT include any executable code statements
- End with a comment containing the final answer: # Final Answer: <number>

Generate the comments now:"""

def prompt_both_code_and_comments(problem: str) -> str:
    return f"""You are solving a problem. Generate Python code WITH comments explaining your reasoning.

Problem: {problem}

Requirements:
- Write Python code to solve the problem
- Include comments (lines starting with #) explaining your reasoning
- The code should compute the answer and print it as: print("__ANS__=" + str(answer))
- Store the final answer in a variable called 'answer'
- Use comments to explain each step of your solution

Generate the code with comments now:"""

def prompt_nothing(problem: str) -> str:
    return f"""Do not use reasoning and solve the problem.

Problem: {problem}

Answer:"""

def prompt_cot(problem: str) -> str:
    return f"""Use chain of thought reasoning to solve this problem and mark the final answer with answer: . Problem: {problem}"""


# ========================= Answer Extraction =========================

def extract_answer_detailed(response: str, condition: str) -> Dict:
    result = {"answer": None, "extraction_source": "none", "code_details": None}
    
    if condition == "only_code":
        code_result = execute_python_code_detailed(response)
        result["code_details"] = code_result
        if code_result["answer"] is not None:
            result["answer"] = code_result["answer"]
            result["extraction_source"] = f"code_exec:{code_result['extraction_method']}"
            return result
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text_fallback"
        return result
        
    elif condition == "only_comments":
        comments = extract_comments(response)
        ans = extract_final_answer_from_text(comments)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "comments"
            return result
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text_fallback"
        return result
        
    elif condition == "both":
        code_result = execute_python_code_detailed(response)
        result["code_details"] = code_result
        if code_result["answer"] is not None:
            result["answer"] = code_result["answer"]
            result["extraction_source"] = f"code_exec:{code_result['extraction_method']}"
            return result
        comments = extract_comments(response)
        ans = extract_final_answer_from_text(comments)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "comments_fallback"
            return result
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text_fallback"
        return result
        
    else:
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text"
            return result
        code_result = execute_python_code_detailed(response)
        result["code_details"] = code_result
        if code_result["answer"] is not None:
            result["answer"] = code_result["answer"]
            result["extraction_source"] = f"code_exec:{code_result['extraction_method']}"
        return result
    
    return result


# ========================= Evaluation =========================

def run_single_condition(runner: ModelRunner, q: str, condition: str, token_limit: int) -> Dict:
    """Run a single condition - used for parallel execution."""
    if condition == "only_code":
        prompt = prompt_only_code(q)
    elif condition == "only_comments":
        prompt = prompt_only_comments(q)
    elif condition == "both":
        prompt = prompt_both_code_and_comments(q)
    elif condition == "nothing":
        prompt = prompt_nothing(q)
    elif condition == "cot":
        prompt = prompt_cot(q)
    
    use_reasoning = (condition != "nothing")
    response, token_usage = runner.generate(prompt, use_reasoning=use_reasoning, max_tokens=token_limit)
    extraction_result = extract_answer_detailed(response, condition)
    
    return {
        "condition": condition,
        "prompt": prompt,
        "response": response,
        "token_usage": token_usage,
        "extraction_result": extraction_result,
    }


def evaluate_problem(runner: ModelRunner, prob: Dict, index: int, token_limit: int, percent: int, parallel: bool = False) -> Dict:
    q = prob["question"]
    true_ans = int(prob["answer"])
    
    results = {
        "problem_index": index,
        "question": q,
        "true_answer": true_ans,
    }
    
    conditions = ["only_code", "only_comments", "both", "nothing", "cot"]
    
    if parallel:
        print(f"  Running {len(conditions)} conditions in parallel...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(run_single_condition, runner, q, cond, token_limit): cond 
                for cond in conditions
            }
            
            for future in as_completed(futures):
                condition = futures[future]
                try:
                    result = future.result()
                    response = result["response"]
                    token_usage = result["token_usage"]
                    extraction_result = result["extraction_result"]
                    pred = extraction_result["answer"]
                    
                    results[f"{condition}_prompt"] = result["prompt"]
                    results[f"{condition}_response"] = response
                    results[f"{condition}_prediction"] = pred
                    results[f"{condition}_correct"] = (pred == true_ans) if pred is not None else False
                    results[f"{condition}_extraction"] = extraction_result
                    results[f"{condition}_token_usage"] = token_usage
                    results[f"{condition}_max_tokens"] = token_limit
                    
                    pred_str = str(pred) if pred is not None else 'None'
                    print(f"  [{condition}] ✓ pred={pred_str}, correct={'✓' if (pred == true_ans) else '✗'}, "
                          f"tokens={token_usage['total_tokens']}")
                except Exception as e:
                    print(f"  [{condition}] ✗ Error: {e}")
    else:
        for condition in conditions:
            if condition == "only_code":
                prompt = prompt_only_code(q)
            elif condition == "only_comments":
                prompt = prompt_only_comments(q)
            elif condition == "both":
                prompt = prompt_both_code_and_comments(q)
            elif condition == "nothing":
                prompt = prompt_nothing(q)
            elif condition == "cot":
                prompt = prompt_cot(q)
            
            use_reasoning = (condition != "nothing")
            max_tokens = token_limit
            
            print(f"\n  [{condition}] max_tokens={max_tokens} ({percent}% of overall avg)")
            
            response, token_usage = runner.generate(prompt, use_reasoning=use_reasoning, max_tokens=max_tokens)
            
            print(f"  [{condition}] Response length: {len(response)} chars")
            print(f"  [{condition}] Token Usage: prompt={token_usage['prompt_tokens']}, "
                  f"completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}")
            
            extraction_result = extract_answer_detailed(response, condition)
            pred = extraction_result["answer"]
            
            results[f"{condition}_prompt"] = prompt
            results[f"{condition}_response"] = response
            results[f"{condition}_prediction"] = pred
            results[f"{condition}_correct"] = (pred == true_ans) if pred is not None else False
            results[f"{condition}_extraction"] = extraction_result
            results[f"{condition}_token_usage"] = token_usage
            results[f"{condition}_max_tokens"] = max_tokens
            
            pred_str = str(pred) if pred is not None else 'None'
            print(f"  [{condition}] Summary: pred={pred_str}, correct={'✓' if (pred == true_ans) else '✗'}")
    
    return results


# ========================= Main =========================

CONDITIONS = ["only_code", "only_comments", "both", "nothing", "cot"]

def main():
    parser = argparse.ArgumentParser(description="Gemini Ablation Test with configurable token limit")
    parser.add_argument("--percent", type=int, default=10, 
                        help="Percentage of average tokens to use as limit (default: 10)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run conditions in parallel for faster execution (up to 5x speedup)")
    args = parser.parse_args()
    
    percent = args.percent
    parallel = args.parallel
    token_limit = int(AVG_TOKENS * percent / 100)
    
    print("=" * 80)
    print(f"ABLATION TEST: Gemini with {percent}% Token Limit")
    print("=" * 80)
    print(f"Average tokens from baseline: {AVG_TOKENS}")
    print(f"Token limit ({percent}%): {token_limit} tokens")
    print(f"Parallel execution: {'ENABLED (5x faster)' if parallel else 'disabled'}")
    print("=" * 80)
    
    runner = ModelRunner()
    problems = AIME_2024_PROBLEMS
    
    all_results = []
    correct_counts = {c: 0 for c in CONDITIONS}
    token_stats = {c: {"total": [], "completion": []} for c in CONDITIONS}
    
    total = len(problems)
    
    for i, prob in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"Problem {i:02d}/{total}")
        print(f"{'='*80}")
        res = evaluate_problem(runner, prob, i, token_limit, percent, parallel=parallel)
        all_results.append(res)
        
        for condition in CONDITIONS:
            if res[f"{condition}_correct"]:
                correct_counts[condition] += 1
            token_usage = res.get(f"{condition}_token_usage", {})
            token_stats[condition]["total"].append(token_usage.get("total_tokens", 0))
            token_stats[condition]["completion"].append(token_usage.get("completion_tokens", 0))
    
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    
    print(f"\nSuccess Rates (with {token_limit} token limit = {percent}%):")
    for condition in CONDITIONS:
        rate = correct_counts[condition] / total * 100
        print(f"  {condition}: {correct_counts[condition]}/{total} = {rate:.1f}%")
    
    print("\nToken Usage Statistics:")
    for condition in CONDITIONS:
        avg_total = sum(token_stats[condition]["total"]) / len(token_stats[condition]["total"])
        avg_comp = sum(token_stats[condition]["completion"]) / len(token_stats[condition]["completion"])
        print(f"  {condition}: avg_total={avg_total:.0f}, avg_completion={avg_comp:.0f}, limit={token_limit}")
    
    output_file = os.path.join(os.path.dirname(__file__), "..", "abalation_results", f"gemini_abalation_{percent}pct_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "experiment": f"ablation_{percent}pct_tokens",
            "percent": percent,
            "avg_tokens_baseline": AVG_TOKENS,
            "token_limit": token_limit,
            "summary": {
                "total_problems": total,
                "success_rates": {c: correct_counts[c] / total for c in CONDITIONS},
            },
            "detailed_results": all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

