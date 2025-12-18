#!/usr/bin/env python3
"""
Test Experiment: Can models reason in code?

This experiment measures the effect of code vs comments vs both vs nothing
on model performance using the AIME condensed dataset and openai/gpt-oss-20b model.

Based on the research paper design:
- X = P(A | only Code)
- X' = P(A | only Comments)  
- X'' = P(A | Both)
- X''' = P(A | Nothing)

Effects:
- Δ_Code = X - X'''
- Δ_Comments = X' - X'''
- Δ_CoT = X''' - X'' (if applicable)

Also includes Adherence Score Optimization to measure instruction following.
This version prints and records model output for each prompt.
"""

import re
import io
import math
import json
import traceback
from typing import Dict, Optional, List, Tuple
from contextlib import redirect_stdout
from collections import Counter

from vllm import LLM, SamplingParams

# Load AIME dataset
try:
    from datasets.aime_condensed import AIME_2024_PROBLEMS
    print(f"Loaded {len(AIME_2024_PROBLEMS)} AIME 2024 problems (condensed)")
except Exception as e:
    print(f"WARNING: Could not load AIME problems ({e})")
    raise FileNotFoundError("AIME problems not found")


# ========================= Utilities =========================

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

def extract_comments(text: str) -> str:
    """Extract comments from code (lines starting with #)."""
    comments = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_text = stripped[1:].strip()
            if comment_text:
                comments.append(comment_text)
    return "\n".join(comments)

def has_executable_code(text: str) -> bool:
    """Check if text contains executable Python code."""
    code = extract_code_block(text)
    if not code:
        return False
    # Check for executable statements (not just comments)
    executable_keywords = ["import", "def ", "for ", "while ", "if ", "=", "print", "return"]
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if any(kw in stripped for kw in executable_keywords):
                return True
    return False

def has_natural_language(text: str) -> bool:
    """Detect if text contains natural language (non-code, non-comment text)."""
    # Remove code blocks
    text_no_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    text_no_code = re.sub(r"PYCODE:.*?ENDPYCODE", "", text_no_code, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove comments
    lines = text_no_code.splitlines()
    non_comment_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            # Check if it's not just a code-like line
            if not any(kw in stripped for kw in ["import", "def ", "for ", "while ", "if ", "=", "print", "return"]):
                non_comment_lines.append(stripped)
    
    # If we have substantial non-code text, it's natural language
    natural_text = " ".join(non_comment_lines)
    return len(natural_text.strip()) > 20  # Threshold for natural language

def execute_python_code(code_text: str) -> Optional[int]:
    """Execute Python code and extract integer answer."""
    code = extract_code_block(code_text)
    if not code:
        return None
    
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
        m = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed)
        if m:
            return _normalize_aime_int(int(m[-1]))
        
        # Try common variable names
        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                v = _coerce_int(safe_locals[key])
                if v is not None:
                    return _normalize_aime_int(v)
        
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
                    return _normalize_aime_int(int(m2[-1]))
                v = _coerce_int(ret)
                if v is not None:
                    return _normalize_aime_int(v)
        
        # Parse from printed output
        nums = re.findall(r"-?\d{1,3}", printed)
        if nums:
            return _normalize_aime_int(int(nums[-1]))
        
        return None
    except Exception:
        return None

def extract_final_answer_from_text(text: str) -> Optional[int]:
    """Extract final answer from natural language text."""
    patterns = [
        r"\*\*Final Answer[:\s]*(-?\d{1,3})\*\*",
        r"Final Answer[:\s]*(-?\d{1,3})",
        r"The final answer is[:\s]*(-?\d{1,3})",
        r"Answer[:\s]*(-?\d{1,3})",
        r"=\s*(-?\d{1,3})\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            v = _coerce_int(m.group(1))
            return _normalize_aime_int(v)
    
    # Try last few numbers in text
    tail = "\n".join(text.strip().splitlines()[-15:])
    nums = re.findall(r"-?\d{1,3}", tail)
    if nums:
        v = _coerce_int(nums[-1])
        return _normalize_aime_int(v)
    
    return None


# ========================= Model Wrapper =========================

class ModelRunner:
    def __init__(self, model_path: str = "openai/gpt-oss-20b"):
        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=16384,
            dtype="bfloat16",
            enforce_eager=False,
            disable_log_stats=True,
            enable_prefix_caching=True,
            max_num_seqs=16,
        )
        self.params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            n=1,
            max_tokens=2048,
        )
        print("✓ Model loaded\n")

    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        out = self.llm.generate([prompt], self.params)[0]
        return (out.outputs[0].text or "").strip() if out.outputs else ""


# ========================= Prompts for Four Conditions =========================

def prompt_only_code(problem: str) -> str:
    """Prompt for ONLY code - no comments, no natural language."""
    return f"""You are solving an AIME problem. Generate ONLY executable Python code. 
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
    """Prompt for ONLY comments - reasoning in comments, no executable code."""
    return f"""You are solving an AIME problem. Generate ONLY comments explaining your reasoning.
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
    """Prompt for BOTH code and comments."""
    return f"""You are solving an AIME problem. Generate Python code WITH comments explaining your reasoning.

Problem: {problem}

Requirements:
- Write Python code to solve the problem
- Include comments (lines starting with #) explaining your reasoning
- The code should compute the answer and print it as: print("__ANS__=" + str(answer))
- Store the final answer in a variable called 'answer'
- Use comments to explain each step of your solution

Generate the code with comments now:"""

def prompt_nothing(problem: str) -> str:
    """Prompt for NOTHING - just the problem, no instructions."""
    return f"""{problem}"""


# ========================= Adherence Score Calculation =========================

def calculate_adherence_score(response: str, condition: str) -> Dict[str, float]:
    """
    Calculate adherence score for how well the model followed instructions.
    
    Returns a dictionary with:
    - overall_score: Overall adherence (0.0 to 1.0)
    - has_code: Whether code is present (when required/forbidden)
    - has_comments: Whether comments are present (when required/forbidden)
    - has_natural_lang: Whether natural language is present (when forbidden)
    - details: Detailed breakdown
    """
    has_code = has_executable_code(response)
    has_comments = bool(extract_comments(response))
    has_nl = has_natural_language(response)
    
    details = {
        "has_executable_code": has_code,
        "has_comments": has_comments,
        "has_natural_language": has_nl,
    }
    
    # Score based on condition
    if condition == "only_code":
        # Should have code, NO comments, NO natural language
        code_score = 1.0 if has_code else 0.0
        no_comments_score = 1.0 if not has_comments else 0.0
        no_nl_score = 1.0 if not has_nl else 0.0
        overall = (code_score + no_comments_score + no_nl_score) / 3.0
        details["code_required"] = code_score
        details["comments_forbidden"] = no_comments_score
        details["nl_forbidden"] = no_nl_score
        
    elif condition == "only_comments":
        # Should have comments, NO code, NO natural language
        comments_score = 1.0 if has_comments else 0.0
        no_code_score = 1.0 if not has_code else 0.0
        no_nl_score = 1.0 if not has_nl else 0.0
        overall = (comments_score + no_code_score + no_nl_score) / 3.0
        details["comments_required"] = comments_score
        details["code_forbidden"] = no_code_score
        details["nl_forbidden"] = no_nl_score
        
    elif condition == "both":
        # Should have BOTH code and comments
        code_score = 1.0 if has_code else 0.0
        comments_score = 1.0 if has_comments else 0.0
        overall = (code_score + comments_score) / 2.0
        details["code_required"] = code_score
        details["comments_required"] = comments_score
        
    elif condition == "nothing":
        # Baseline - no specific requirements, but we still measure what was produced
        overall = 1.0  # Always adherent (no restrictions)
        details["no_requirements"] = True
        
    else:
        overall = 0.0
    
    return {
        "overall_score": overall,
        "details": details
    }


# ========================= Answer Extraction by Condition =========================

def extract_answer(response: str, condition: str) -> Optional[int]:
    """Extract answer from response based on condition."""
    if condition == "only_code":
        # Try to execute code
        ans = execute_python_code(response)
        if ans is not None:
            return ans
        # Fallback: try to parse from code text
        return extract_final_answer_from_text(response)
        
    elif condition == "only_comments":
        # Extract from comments
        comments = extract_comments(response)
        ans = extract_final_answer_from_text(comments)
        if ans is not None:
            return ans
        # Fallback: try natural language extraction
        return extract_final_answer_from_text(response)
        
    elif condition == "both":
        # Try code execution first
        ans = execute_python_code(response)
        if ans is not None:
            return ans
        # Fallback: try comments
        comments = extract_comments(response)
        ans = extract_final_answer_from_text(comments)
        if ans is not None:
            return ans
        # Final fallback: natural language
        return extract_final_answer_from_text(response)
        
    elif condition == "nothing":
        # Try all methods
        ans = execute_python_code(response)
        if ans is not None:
            return ans
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            return ans
        return None
    
    return None


# ========================= Evaluation =========================

def evaluate_problem(runner: ModelRunner, prob: Dict, index: int) -> Dict:
    """Evaluate a single problem across all four conditions."""
    q = prob["question"]
    true_ans = int(prob["answer"])
    
    results = {
        "problem_index": index,
        "question": q,
        "true_answer": true_ans,
    }
    
    conditions = ["only_code", "only_comments", "both", "nothing"]
    
    for condition in conditions:
        # Generate prompt
        if condition == "only_code":
            prompt = prompt_only_code(q)
        elif condition == "only_comments":
            prompt = prompt_only_comments(q)
        elif condition == "both":
            prompt = prompt_both_code_and_comments(q)
        else:  # nothing
            prompt = prompt_nothing(q)
        
        # Print prompt
        print(f"\n  [{condition}] Prompt:")
        print("  " + "-" * 76)
        for line in prompt.splitlines():
            print(f"  {line}")
        print("  " + "-" * 76)
        
        # Generate response
        response = runner.generate(prompt)
        
        # Print response
        print(f"\n  [{condition}] Model Response:")
        print("  " + "-" * 76)
        for line in response.splitlines():
            print(f"  {line}")
        print("  " + "-" * 76)
        
        # Extract answer
        pred = extract_answer(response, condition)
        
        # Calculate adherence
        adherence = calculate_adherence_score(response, condition)
        
        # Store results
        results[f"{condition}_prompt"] = prompt
        results[f"{condition}_response"] = response
        results[f"{condition}_prediction"] = pred
        results[f"{condition}_correct"] = (pred == true_ans) if pred is not None else False
        results[f"{condition}_adherence"] = adherence
        
        pred_str = str(pred) if pred is not None else 'None'
        print(f"\n  [{condition}] Summary: pred={pred_str:>3s}, "
              f"correct={'✓' if (pred == true_ans) else '✗'}, "
              f"adherence={adherence['overall_score']:.2f}")
    
    return results


# ========================= Main Experiment =========================

def main():
    print("=" * 80)
    print("Can models reason in code? - Test Experiment")
    print("=" * 80)
    print(f"Model: openai/gpt-oss-20b")
    print(f"Dataset: AIME 2024 Condensed ({len(AIME_2024_PROBLEMS)} problems)")
    print(f"Conditions: only_code, only_comments, both, nothing")
    print("=" * 80)
    print()
    
    runner = ModelRunner("openai/gpt-oss-20b")
    problems = AIME_2024_PROBLEMS
    
    all_results = []
    correct_counts = {
        "only_code": 0,
        "only_comments": 0,
        "both": 0,
        "nothing": 0,
    }
    
    adherence_scores = {
        "only_code": [],
        "only_comments": [],
        "both": [],
        "nothing": [],
    }
    
    total = len(problems)
    
    for i, prob in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"Problem {i:02d}/{total}:")
        print(f"{'='*80}")
        res = evaluate_problem(runner, prob, i)
        all_results.append(res)
        
        # Update counts
        for condition in correct_counts.keys():
            if res[f"{condition}_correct"]:
                correct_counts[condition] += 1
            adherence_scores[condition].append(res[f"{condition}_adherence"]["overall_score"])
    
    # Calculate success rates
    success_rates = {}
    for condition in correct_counts.keys():
        success_rates[condition] = correct_counts[condition] / total if total > 0 else 0.0
    
    # Calculate effects
    X = success_rates["only_code"]  # P(A | only Code)
    X_prime = success_rates["only_comments"]  # P(A | only Comments)
    X_double_prime = success_rates["both"]  # P(A | Both)
    X_triple_prime = success_rates["nothing"]  # P(A | Nothing)
    
    delta_code = X - X_triple_prime
    delta_comments = X_prime - X_triple_prime
    delta_cot = X_triple_prime - X_double_prime  # Note: as defined in paper
    
    # Calculate average adherence scores
    avg_adherence = {}
    for condition in adherence_scores.keys():
        scores = adherence_scores[condition]
        avg_adherence[condition] = sum(scores) / len(scores) if scores else 0.0
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("\nSuccess Rates:")
    print(f"  X  (only Code):        {correct_counts['only_code']:3d}/{total} = {X*100:6.2f}%")
    print(f"  X' (only Comments):    {correct_counts['only_comments']:3d}/{total} = {X_prime*100:6.2f}%")
    print(f"  X'' (Both):            {correct_counts['both']:3d}/{total} = {X_double_prime*100:6.2f}%")
    print(f"  X''' (Nothing):        {correct_counts['nothing']:3d}/{total} = {X_triple_prime*100:6.2f}%")
    
    print("\nEffects:")
    print(f"  Δ_Code     = X - X'''  = {delta_code*100:+.2f}%")
    print(f"  Δ_Comments = X' - X''' = {delta_comments*100:+.2f}%")
    print(f"  Δ_CoT      = X''' - X'' = {delta_cot*100:+.2f}%")
    
    print("\nAverage Adherence Scores:")
    for condition in avg_adherence.keys():
        print(f"  {condition:15s}: {avg_adherence[condition]:.3f}")
    
    print("=" * 80)
    
    # Save detailed results to JSON
    output_file = "test_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_problems": total,
                "success_rates": success_rates,
                "effects": {
                    "delta_code": delta_code,
                    "delta_comments": delta_comments,
                    "delta_cot": delta_cot,
                },
                "average_adherence": avg_adherence,
            },
            "detailed_results": all_results,
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

