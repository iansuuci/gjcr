#!/usr/bin/env python3
"""
Gemini Experiment: Can models reason in code? (GSM8K Condensed Version)

This is a condensed version of the GSM8K experiment that runs on a small subset
of problems for faster testing and iteration.

This experiment measures the effect of code vs comments vs both vs nothing
on model performance using a subset of the GSM8K test dataset and Gemini API.

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
import random
from typing import Dict, Optional, List, Tuple
from contextlib import redirect_stdout
from collections import Counter

# Import Hugging Face datasets BEFORE adding project root to sys.path
# to avoid conflict with local datasets/ folder
try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    print("ERROR: datasets package not found. Install with: pip install datasets")
    raise

# Add project root to path for imports (after importing HF datasets)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("WARNING: python-dotenv not found. Install with: pip install python-dotenv")
    print("Will try to use environment variables directly.")

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package not found. Install with: pip install google-genai")
    raise

# Configuration for condensed experiment
NUM_PROBLEMS = 100  # Number of problems to sample from GSM8K
RANDOM_SEED = 42   # For reproducible sampling

# Load GSM8K dataset from Hugging Face and sample a subset
try:
    load_dataset = hf_load_dataset  # Use the pre-imported function
    print("Loading GSM8K test dataset from Hugging Face...")
    gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # Convert to list of dicts with 'question' and 'answer' fields
    all_problems = []
    for item in gsm8k_dataset:
        # GSM8K answer format typically ends with "#### <number>"
        answer_text = item["answer"]
        # Extract the final numeric answer after ####
        match = re.search(r"####\s*(-?[\d,]+)", answer_text)
        if match:
            # Remove commas from numbers like "1,234"
            answer_num = match.group(1).replace(",", "")
        else:
            # Fallback: try to get last number
            nums = re.findall(r"-?[\d,]+", answer_text)
            answer_num = nums[-1].replace(",", "") if nums else "0"
        
        all_problems.append({
            "question": item["question"],
            "answer": answer_num,
            "full_answer": answer_text  # Keep the full solution for reference
        })
    
    print(f"Loaded {len(all_problems)} GSM8K test problems total")
    
    # Sample a subset for condensed experiment
    random.seed(RANDOM_SEED)
    GSM8K_CONDENSED_PROBLEMS = random.sample(all_problems, min(NUM_PROBLEMS, len(all_problems)))
    print(f"Sampled {len(GSM8K_CONDENSED_PROBLEMS)} problems for condensed experiment (seed={RANDOM_SEED})")
    
except Exception as e:
    print(f"ERROR: Could not load GSM8K dataset ({e})")
    print("Install with: pip install datasets")
    raise


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
        # Handle comma-separated numbers
        s = str(x).strip().replace(",", "")
        return int(s)
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

def execute_python_code_detailed(code_text: str) -> Dict:
    """Execute Python code and extract integer answer with detailed status."""
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
        
        # Try to extract from printed output
        printed = out_buf.getvalue().strip()
        m = re.findall(r"__ANS__\s*=\s*(-?[\d,]+)", printed)
        if m:
            result["answer"] = _coerce_int(m[-1])
            result["extraction_method"] = "print_ANS"
            return result
        
        # Try common variable names
        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                v = _coerce_int(safe_locals[key])
                if v is not None:
                    result["answer"] = v
                    result["extraction_method"] = f"variable_{key}"
                    return result
        
        # Try executing solve() or main() if present
        for fn_name in ("solve", "main"):
            fn = safe_locals.get(fn_name)
            if callable(fn):
                out_buf2 = io.StringIO()
                with redirect_stdout(out_buf2):
                    ret = fn()
                printed2 = out_buf2.getvalue().strip()
                m2 = re.findall(r"__ANS__\s*=\s*(-?[\d,]+)", printed2)
                if m2:
                    result["answer"] = _coerce_int(m2[-1])
                    result["extraction_method"] = f"function_{fn_name}_print"
                    return result
                v = _coerce_int(ret)
                if v is not None:
                    result["answer"] = v
                    result["extraction_method"] = f"function_{fn_name}_return"
                    return result
        
        # Parse from printed output
        nums = re.findall(r"-?[\d,]+", printed)
        if nums:
            result["answer"] = _coerce_int(nums[-1])
            result["extraction_method"] = "printed_number"
            return result
        
        result["error"] = "no_answer_found"
        return result
    except Exception as e:
        result["error"] = str(e)[:100]
        return result


def execute_python_code(code_text: str) -> Optional[int]:
    """Execute Python code and extract integer answer (simple wrapper)."""
    result = execute_python_code_detailed(code_text)
    return result["answer"]

def extract_final_answer_from_text(text: str) -> Optional[int]:
    """Extract final answer from natural language text."""
    patterns = [
        r"\\boxed\{(-?[\d,]+)\}",
        r"\*\*Final Answer[:\s]*(-?[\d,]+)\*\*",
        r"Final Answer[:\s]*(-?[\d,]+)",
        r"The final answer is[:\s]*(-?[\d,]+)",
        r"The answer is[:\s]*(-?[\d,]+)",
        r"Answer[:\s]*(-?[\d,]+)",
        r"####\s*(-?[\d,]+)",
        r"=\s*(-?[\d,]+)\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            v = _coerce_int(m.group(1))
            return v
    
    tail = "\n".join(text.strip().splitlines()[-15:])
    nums = re.findall(r"-?[\d,]+", tail)
    if nums:
        v = _coerce_int(nums[-1])
        return v
    
    return None


# ========================= Model Wrapper =========================

class ModelRunner:
    def __init__(self, api_key: Optional[str] = None, 
                 model_name: str = "gemini-3-flash-preview"):
        """Initialize Gemini API client using google-genai package."""
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key is None or api_key == "your_secret_key_here":
                raise ValueError(
                    "Gemini API key not provided. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
                    "in your .env file or pass api_key parameter."
                )
        
        print(f"Initializing Gemini API client")
        print(f"  Model: {model_name}")
        print(f"  Reasoning mode: thinking_level='high'")
        print(f"  Non-reasoning mode: thinking_level='low'")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print("✓ Gemini API client initialized\n")

    def generate(self, prompt: str, use_reasoning: bool = True) -> str:
        """Generate response from prompt."""
        thinking_level = "high" if use_reasoning else "low"
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=32000,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                )
            )
            return (response.text or "").strip()
        except Exception as e:
            print(f"ERROR: Failed to generate response: {e}")
            return ""


# ========================= Prompts =========================

def prompt_only_code(problem: str) -> str:
    return f"""You are solving a math problem. Generate ONLY executable Python code. 
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


# ========================= Adherence Score =========================

def calculate_adherence_score(response: str, condition: str) -> Dict[str, float]:
    has_code = has_executable_code(response)
    has_comments = bool(extract_comments(response))
    has_nl = has_natural_language(response)
    
    details = {
        "has_executable_code": has_code,
        "has_comments": has_comments,
        "has_natural_language": has_nl,
    }
    
    if condition == "only_code":
        code_score = 1.0 if has_code else 0.0
        no_comments_score = 1.0 if not has_comments else 0.0
        no_nl_score = 1.0 if not has_nl else 0.0
        overall = (code_score + no_comments_score + no_nl_score) / 3.0
        details["code_required"] = code_score
        details["comments_forbidden"] = no_comments_score
        details["nl_forbidden"] = no_nl_score
    elif condition == "only_comments":
        comments_score = 1.0 if has_comments else 0.0
        no_code_score = 1.0 if not has_code else 0.0
        no_nl_score = 1.0 if not has_nl else 0.0
        overall = (comments_score + no_code_score + no_nl_score) / 3.0
        details["comments_required"] = comments_score
        details["code_forbidden"] = no_code_score
        details["nl_forbidden"] = no_nl_score
    elif condition == "both":
        code_score = 1.0 if has_code else 0.0
        comments_score = 1.0 if has_comments else 0.0
        overall = (code_score + comments_score) / 2.0
        details["code_required"] = code_score
        details["comments_required"] = comments_score
    elif condition == "nothing":
        overall = 1.0
        details["no_requirements"] = True
    elif condition == "cot":
        overall = 1.0
        details["cot_condition"] = True
    else:
        overall = 0.0
    
    return {"overall_score": overall, "details": details}


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
    elif condition == "nothing":
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
    elif condition == "cot":
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


def extract_answer(response: str, condition: str) -> Optional[int]:
    result = extract_answer_detailed(response, condition)
    return result["answer"]


# ========================= Evaluation =========================

def evaluate_problem(runner: ModelRunner, prob: Dict, index: int) -> Dict:
    q = prob["question"]
    true_ans = int(prob["answer"])
    
    results = {"problem_index": index, "question": q, "true_answer": true_ans}
    conditions = ["only_code", "only_comments", "both", "nothing", "cot"]
    
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
        response = runner.generate(prompt, use_reasoning=use_reasoning)
        
        extraction_result = extract_answer_detailed(response, condition)
        pred = extraction_result["answer"]
        adherence = calculate_adherence_score(response, condition)
        
        results[f"{condition}_prompt"] = prompt
        results[f"{condition}_response"] = response
        results[f"{condition}_prediction"] = pred
        results[f"{condition}_correct"] = (pred == true_ans) if pred is not None else False
        results[f"{condition}_adherence"] = adherence
        results[f"{condition}_extraction"] = extraction_result
    
    return results


# ========================= Main Experiment =========================

def main():
    model_name = "gemini-3-flash-preview"
    print("=" * 80)
    print("Can models reason in code? - Gemini Experiment (GSM8K CONDENSED)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Reasoning: thinking_level='high'")
    print(f"Non-Reasoning: thinking_level='low'")
    print(f"Dataset: GSM8K Condensed ({len(GSM8K_CONDENSED_PROBLEMS)} problems sampled from test set)")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Conditions: only_code, only_comments, both, nothing (low thinking), cot")
    print("=" * 80)
    print()
    
    runner = ModelRunner(model_name=model_name)
    problems = GSM8K_CONDENSED_PROBLEMS
    
    all_results = []
    correct_counts = {"only_code": 0, "only_comments": 0, "both": 0, "nothing": 0, "cot": 0}
    adherence_scores = {"only_code": [], "only_comments": [], "both": [], "nothing": [], "cot": []}
    code_stats = {
        "only_code": {"code_found": 0, "exec_success": 0, "exec_fail": 0, "answer_from_code": 0},
        "both": {"code_found": 0, "exec_success": 0, "exec_fail": 0, "answer_from_code": 0},
    }
    extraction_sources = {condition: Counter() for condition in ["only_code", "only_comments", "both", "nothing", "cot"]}
    
    total = len(problems)
    
    for i, prob in enumerate(problems, 1):
        print(f"\rProcessing problem {i}/{total}...", end="", flush=True)
        res = evaluate_problem(runner, prob, i)
        all_results.append(res)
        
        for condition in correct_counts.keys():
            if res[f"{condition}_correct"]:
                correct_counts[condition] += 1
            adherence_scores[condition].append(res[f"{condition}_adherence"]["overall_score"])
            extraction = res.get(f"{condition}_extraction", {})
            source = extraction.get("extraction_source", "unknown")
            extraction_sources[condition][source] += 1
            if condition in code_stats:
                code_details = extraction.get("code_details")
                if code_details:
                    if code_details.get("code_found"):
                        code_stats[condition]["code_found"] += 1
                        if code_details.get("execution_success"):
                            code_stats[condition]["exec_success"] += 1
                        else:
                            code_stats[condition]["exec_fail"] += 1
                    if source.startswith("code_exec:"):
                        code_stats[condition]["answer_from_code"] += 1
    
    print()  # Newline after progress indicator
    
    success_rates = {}
    for condition in correct_counts.keys():
        success_rates[condition] = correct_counts[condition] / total if total > 0 else 0.0
    
    X = success_rates["only_code"]
    X_prime = success_rates["only_comments"]
    X_double_prime = success_rates["both"]
    X_triple_prime = success_rates["nothing"]
    
    delta_code = X - X_triple_prime
    delta_comments = X_prime - X_triple_prime
    delta_cot = X_triple_prime - X_double_prime
    
    avg_adherence = {}
    for condition in adherence_scores.keys():
        scores = adherence_scores[condition]
        avg_adherence[condition] = sum(scores) / len(scores) if scores else 0.0
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS (CONDENSED)")
    print("=" * 80)
    print("\nSuccess Rates:")
    print(f"  X  (only Code):        {correct_counts['only_code']:3d}/{total} = {X*100:6.2f}%")
    print(f"  X' (only Comments):    {correct_counts['only_comments']:3d}/{total} = {X_prime*100:6.2f}%")
    print(f"  X'' (Both):            {correct_counts['both']:3d}/{total} = {X_double_prime*100:6.2f}%")
    print(f"  X''' (Nothing):        {correct_counts['nothing']:3d}/{total} = {X_triple_prime*100:6.2f}%")
    print(f"  CoT (Chain of Thought): {correct_counts['cot']:3d}/{total} = {success_rates['cot']*100:6.2f}%")
    
    print("\nEffects:")
    print(f"  Δ_Code     = X - X'''  = {delta_code*100:+.2f}%")
    print(f"  Δ_Comments = X' - X''' = {delta_comments*100:+.2f}%")
    print(f"  Δ_CoT      = X''' - X'' = {delta_cot*100:+.2f}%")
    
    print("\nAverage Adherence Scores:")
    for condition in avg_adherence.keys():
        print(f"  {condition:15s}: {avg_adherence[condition]:.3f}")
    
    print("\n" + "-" * 80)
    print("CODE EXECUTION ANALYSIS")
    print("-" * 80)
    for condition in ["only_code", "both"]:
        stats = code_stats[condition]
        print(f"\n  {condition}:")
        print(f"    Code blocks found:     {stats['code_found']:3d}/{total}")
        print(f"    Execution success:     {stats['exec_success']:3d}/{stats['code_found'] if stats['code_found'] > 0 else 1} "
              f"({100*stats['exec_success']/max(stats['code_found'],1):.1f}%)")
        print(f"    Execution failed:      {stats['exec_fail']:3d}/{stats['code_found'] if stats['code_found'] > 0 else 1} "
              f"({100*stats['exec_fail']/max(stats['code_found'],1):.1f}%)")
        print(f"    Answer from code:      {stats['answer_from_code']:3d}/{total} "
              f"({100*stats['answer_from_code']/total:.1f}%)")
    
    print("\n" + "-" * 80)
    print("EXTRACTION SOURCE BREAKDOWN")
    print("-" * 80)
    for condition in ["only_code", "only_comments", "both", "nothing", "cot"]:
        print(f"\n  {condition}:")
        for source, count in extraction_sources[condition].most_common():
            print(f"    {source:30s}: {count:3d} ({100*count/total:.1f}%)")
    
    print("\n" + "=" * 80)
    
    output_file = "gemini_gsm_condensed_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_problems": total,
                "sample_size": NUM_PROBLEMS,
                "random_seed": RANDOM_SEED,
                "success_rates": success_rates,
                "effects": {"delta_code": delta_code, "delta_comments": delta_comments, "delta_cot": delta_cot},
                "average_adherence": avg_adherence,
                "code_execution_stats": code_stats,
                "extraction_sources": {k: dict(v) for k, v in extraction_sources.items()},
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

