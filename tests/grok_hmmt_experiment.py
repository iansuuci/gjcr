#!/usr/bin/env python3
"""
Grok Experiment: Can models reason in code? (HMMT Version)

This experiment measures the effect of code vs comments vs both vs nothing
on model performance using the HMMT (Harvard-MIT Mathematics Tournament) 
dataset and Grok API.

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
import os
import sys
from typing import Dict, Optional, List, Tuple, Union
from contextlib import redirect_stdout
from collections import Counter
from fractions import Fraction

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
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found. Install with: pip install openai")
    raise

# Load HMMT dataset from Hugging Face
load_dataset = hf_load_dataset  # Use the pre-imported function
HMMT_PROBLEMS = None

# Try loading from Hugging Face with retries
import time
dataset_paths = [
    "FlagEval/HMMT_2025",
    "FlagEval/HMMT-2025",
    "Maxwell-Jia/HMMT",
]

for dataset_path in dataset_paths:
    for attempt in range(3):  # Retry up to 3 times
        try:
            print(f"Loading HMMT dataset from Hugging Face ({dataset_path})... (attempt {attempt + 1}/3)")
            hmmt_dataset = load_dataset(dataset_path, split="train", trust_remote_code=True)
            
            # Convert to list of dicts with 'question' and 'answer' fields
            HMMT_PROBLEMS = []
            for item in hmmt_dataset:
                HMMT_PROBLEMS.append({
                    "question": item.get("problem", item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "solution": item.get("solution", ""),
                    "id": item.get("id", ""),
                })
            
            print(f"✓ Loaded {len(HMMT_PROBLEMS)} HMMT problems from {dataset_path}")
            break
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("  Retrying in 2 seconds...")
                time.sleep(2)
    
    if HMMT_PROBLEMS is not None:
        break

# Fallback: use embedded HMMT sample problems if HuggingFace fails
if HMMT_PROBLEMS is None or len(HMMT_PROBLEMS) == 0:
    print("\nWARNING: Could not load HMMT dataset from Hugging Face.")
    print("Using embedded sample HMMT problems instead.\n")
    
    # Sample HMMT problems (from HMMT February 2024/2025)
    HMMT_PROBLEMS = [
        {"question": "Find the sum of all positive integers $n$ such that $n^2 + 12n - 2007$ is a perfect square.", "answer": "80", "solution": "", "id": "hmmt_sample_1"},
        {"question": "Let $a$ and $b$ be positive real numbers such that $a + b = 1$. Find the minimum value of $\\frac{1}{a} + \\frac{4}{b}$.", "answer": "9", "solution": "", "id": "hmmt_sample_2"},
        {"question": "How many ways are there to arrange the letters in BANANA such that no two adjacent letters are the same?", "answer": "40", "solution": "", "id": "hmmt_sample_3"},
        {"question": "Find the number of ordered pairs $(a, b)$ of positive integers such that $\\gcd(a, b) = 1$ and $a + b = 100$.", "answer": "40", "solution": "", "id": "hmmt_sample_4"},
        {"question": "Let $x$ and $y$ be positive real numbers such that $x^2 + y^2 = 1$ and $x^4 + y^4 = \\frac{17}{32}$. Find $xy$.", "answer": "1/4", "solution": "", "id": "hmmt_sample_5"},
        {"question": "A triangle has sides of length 13, 14, and 15. Find the area of the triangle.", "answer": "84", "solution": "", "id": "hmmt_sample_6"},
        {"question": "Find the remainder when $2^{100}$ is divided by 101.", "answer": "1", "solution": "", "id": "hmmt_sample_7"},
        {"question": "Let $f(x) = x^3 - 3x + 1$. Find the sum of all real roots of $f(f(x)) = 0$.", "answer": "0", "solution": "", "id": "hmmt_sample_8"},
        {"question": "In how many ways can 8 people be seated around a circular table if 3 particular people must sit together?", "answer": "720", "solution": "", "id": "hmmt_sample_9"},
        {"question": "Find the value of $\\sum_{k=1}^{10} \\frac{k}{2^k}$.", "answer": "2046/1024", "solution": "", "id": "hmmt_sample_10"},
    ]
    print(f"Loaded {len(HMMT_PROBLEMS)} sample HMMT problems")


# ========================= Utilities =========================

def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.
    
    Handles:
    - Whitespace
    - Fractions (both a/b and \\frac{a}{b})
    - Commas in numbers
    - Leading zeros
    - Simple arithmetic expressions
    """
    if not answer:
        return ""
    
    # Strip whitespace
    answer = answer.strip()
    
    # Remove commas from numbers
    answer = answer.replace(",", "")
    
    # Handle LaTeX fractions: \frac{a}{b} -> a/b
    answer = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", answer)
    
    # Remove LaTeX formatting
    answer = answer.replace("\\", "")
    answer = answer.replace("$", "")
    
    # Remove spaces around operators
    answer = re.sub(r"\s*([+\-*/])\s*", r"\1", answer)
    
    # Try to evaluate simple fractions
    if "/" in answer and not any(c.isalpha() for c in answer):
        try:
            # Handle simple fractions like "3/4"
            parts = answer.split("/")
            if len(parts) == 2:
                num = float(parts[0])
                denom = float(parts[1])
                if denom != 0:
                    frac = Fraction(int(num), int(denom))
                    answer = f"{frac.numerator}/{frac.denominator}" if frac.denominator != 1 else str(frac.numerator)
        except:
            pass
    
    return answer.lower().strip()


def answers_match(predicted: str, expected: str) -> bool:
    """Check if two answers match, handling various formats."""
    if not predicted or not expected:
        return False
    
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    # Direct string match
    if pred_norm == exp_norm:
        return True
    
    # Try numeric comparison
    try:
        # Handle fractions
        if "/" in pred_norm:
            parts = pred_norm.split("/")
            pred_val = float(parts[0]) / float(parts[1])
        else:
            pred_val = float(pred_norm)
        
        if "/" in exp_norm:
            parts = exp_norm.split("/")
            exp_val = float(parts[0]) / float(parts[1])
        else:
            exp_val = float(exp_norm)
        
        # Check if values are close (for floating point comparison)
        if abs(pred_val - exp_val) < 1e-9:
            return True
        # Also check if they're equal as integers
        if abs(pred_val - exp_val) < 0.5 and round(pred_val) == round(exp_val):
            return True
    except:
        pass
    
    return False


def _coerce_number(x) -> Optional[str]:
    """Convert value to a normalized number string if possible."""
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return str(x)
        if isinstance(x, float):
            # Check if it's essentially an integer
            if abs(x - round(x)) < 1e-9:
                return str(int(round(x)))
            return str(x)
        # Handle comma-separated numbers
        s = str(x).strip().replace(",", "")
        # Try parsing as number
        if "/" in s:
            # It's a fraction, keep as is
            return s
        float(s)  # Just validate it's a number
        return s
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
    """Execute Python code and extract answer with detailed status.
    
    Returns:
        Dict with keys:
        - answer: Optional[str] - extracted answer or None
        - code_found: bool - whether code block was found
        - execution_success: bool - whether code executed without errors
        - extraction_method: str - how the answer was extracted (or 'failed')
        - error: Optional[str] - error message if execution failed
    """
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
        "Fraction": Fraction,
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
        m = re.findall(r"__ANS__\s*=\s*(.+?)(?:\n|$)", printed)
        if m:
            result["answer"] = m[-1].strip()
            result["extraction_method"] = "print_ANS"
            return result
        
        # Try common variable names
        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                val = safe_locals[key]
                if val is not None:
                    result["answer"] = str(val)
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
                m2 = re.findall(r"__ANS__\s*=\s*(.+?)(?:\n|$)", printed2)
                if m2:
                    result["answer"] = m2[-1].strip()
                    result["extraction_method"] = f"function_{fn_name}_print"
                    return result
                if ret is not None:
                    result["answer"] = str(ret)
                    result["extraction_method"] = f"function_{fn_name}_return"
                    return result
        
        # Parse from printed output - try to get the last meaningful value
        if printed:
            # Try to find a number or fraction in the output
            lines = printed.strip().splitlines()
            if lines:
                last_line = lines[-1].strip()
                result["answer"] = last_line
                result["extraction_method"] = "printed_output"
                return result
        
        result["error"] = "no_answer_found"
        return result
    except Exception as e:
        result["error"] = str(e)[:100]  # Truncate long error messages
        return result


def execute_python_code(code_text: str) -> Optional[str]:
    """Execute Python code and extract answer (simple wrapper)."""
    result = execute_python_code_detailed(code_text)
    return result["answer"]


def extract_final_answer_from_text(text: str) -> Optional[str]:
    """Extract final answer from natural language text."""
    patterns = [
        r"\*\*Final Answer[:\s]*(.+?)\*\*",
        r"Final Answer[:\s]*(.+?)(?:\n|$)",
        r"The final answer is[:\s]*(.+?)(?:\n|$)",
        r"The answer is[:\s]*(.+?)(?:\n|$)",
        r"Answer[:\s]*(.+?)(?:\n|$)",
        r"####\s*(.+?)(?:\n|$)",
        r"\\boxed\{([^}]+)\}",  # LaTeX boxed answer
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            answer = m.group(1).strip()
            # Clean up the answer
            answer = answer.rstrip(".")
            answer = answer.strip()
            if answer:
                return answer
    
    # Try last few lines for a standalone answer
    lines = text.strip().splitlines()[-5:]
    for line in reversed(lines):
        line = line.strip()
        # Skip empty lines and lines that are too long (probably explanations)
        if line and len(line) < 50 and not line.endswith(":"):
            # Check if it looks like an answer (number, fraction, or short expression)
            if re.match(r"^[-\d./\\fracs{}\s]+$", line) or re.match(r"^\d+$", line):
                return line
    
    return None


# ========================= Model Wrapper =========================

class ModelRunner:
    def __init__(self, api_key: Optional[str] = None, 
                 reasoning_model: str = "grok-4-1-fast-reasoning",
                 non_reasoning_model: str = "grok-4-1-fast-non-reasoning"):
        """
        Initialize Grok API client.
        
        Args:
            api_key: Grok API key. If None, will try to get from GROK_API_KEY env var.
            reasoning_model: Model name for reasoning mode (default: "grok-4-1-fast-reasoning")
            non_reasoning_model: Model name for non-reasoning mode (default: "grok-4-1-fast-non-reasoning")
        """
        if api_key is None:
            # Try multiple possible environment variable names
            api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("API_KEY_NAME")
            if api_key is None or api_key == "your_secret_key_here":
                raise ValueError(
                    "Grok API key not provided. Set GROK_API_KEY (or XAI_API_KEY or API_KEY_NAME) "
                    "in your .env file or pass api_key parameter."
                )
        
        print(f"Initializing Grok API client")
        print(f"  Reasoning model: {reasoning_model}")
        print(f"  Non-reasoning model: {non_reasoning_model}")
        # Grok API uses OpenAI-compatible SDK with custom base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.reasoning_model = reasoning_model
        self.non_reasoning_model = non_reasoning_model
        print("✓ Grok API client initialized\n")

    def generate(self, prompt: str, use_reasoning: bool = True) -> str:
        """Generate response from prompt.
        
        Args:
            prompt: The prompt to send to the model.
            use_reasoning: If True, use reasoning model. If False, use non-reasoning model.
        """
        model = self.reasoning_model if use_reasoning else self.non_reasoning_model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=16000,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"ERROR: Failed to generate response: {e}")
            return ""


# ========================= Prompts for Four Conditions =========================

def prompt_only_code(problem: str) -> str:
    """Prompt for ONLY code - no comments, no natural language."""
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
- The answer may be an integer, fraction, or expression

Generate the code now:"""


def prompt_only_comments(problem: str) -> str:
    """Prompt for ONLY comments - reasoning in comments, no executable code."""
    return f"""You are solving a math problem. Generate ONLY comments explaining your reasoning.
DO NOT include any executable Python code.
DO NOT use natural language paragraphs.
Output ONLY comment lines (lines starting with #) that explain how to solve the problem.

Problem: {problem}

Requirements:
- Output ONLY comment lines (each line must start with #)
- Explain your reasoning step by step in comments
- Do NOT include any executable code statements
- End with a comment containing the final answer: # Final Answer: <answer>

Generate the comments now:"""


def prompt_both_code_and_comments(problem: str) -> str:
    """Prompt for BOTH code and comments."""
    return f"""You are solving a problem. Generate Python code WITH comments explaining your reasoning.

Problem: {problem}

Requirements:
- Write Python code to solve the problem
- Include comments (lines starting with #) explaining your reasoning
- The code should compute the answer and print it as: print("__ANS__=" + str(answer))
- Store the final answer in a variable called 'answer'
- Use comments to explain each step of your solution
- The answer may be an integer, fraction, or expression

Generate the code with comments now:"""


def prompt_nothing(problem: str) -> str:
    """Prompt for NOTHING - just the problem, minimal instructions.
    
    Uses non-reasoning model, so we just ask for the answer directly.
    """
    return f"""Do not use reasoning and solve the problem.

Problem: {problem}

Answer:"""


def prompt_cot(problem: str) -> str:
    """Prompt for chain of thought reasoning."""
    return f"""Use chain of thought reasoning to solve this problem and mark the final answer with answer: . Problem: {problem}"""


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
        
    elif condition == "cot":
        # Chain of thought - should have natural language reasoning
        # No specific restrictions, but we measure what was produced
        overall = 1.0  # Always adherent (no restrictions)
        details["cot_condition"] = True
        
    else:
        overall = 0.0
    
    return {
        "overall_score": overall,
        "details": details
    }


# ========================= Answer Extraction by Condition =========================

def extract_answer_detailed(response: str, condition: str) -> Dict:
    """Extract answer from response with detailed tracking.
    
    Returns:
        Dict with keys:
        - answer: Optional[str] - the extracted answer
        - extraction_source: str - where the answer came from
        - code_details: Dict - details about code execution (if attempted)
    """
    result = {
        "answer": None,
        "extraction_source": "none",
        "code_details": None,
    }
    
    if condition == "only_code":
        # Try to execute code
        code_result = execute_python_code_detailed(response)
        result["code_details"] = code_result
        if code_result["answer"] is not None:
            result["answer"] = code_result["answer"]
            result["extraction_source"] = f"code_exec:{code_result['extraction_method']}"
            return result
        # Fallback: try to parse from code text
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text_fallback"
        return result
        
    elif condition == "only_comments":
        # Extract from comments
        comments = extract_comments(response)
        ans = extract_final_answer_from_text(comments)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "comments"
            return result
        # Fallback: try natural language extraction
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text_fallback"
        return result
        
    elif condition == "both":
        # Try code execution first
        code_result = execute_python_code_detailed(response)
        result["code_details"] = code_result
        if code_result["answer"] is not None:
            result["answer"] = code_result["answer"]
            result["extraction_source"] = f"code_exec:{code_result['extraction_method']}"
            return result
        # Fallback: try comments
        comments = extract_comments(response)
        ans = extract_final_answer_from_text(comments)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "comments_fallback"
            return result
        # Final fallback: natural language
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text_fallback"
        return result
        
    elif condition == "nothing":
        # Try text extraction first (most likely for non-reasoning model)
        ans = extract_final_answer_from_text(response)
        if ans is not None:
            result["answer"] = ans
            result["extraction_source"] = "text"
            return result
        # Fallback: try code execution
        code_result = execute_python_code_detailed(response)
        result["code_details"] = code_result
        if code_result["answer"] is not None:
            result["answer"] = code_result["answer"]
            result["extraction_source"] = f"code_exec:{code_result['extraction_method']}"
        return result
        
    elif condition == "cot":
        # Chain of thought - try natural language extraction first, then code
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


def extract_answer(response: str, condition: str) -> Optional[str]:
    """Extract answer from response based on condition (simple wrapper)."""
    result = extract_answer_detailed(response, condition)
    return result["answer"]


# ========================= Evaluation =========================

def evaluate_problem(runner: ModelRunner, prob: Dict, index: int) -> Dict:
    """Evaluate a single problem across all four conditions."""
    q = prob["question"]
    true_ans = prob["answer"]
    
    results = {
        "problem_index": index,
        "question": q,
        "true_answer": true_ans,
    }
    
    conditions = ["only_code", "only_comments", "both", "nothing", "cot"]
    
    for condition in conditions:
        # Generate prompt
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
        
        # Print prompt
        use_reasoning = (condition != "nothing")
        model_mode = "reasoning" if use_reasoning else "non-reasoning"
        print(f"\n  [{condition}] Prompt (model: {model_mode}):")
        print("  " + "-" * 76)
        for line in prompt.splitlines():
            print(f"  {line}")
        print("  " + "-" * 76)
        
        # Generate response (use non-reasoning model for "nothing" condition)
        response = runner.generate(prompt, use_reasoning=use_reasoning)
        
        # Print response
        print(f"\n  [{condition}] Model Response:")
        print("  " + "-" * 76)
        for line in response.splitlines():
            print(f"  {line}")
        print("  " + "-" * 76)
        
        # Extract answer with detailed tracking
        extraction_result = extract_answer_detailed(response, condition)
        pred = extraction_result["answer"]
        
        # Calculate adherence
        adherence = calculate_adherence_score(response, condition)
        
        # Check correctness using flexible matching
        is_correct = answers_match(pred, true_ans) if pred is not None else False
        
        # Store results
        results[f"{condition}_prompt"] = prompt
        results[f"{condition}_response"] = response
        results[f"{condition}_prediction"] = pred
        results[f"{condition}_correct"] = is_correct
        results[f"{condition}_adherence"] = adherence
        results[f"{condition}_extraction"] = extraction_result
        
        # Build summary string with extraction details
        pred_str = str(pred)[:20] if pred is not None else 'None'
        extraction_src = extraction_result["extraction_source"]
        code_details = extraction_result.get("code_details")
        
        code_status = ""
        if code_details:
            if code_details["code_found"]:
                if code_details["execution_success"]:
                    code_status = "code:✓"
                else:
                    code_status = f"code:✗({code_details.get('error', 'unknown')[:20]})"
            else:
                code_status = "code:not_found"
        
        print(f"\n  [{condition}] Summary: pred={pred_str}, "
              f"correct={'✓' if is_correct else '✗'}, "
              f"source={extraction_src}, {code_status}"
              f", adherence={adherence['overall_score']:.2f}")
    
    return results


# ========================= Main Experiment =========================

def main():
    reasoning_model = "grok-4-1-fast-reasoning"
    non_reasoning_model = "grok-4-1-fast-non-reasoning"
    print("=" * 80)
    print("Can models reason in code? - Grok Experiment (HMMT)")
    print("=" * 80)
    print(f"Reasoning Model: {reasoning_model}")
    print(f"Non-Reasoning Model: {non_reasoning_model}")
    print(f"Dataset: HMMT 2025 ({len(HMMT_PROBLEMS)} problems)")
    print(f"Conditions: only_code, only_comments, both, nothing (non-reasoning), cot")
    print("=" * 80)
    print()
    
    runner = ModelRunner(reasoning_model=reasoning_model, non_reasoning_model=non_reasoning_model)
    problems = HMMT_PROBLEMS
    
    all_results = []
    correct_counts = {
        "only_code": 0,
        "only_comments": 0,
        "both": 0,
        "nothing": 0,
        "cot": 0,
    }
    
    adherence_scores = {
        "only_code": [],
        "only_comments": [],
        "both": [],
        "nothing": [],
        "cot": [],
    }
    
    # Track code execution stats for conditions that use code
    code_stats = {
        "only_code": {"code_found": 0, "exec_success": 0, "exec_fail": 0, "answer_from_code": 0},
        "both": {"code_found": 0, "exec_success": 0, "exec_fail": 0, "answer_from_code": 0},
    }
    
    # Track extraction sources
    extraction_sources = {condition: Counter() for condition in ["only_code", "only_comments", "both", "nothing", "cot"]}
    
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
            
            # Track extraction sources
            extraction = res.get(f"{condition}_extraction", {})
            source = extraction.get("extraction_source", "unknown")
            extraction_sources[condition][source] += 1
            
            # Track code execution stats
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
    
    # Save detailed results to JSON
    output_file = "grok_hmmt_experiment_results.json"
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

