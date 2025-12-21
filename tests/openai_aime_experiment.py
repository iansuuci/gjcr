#!/usr/bin/env python3
"""
OpenAI Experiment: Can models reason in code?

This experiment measures the effect of code vs comments vs both vs nothing
on model performance using the AIME condensed dataset and OpenAI API.

Uses gpt-5.1 with reasoning effort toggle:
- reasoning={"effort": "high"} for reasoning mode
- reasoning={"effort": "none"} for non-reasoning mode

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
from typing import Dict, Optional, List, Tuple
from contextlib import redirect_stdout
from collections import Counter

# Add project root to path for imports
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

# Load AIME dataset
try:
    from datasets.aime_problems import AIME_2024_PROBLEMS
    print(f"Loaded {len(AIME_2024_PROBLEMS)} AIME 2024 problems")
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

def execute_python_code_detailed(code_text: str) -> Dict:
    """Execute Python code and extract integer answer with detailed status.
    
    Returns:
        Dict with keys:
        - answer: Optional[int] - extracted answer or None
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
        m = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed)
        if m:
            result["answer"] = _normalize_aime_int(int(m[-1]))
            result["extraction_method"] = "print_ANS"
            return result
        
        # Try common variable names
        for key in ("answer", "result", "final_answer", "ans", "solution", "output"):
            if key in safe_locals:
                v = _coerce_int(safe_locals[key])
                if v is not None:
                    result["answer"] = _normalize_aime_int(v)
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
                m2 = re.findall(r"__ANS__\s*=\s*(-?\d+)", printed2)
                if m2:
                    result["answer"] = _normalize_aime_int(int(m2[-1]))
                    result["extraction_method"] = f"function_{fn_name}_print"
                    return result
                v = _coerce_int(ret)
                if v is not None:
                    result["answer"] = _normalize_aime_int(v)
                    result["extraction_method"] = f"function_{fn_name}_return"
                    return result
        
        # Parse from printed output
        nums = re.findall(r"-?\d{1,3}", printed)
        if nums:
            result["answer"] = _normalize_aime_int(int(nums[-1]))
            result["extraction_method"] = "printed_number"
            return result
        
        result["error"] = "no_answer_found"
        return result
    except Exception as e:
        result["error"] = str(e)[:100]  # Truncate long error messages
        return result


def execute_python_code(code_text: str) -> Optional[int]:
    """Execute Python code and extract integer answer (simple wrapper)."""
    result = execute_python_code_detailed(code_text)
    return result["answer"]

def extract_final_answer_from_text(text: str) -> Optional[int]:
    """Extract final answer from natural language text."""
    patterns = [
        r"\\boxed\{(\d{1,3})\}",  # LaTeX boxed answer (highest priority)
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
    
    # Try last few numbers in text
    tail = "\n".join(text.strip().splitlines()[-15:])
    nums = re.findall(r"-?\d{1,3}", tail)
    if nums:
        v = _coerce_int(nums[-1])
        return _normalize_aime_int(v)
    
    return None


# ========================= Model Wrapper =========================

class ModelRunner:
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "gpt-5.1",
                 reasoning_effort: str = "high",
                 no_reasoning_effort: str = "none"):
        """
        Initialize OpenAI API client with reasoning effort toggle.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
            model: Model name (default: "gpt-5.1")
            reasoning_effort: Effort level for reasoning mode (default: "high")
            no_reasoning_effort: Effort level for non-reasoning mode (default: "none")
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None or api_key == "your_secret_key_here":
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY "
                    "in your .env file or pass api_key parameter."
                )
        
        print(f"Initializing OpenAI API client")
        print(f"  Model: {model}")
        print(f"  Reasoning effort (reasoning): {reasoning_effort}")
        print(f"  Reasoning effort (non-reasoning): {no_reasoning_effort}")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.no_reasoning_effort = no_reasoning_effort
        print("✓ OpenAI API client initialized\n")

    def generate(self, prompt: str, use_reasoning: bool = True) -> str:
        """Generate response from prompt.
        
        Args:
            prompt: The prompt to send to the model.
            use_reasoning: If True, use high reasoning effort. If False, use no reasoning.
        """
        effort = self.reasoning_effort if use_reasoning else self.no_reasoning_effort
        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                reasoning={"effort": effort},
                max_output_tokens=16000,  # Increased to allow for reasoning + output
            )
            
            # Debug: Check response status and details
            if hasattr(response, 'status') and response.status != 'completed':
                print(f"DEBUG: Response status: {response.status}")
                if hasattr(response, 'incomplete_details') and response.incomplete_details:
                    print(f"DEBUG: Incomplete details: {response.incomplete_details}")
            
            # Try multiple ways to extract the response content
            # Method 1: Try output_text property (convenience accessor)
            if hasattr(response, 'output_text') and response.output_text:
                return response.output_text.strip()
            
            # Method 2: Try the text property directly (only if it's a string)
            if hasattr(response, 'text') and isinstance(response.text, str) and response.text:
                return response.text.strip()
            
            # Method 3: Iterate through output items to find text content
            if hasattr(response, 'output') and response.output:
                text_parts = []
                for item in response.output:
                    item_type = getattr(item, 'type', None)
                    
                    # Skip reasoning items, look for message/text items
                    if item_type == 'message':
                        if hasattr(item, 'content') and item.content:
                            for content_block in item.content:
                                if hasattr(content_block, 'text'):
                                    text_parts.append(content_block.text)
                    elif item_type == 'text':
                        if hasattr(item, 'text'):
                            text_parts.append(item.text)
                    # Also check for 'output_text' type
                    elif item_type == 'output_text':
                        if hasattr(item, 'text'):
                            text_parts.append(item.text)
                
                if text_parts:
                    return "\n".join(text_parts).strip()
            
            # If we get here, debug the issue
            print(f"DEBUG: Response status: {getattr(response, 'status', 'unknown')}")
            print(f"DEBUG: output_text: {repr(getattr(response, 'output_text', None))}")
            print(f"DEBUG: text: {repr(getattr(response, 'text', None))}")
            if hasattr(response, 'output'):
                print(f"DEBUG: Output items ({len(response.output)}):")
                for i, item in enumerate(response.output):
                    print(f"  [{i}] type={getattr(item, 'type', '?')}, attrs={[a for a in dir(item) if not a.startswith('_')]}")
            if hasattr(response, 'incomplete_details') and response.incomplete_details:
                print(f"DEBUG: Incomplete details: {response.incomplete_details}")
            
            return ""
        except Exception as e:
            print(f"ERROR: Failed to generate response: {e}")
            import traceback
            traceback.print_exc()
            return ""


# ========================= Prompts for Four Conditions =========================

def prompt_only_code(problem: str) -> str:
    """Prompt for ONLY code - no comments, no natural language."""
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
- End with a comment containing the final answer: # Final Answer: <number>

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

Generate the code with comments now:"""

def prompt_nothing(problem: str) -> str:
    """Prompt for NOTHING - just the problem, minimal instructions.
    
    Uses non-reasoning mode (effort="none"), so we just ask for the answer directly.
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
        - answer: Optional[int] - the extracted answer
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
        # Try text extraction first (most likely for non-reasoning mode)
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


def extract_answer(response: str, condition: str) -> Optional[int]:
    """Extract answer from response based on condition (simple wrapper)."""
    result = extract_answer_detailed(response, condition)
    return result["answer"]


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
        effort = runner.reasoning_effort if use_reasoning else runner.no_reasoning_effort
        print(f"\n  [{condition}] Prompt (reasoning effort: {effort}):")
        print("  " + "-" * 76)
        for line in prompt.splitlines():
            print(f"  {line}")
        print("  " + "-" * 76)
        
        # Generate response
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
        
        # Store results
        results[f"{condition}_prompt"] = prompt
        results[f"{condition}_response"] = response
        results[f"{condition}_prediction"] = pred
        results[f"{condition}_correct"] = (pred == true_ans) if pred is not None else False
        results[f"{condition}_adherence"] = adherence
        results[f"{condition}_extraction"] = extraction_result
        
        # Build summary string with extraction details
        pred_str = str(pred) if pred is not None else 'None'
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
        
        print(f"\n  [{condition}] Summary: pred={pred_str:>3s}, "
              f"correct={'✓' if (pred == true_ans) else '✗'}, "
              f"source={extraction_src}, {code_status}"
              f", adherence={adherence['overall_score']:.2f}")
    
    return results


# ========================= Main Experiment =========================

def main():
    model = "gpt-5.1"
    reasoning_effort = "high"
    no_reasoning_effort = "none"
    
    print("=" * 80)
    print("Can models reason in code? - OpenAI Experiment")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Reasoning effort (reasoning): {reasoning_effort}")
    print(f"Reasoning effort (non-reasoning): {no_reasoning_effort}")
    print(f"Dataset: AIME 2024 Condensed ({len(AIME_2024_PROBLEMS)} problems)")
    print(f"Conditions: only_code, only_comments, both, nothing (no reasoning), cot")
    print("=" * 80)
    print()
    
    runner = ModelRunner(
        model=model,
        reasoning_effort=reasoning_effort,
        no_reasoning_effort=no_reasoning_effort
    )
    problems = AIME_2024_PROBLEMS
    
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
    output_file = "openai_aime_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total_problems": total,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "no_reasoning_effort": no_reasoning_effort,
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

