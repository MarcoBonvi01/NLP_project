from typing import Optional
import re

_HASH_RE = re.compile(r"####\s*([-+]?\d+(?:[.,]\d+)?)")
_NUM_RE  = re.compile(r"([-+]?\d+(?:[.,]\d+)?)")

def normalize_number(s: Optional[str]) -> Optional[str]:
    """
    Extract the last occurring number in the text.
    GSM8K answers are typically integers, but keep float support.
    Returns the number as string, or None if not found.
    """
    if not s:
        return ""
    
    # Remove commas and spaces
    cleaned = re.sub(r'[,\s]', '', str(s))

    try:
        # Convert to float and then to string to normalize
        return str(float(cleaned))
    except:
        return cleaned
    

def extract_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    
    text = text.strip()

    # Pattern 1: "The answer is X"
    match = re.search(r'(?:the answer is|answer:)\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))
    
    # Pattern 2: "#### X" (formato GSM8K originale)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return normalize_number(match.group(1))
    
    # Pattern 3: Ultimo numero nel testo
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return normalize_number(numbers[-1])
    
    return ""

def exact_match(pred_text: Optional[str], gold_answer: Optional[str]) -> bool:
    pred_norm = normalize_number(pred_text)
    gold_norm = normalize_number(gold_answer)

    if not pred_norm or not gold_norm:
        return False
    
    try:
        return float(pred_norm) == float(gold_norm)
    except:
        return pred_norm == gold_norm