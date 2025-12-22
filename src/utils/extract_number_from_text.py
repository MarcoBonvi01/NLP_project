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
    if s is None:
        return None
    s = s.strip().replace("€", "").replace("$", "")
    s = s.replace(",", ".").strip().strip(".")
    if not s:
        return None
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
        return str(v)
    except:
        return None
    

def extract_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    # 1) priorità assoluta: ultima occorrenza dopo ####
    hits = _HASH_RE.findall(text)
    if hits:
        return normalize_number(hits[-1])

    # 2) fallback: ultimo numero nel testo
    nums = _NUM_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])

    return None

def exact_match(pred_text: Optional[str], gold_answer: Optional[str]) -> bool:
    p = extract_answer(pred_text)
    g = normalize_number(gold_answer)
    return (p is not None) and (g is not None) and (p == g)