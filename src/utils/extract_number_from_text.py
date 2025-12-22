from typing import Optional
import re

# Regex to extract final number from text
_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

def extract_final_number(text: str) -> Optional[str]:
    """
    Extract the last occurring number in the text.
    GSM8K answers are typically integers, but keep float support.
    Returns the number as string, or None if not found.
    """
    if text is None:
        return None
    
    matches = _NUM_RE.findall(text.replace(",", ""))

    return matches[-1] if matches else None
