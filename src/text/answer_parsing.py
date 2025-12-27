import re
from text.numeric import normalize_number

def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    text = normalize_number(text)

    # 1) GSM8K delimiter
    if "####" in text:
        tail = text.split("####")[-1]
        m = re.search(r"-?\d+(?:\.\d+)?", tail)
        return m.group(0) if m else ""

    # 2) Common model formats (highest priority)
    # Handles: "Final answer: 123", "Final Answer - 123", "Answer: 123", "The answer is 123"
    patterns = [
        r"(?:final\s*answer)\s*[:\-]\s*(-?\d+(?:\.\d+)?)",
        r"(?:answer)\s*[:\-]\s*(-?\d+(?:\.\d+)?)",
        r"(?:the\s*answer\s*is)\s*[:\-]?\s*(-?\d+(?:\.\d+)?)",
    ]
    
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 3) fallback: last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""

