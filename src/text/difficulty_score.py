from data.gsm8k_types import Complexity

def _count_ops(text: str) -> int:
    return sum(text.count(ch) for ch in ["+", "-", "*", "/", "%"])

def _count_numbers(text: str) -> int:
    import re
    return len(re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")))


def compute_complexity(question: str, reasoning: str) -> Complexity:
    q = question or ""
    r = reasoning or ""

    return Complexity(
        ops=_count_ops(r),
        nums=_count_numbers(r),
    )

def tag_difficulty(c: Complexity) -> str:
    # euristiche semplici (tweakabili)
    ops, nums = c.ops, c.nums
    
    if ops <= 2 and nums <= 6:
        return "easy"
    
    elif ops <= 4:
        return "medium"
    
    return "hard"