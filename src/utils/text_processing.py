import re

def clean_reasoning(r: str) -> str:
    if not r:
        return r
    
    # Remove any special tokens like <<...>>
    r = re.sub(r"<<[^>]*>>", "", r)

    # Remove any text within parentheses
    r = re.sub(r"[ \t]+", " ", r).strip()
    return r