import re

def extract_final_answer(text: str) -> str:
    """
    Extract final numeric answer from generated text.
    
    Handles multiple formats:
    - "Final answer: 42"
    - "#### 42"
    - "The answer is 42"
    - Just "42" at the end
    
    Returns the numeric answer as a string, or empty string if not found.
    """
    if not text:
        return ""
    
    text = str(text).strip()
    
    # Pattern 1: "Final answer: X" or "Final Answer: X"
    match = re.search(r'[Ff]inal [Aa]nswer\s*:\s*(-?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # Pattern 2: "#### X" (GSM8K standard format)
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # Pattern 3: "The answer is X"
    match = re.search(r'[Tt]he answer is\s*(-?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # Pattern 4: "Answer: X"
    match = re.search(r'[Aa]nswer\s*:\s*(-?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # Fallback: Find the last number in the text
    # This handles cases where the model just outputs the number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return ""
