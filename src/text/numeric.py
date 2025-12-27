
from typing import Optional
import re

def normalize_number(text: str) -> str:
    """
    1) Remove thousands separators: 1,234,567 -> 1234567 (only safe 3-digit groups)
    2) Expand k/K: 25k -> 25000, 2.5k -> 2500
    Keeps currency symbols as-is (e.g., $50,000 -> $50000)
    """
    if not text:
        return text

    # --- 1) Remove thousands separators (safe groups of 3 digits) ---
    # Replace commas that are between digits and followed by exactly 3 digits (possibly repeated).
    # Example: 1,234 -> 1234 ; 12,345,678 -> 12345678 ; but avoids "1,2" style.
    def _drop_thousands_commas(s: str) -> str:
        # Iteratively remove commas that match the pattern to handle multiple commas.
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
        return s

    text = _drop_thousands_commas(text)

    # --- 2) Expand k/K suffix ---
    # Handles: 25k, 25K, 2.5k, 2.50K. Optional space: "25 k".
    def _k_repl(m: re.Match) -> str:
        num_str = m.group(1)
        val = float(num_str) * 1000.0
        # If integer-ish, print as int; else keep compact without trailing zeros
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        out = f"{val:.10f}".rstrip("0").rstrip(".")
        return out

    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*[kK]\b", _k_repl, text)

    return text