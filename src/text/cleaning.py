import re

def clean_reasoning(text: str,) -> str:
    """
    strategy: remove everything between <<...>> but keeping the resulting value
    Example: $<<12/60=0.2>>0.2  -> $0.2
    """
    if not text:
        return text

    # Esempio: $<<12/60=0.2>>0.2  -> $0.2
    return re.sub(r"<<.*?>>", "", text)
    