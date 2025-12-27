from dataclasses import dataclass
@dataclass
class GSM8KExample:
    """ Single GSM8K example with Chain-of-Thought """
    question: str
    reasoning: str  # Chain-of-Thought steps
    answer: str  # Final numerical answer
    split: str # Dataset split (train/test/val)
    index: int  # Example index