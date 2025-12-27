from datasets import load_dataset
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import re


@dataclass
class GSM8KExample:
    """ Single GSM8K example with Chain-of-Thought """
    question: str
    reasoning: str  # Chain-of-Thought steps
    answer: str  # Final numerical answer
    split: str # Dataset split (train/test/val)
    index: int  # Example index


class GSM8KDatasetLoader:

    def __init__(self, file_path: Path):
        """ Initialize dataset loader """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        self.examples: List[GSM8KExample] = []
        self._load_dataset()

    def _load_dataset(self):
        """ Load and parse the JSON dataset file """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different possible formats
            if isinstance(data, list):
                raw_examples = data
            elif isinstance(data, dict) and 'examples' in data:
                raw_examples = data['examples']
            else:
                raise ValueError("Unknown dataset format")
            
            # Parse examples
            for idx, item in enumerate(raw_examples):
                example = self._parse_example(item, idx)
                if example:
                    self.examples.append(example)
            
            print(f"✓ Loaded {len(self.examples)} examples from {self.file_path.name}")
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
        

    def _parse_example(self, item: Dict, index: int) -> Optional[GSM8KExample]:
        """ Parse a single example from the dataset """
        try:
            # Support multiple field name variations
            question = item.get('question') or item.get('input') or item.get('problem')
            reasoning = item.get('reasoning') or item.get('cot') or item.get('chain_of_thought')
            answer = item.get('answer') or item.get('target') or item.get('solution')
            split = item.get('split', 'unknown')
            
            if not question or not answer:
                print(f"Warning: Skipping example {index} - missing required fields")
                return None
            
            # If reasoning is missing, try to extract from answer field
            if not reasoning and '####' in str(answer):
                parts = str(answer).split('####')
                reasoning = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else answer
            
            # If still no reasoning, use empty string
            reasoning = reasoning or ""
            
            return GSM8KExample(
                question=str(question).strip(),
                reasoning=str(reasoning).strip(),
                answer=str(answer).strip(),
                split=str(split),
                index=index
            )
        
        except Exception as e:
            print(f"Warning: Error parsing example {index}: {e}")
            return None
        

    
    def save_processed_dataset(self, examples: List[GSM8KExample], output_path: Path):
        """Save processed examples to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        payload = []
        for ex in examples:
            d = asdict(ex)

            if "reasoning" in d and isinstance(d["reasoning"], str):
                d["reasoning"] = self._clean_reasoning(d["reasoning"])

            payload.append(d)
        
        with open(output_file, 'w') as f:
            json.dump(payload, f, indent=2)
        
        print(f"Saved to {output_path}")
    
    def _clean_reasoning(self, text: str,) -> str:
        """
        strategy: remove everything between <<...>> but keeping the resulting value
        Example: $<<12/60=0.2>>0.2  -> $0.2
        """
        if not text:
            return text

        # Esempio: $<<12/60=0.2>>0.2  -> $0.2
        return re.sub(r"<<.*?>>", "", text)

    def __len__(self) -> int:
        """ Return number of examples """
        return len(self.examples)
    
    def __getitem__(self, index: int) -> GSM8KExample:
        """ Get example by index """
        return self.examples[index]
    
    def __iter__(self):
        """ Iterate over examples """
        return iter(self.examples)
    
    def get_statistics(self) -> Dict:
        """ Get dataset statistics. """
        if not self.examples:
            return {}
        
        question_lengths = [len(ex.question.split()) for ex in self.examples]
        reasoning_lengths = [len(ex.reasoning.split()) for ex in self.examples]
        answer_lengths = [len(ex.answer.split()) for ex in self.examples]
        
        return {
            'total_examples': len(self.examples),
            'split': self.examples[0].split if self.examples else 'unknown',
            'question_length': {
                'mean': sum(question_lengths) / len(question_lengths),
                'min': min(question_lengths),
                'max': max(question_lengths),
            },
            'reasoning_length': {
                'mean': sum(reasoning_lengths) / len(reasoning_lengths),
                'min': min(reasoning_lengths),
                'max': max(reasoning_lengths),
            },
            'answer_length': {
                'mean': sum(answer_lengths) / len(answer_lengths),
                'min': min(answer_lengths),
                'max': max(answer_lengths),
            },
            'has_reasoning': sum(1 for ex in self.examples if ex.reasoning) / len(self.examples) * 100,
        }
    
    def print_statistics(self):
        """ Print dataset statistics in a readable format """
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"File: {self.file_path.name}")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Split: {stats['split']}")
        print(f"\nQuestion length (words):")
        print(f"  Mean: {stats['question_length']['mean']:.1f}")
        print(f"  Range: [{stats['question_length']['min']}, {stats['question_length']['max']}]")
        print(f"\nReasoning length (words):")
        print(f"  Mean: {stats['reasoning_length']['mean']:.1f}")
        print(f"  Range: [{stats['reasoning_length']['min']}, {stats['reasoning_length']['max']}]")
        print(f"\nExamples with reasoning: {stats['has_reasoning']:.1f}%")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Carica il dataset di training
    print("Loading training data...")
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent

    train_path = ROOT_DIR / "gsm8k-distillation" / "data" / "raw" / "gsm8k_cot_train.json"
    test_path = ROOT_DIR / "gsm8k-distillation" / "data" / "raw" / "gsm8k_cot_test.json"

    train_dataset = GSM8KDatasetLoader(train_path)
    train_dataset.print_statistics()
    
    # Salva il dataset di training processato
    train_output_path = ROOT_DIR / "gsm8k-distillation" / "data" / "processed" / "gsm8k_train_processed.json"
    train_dataset.save_processed_dataset(train_dataset.examples, train_output_path)
    
    # Carica il dataset di test
    print("Loading test data...")
    test_dataset = GSM8KDatasetLoader(test_path)
    test_dataset.print_statistics()
    
    # Salva il dataset di test processato
    test_output_path =  ROOT_DIR / "gsm8k-distillation" / "data" / "processed" / "gsm8k_test_processed.json"
    test_dataset.save_processed_dataset(test_dataset.examples, test_output_path)
    
    print("✓ All datasets processed and saved successfully.")