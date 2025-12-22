from datasets import load_dataset
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import statistics
import argparse

@dataclass
class GSM8KExample:
    """ Single GSM8K example with Chain-of-Thought """
    question: str
    reasoning: str  # Chain-of-Thought steps
    answer: str  # Final numerical answer
    split: str # Dataset split (train/test/val)
    index: int  # Example index


class GSM8KDownloader:
    def __init__(self, base_path: str = "gsm8k-distillation/data/raw"):
        self.output_path = Path(base_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def download_and_prepare_gsm8k(self):
        # Download both splits
        for split_name in ["train", "test"]:
            print(f"\nProcessing {split_name} split...")
            
            # Load from HuggingFace
            dataset = load_dataset("gsm8k", "main", split=split_name)
            print(f"✓ Downloaded {len(dataset)} examples")
            
            # Convert to our format
            index = 0
            examples = []

            for item in dataset:
                question = item['question']
                full_answer = item['answer']
                
                # Parse the answer field: "reasoning #### number"
                if '####' in full_answer:
                    reasoning_part, answer_part = full_answer.split('####')
                    reasoning = reasoning_part.strip()
                    answer = answer_part.strip()
                else:
                    # Fallback if no #### marker
                    reasoning = full_answer
                    answer = ""
                
                examples.append(GSM8KExample(
                    question=question,
                    reasoning=reasoning,
                    answer=answer,
                    split=split_name,
                    index=index
                ))

                index += 1
            
            # Save to JSON
            output_file = self.output_path / f"gsm8k_cot_{split_name}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(ex) for ex in examples], f, indent=2, ensure_ascii=False)
            
            print(f" Saved {len(examples)} examples to {output_file}")
            
            # Show example
            if examples:
                print(f"\n--- Example from {split_name} split ---")
                ex = examples[0]
                print(f"Question: {ex.question[:80]}...")
                print(f"Reasoning: {ex.reasoning[:100]}...")
                print(f"Answer: {ex.answer}")
                print("-" * 40)
        
        print(f"\nFiles created:")
        print(f"{self.output_path}/gsm8k_cot_train.json")
        print(f"{self.output_path}/gsm8k_cot_test.json")


    def analyze_dataset_quality(self, json_path: str):
        """ Analyze the quality of Chain-of-Thought in the dataset """
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print("\n" + "="*60)
        print("DATASET QUALITY ANALYSIS")
        print("="*60)
        
        # Count examples with reasoning
        has_reasoning = sum(1 for ex in data if ex['reasoning'])
        reasoning_lengths = [len(ex['reasoning'].split()) for ex in data if ex['reasoning']]
        
        print(f"\nFile: {json_path}")
        print(f"Total examples: {len(data)}")
        print(f"Examples with reasoning: {has_reasoning} ({has_reasoning/len(data)*100:.1f}%)")
        
        if reasoning_lengths:
            print(f"\nReasoning length (words):")
            print(f"  Mean: {statistics.mean(reasoning_lengths):.1f}")
            print(f"  Median: {statistics.median(reasoning_lengths):.1f}")
            print(f"  Min: {min(reasoning_lengths)}")
            print(f"  Max: {max(reasoning_lengths)}")
        
        # Show examples of different reasoning lengths
        print("\n" + "="*60)
        print("REASONING EXAMPLES")
        print("="*60)
        
        # Short reasoning
        short_examples = sorted(data, key=lambda x: len(x['reasoning'].split()))[:1]
        print("\n📝 Shortest reasoning:")
        for ex in short_examples:
            print(f"Q: {ex['question'][:60]}...")
            print(f"R: {ex['reasoning']}")
            print(f"A: {ex['answer']}\n")
        
        # Long reasoning
        long_examples = sorted(data, key=lambda x: len(x['reasoning'].split()), reverse=True)[:1]
        print("📝 Longest reasoning:")
        for ex in long_examples:
            print(f"Q: {ex['question'][:60]}...")
            print(f"R: {ex['reasoning'][:200]}...")
            print(f"A: {ex['answer']}\n")
        
        print("="*60)


    def verify_dataset_format(self, json_path: str) -> bool:
        """ Verify that the JSON file has the correct format for GSM8KDatasetLoader. """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check it's a list
            if not isinstance(data, list):
                print(f" Error: Expected list, got {type(data)}")
                return False
            
            # Check first item has required fields
            if data:
                required_fields = {'question', 'reasoning', 'answer', 'split'}
                first_item = data[0]
                
                if not all(field in first_item for field in required_fields):
                    missing = required_fields - set(first_item.keys())
                    print(f" Error: Missing fields: {missing}")
                    return False
            
            print(f" Format verified: {len(data)} examples with correct structure")
            return True
        
        except Exception as e:
            print(f" Error reading file: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare GSM8K dataset")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze dataset quality after downloading")
    parser.add_argument("--verify", type=str, default=None,
                       help="Verify format of existing JSON file")
    
    args = parser.parse_args()

    processor = GSM8KDownloader()
    
    if args.verify:
        # In that case we don't need to download anything because we just want to verify the format
        # of an existing file indicated by the user
        processor.verify_dataset_format(args.verify)
    else:
        # Download and prepare dataset
        processor.download_and_prepare_gsm8k()
        
        # Analyze if requested
        if args.analyze:
            train_file = Path("gsm8k-distillation/data/raw") / "gsm8k_cot_train.json"

            # At the end of the process we analyze it for quality checks
            if train_file.exists():
                processor.analyze_dataset_quality(str(train_file))
        
        print("\n All done")
