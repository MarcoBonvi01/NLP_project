import requests
import json
from pathlib import Path
from typing import Dict

class OllamaPlanGenerator:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url
        
        self.system_prompt = """You are a math planning assistant.
Create:
1) PLAN: logical steps (no calculations)
2) EXPRESSION: final formula with numbers

Format:
PLAN:
- step 1
- step 2

EXPRESSION:
formula"""

    def generate_plan_and_expression(self, question: str, reasoning: str = "") -> Dict[str, str]:
        
        prompt = f"{self.system_prompt}\n\nQuestion: {question}"
        if reasoning:
            prompt += f"\nReasoning: {reasoning}"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 400
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                plan, expression = self._parse_response(response_text)
                
                return {
                    "plan": plan,
                    "expression": expression,
                    "raw_response": response_text
                }
            else:
                print(f"Error: {response.status_code}")
                return {"plan": "", "expression": "", "raw_response": ""}
        
        except Exception as e:
            print(f"Error: {e}")
            return {"plan": "", "expression": "", "raw_response": ""}
    
    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse response"""
        plan = ""
        expression = ""
        
        lines = response.split('\n')
        current_section = None
        plan_lines = []
        expr_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if 'PLAN:' in line_stripped.upper():
                current_section = 'plan'
                continue
            elif 'EXPRESSION:' in line_stripped.upper():
                current_section = 'expression'
                continue
            
            if current_section == 'plan' and line_stripped:
                plan_lines.append(line_stripped)
            elif current_section == 'expression' and line_stripped:
                expr_lines.append(line_stripped)
        
        plan = '\n'.join(plan_lines)
        expression = ' '.join(expr_lines)
        
        return plan, expression
    
    def augment_dataset(self, input_path: Path, output_path: Path, max_examples: int = None):
        """Augment dataset locally with Ollama"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_examples:
            data = data[:max_examples]
        
        print(f"Augmenting {len(data)} examples with Ollama ({self.model})...")
        
        augmented_examples = []
        
        for idx, item in enumerate(data):
            print(f"Processing {idx + 1}/{len(data)}...", end='\r')
            
            result = self.generate_plan_and_expression(
                question=item['question'],
                reasoning=item.get('reasoning', '')
            )
            
            item['plan'] = result['plan']
            item['expression'] = result['expression']
            augmented_examples.append(item)
        
        print(f"\n✓ Completed {len(augmented_examples)} examples")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_examples, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to {output_path}")
        return augmented_examples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    
    args = parser.parse_args()
    
    generator = OllamaPlanGenerator(model=args.model)
    augmented = generator.augment_dataset(
        Path(args.input), 
        Path(args.output),
        args.max_examples
    )