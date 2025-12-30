from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from functools import partial
from text.cleaning import clean_reasoning
from text.answer_parsing import extract_final_answer
from metrics.gsm8k_metrics import compute_gsm8k_metrics, exact_match

@dataclass
class StudentModelConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"

    # Model hyperparameters
    max_source_length: int = 768
    max_target_length: int = 512
    
    # Prompt templates (customize as needed)
    input_prefix: str = "Solve the math word problem step by step.\n" \
                        "The problem is:\n"

    # For answer-only training, we format target as: "Final answer: <num>"
    answer_prefix: str = "Final answer: "

    # For CoT distillation, we format target as:
    reasoning_prefix: str = "Reasoning:\n"

class StudentModel:
    def __init__(self, config: Optional[StudentModelConfig] = None, device: Optional[str] = None):
        self.config = config or StudentModelConfig()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Qwen spesso non ha pad_token: lo allinei all’eos
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Per batch generation con causal LM è meglio padding a sinistra
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ----------------------------
    # Data loading / formatting
    # ----------------------------
    @staticmethod
    def load_processed_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
        # Load a JSON file containing a list of examples and return as list of dicts
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if data is a list
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list at {path}, got {type(data)}")
        
        return data

    # ----------------------------
    # Data processing
    # ----------------------------
    def format_input(self, question: str) -> str:
        # Produce input string for the model
        # Format the input question with prefix to match model expectations
        return f"{self.config.input_prefix}{question.strip()}"

    # ----------------------------
    # Target formatting
    # ----------------------------
    def format_target(self, reasoning: str | None, answer: str) -> str:
        """
        Standard target:
        - reasoning (optional) prefixed by 'Reasoning:'
        - parsable final row: 'Final answer: <answer>'
        """
        ans = str(answer).strip()
        if reasoning and reasoning.strip():
            r = reasoning.strip()
            return f"{self.config.reasoning_prefix}{r}\n\n{self.config.answer_prefix}{ans}"
        else:
            return f"{self.config.answer_prefix}{ans}"

    # ----------------------------
    # Dataset building
    # ----------------------------
    def build_hf_dataset(
        self,
        examples: List[Dict[str, Any]],
        supervision: str = "cot",  # "cot" or "answer"
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Dataset:
        # Build a HuggingFace Dataset with tokenized inputs and targets
        # Tokenization is done using the model's tokenizer
        # supervision: "cot" for chain-of-thought, "answer" for answer-only


        # Apply limit
        if limit is not None:
            examples = examples[:limit]

        # Create Dataset
        ds = Dataset.from_list(examples)

        # Shuffle if needed
        if shuffle:
            ds = ds.shuffle(seed=seed)

        def _map(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            inputs = [self.format_input(q) for q in batch["question"]]

            # Format targets based on supervision type
            if supervision == "answer":
                # in that case we are interested only in answer 
                targets = [self.format_target(None, a) for a in batch["answer"]]
            elif supervision == "cot":
                # in that case we are interested also in chain of tought
                reasons = [clean_reasoning(x) for x in batch.get("reasoning", [""] * len(batch["question"]))]

                # answer may be missing in some processed formats
                targets = [self.format_target(r, a) for r, a in zip(reasons, batch["answer"])]
            else:
                raise ValueError("supervision must be 'answer' or 'cot'")
            # ---- SFT tokenization for causal LM (Qwen) ----
            # We concatenate prompt + target and compute loss only on the target part
            full_texts = [p + t for p, t in zip(inputs, targets)]

            tok = self.tokenizer(
                full_texts,
                max_length=self.config.max_source_length + self.config.max_target_length,
                truncation=True,
            )

            # Tokenize prompts separately to get prompt lengths (for label masking)
            prompt_tok = self.tokenizer(
                inputs,
                max_length=self.config.max_source_length,
                truncation=True,
            )

            labels = []
            for ids, p_ids in zip(tok["input_ids"], prompt_tok["input_ids"]):
                lab = list(ids)
                prompt_len = len(p_ids)
                # Mask the prompt tokens so loss is computed only on the generated target
                lab[:prompt_len] = [-100] * prompt_len
                labels.append(lab)

            tok["labels"] = labels
            return tok

        # Remove original columns to avoid trainer complaints; keep only tensors
        tokenized = ds.map(_map, batched=True, remove_columns=ds.column_names)

        # Return tokenized dataset
        return tokenized

    # ----------------------------
    # Training / evaluation
    # ----------------------------
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Union[str, Path] = "outputs/student_model",
        *,
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        num_train_epochs: float = 1.0,
        
        
        save_strategy: str = "steps",
        save_total_limit: int = 3,
        report_to: Union[str, List[str]] = "none",

        logging_steps: int = 50,
        eval_steps: int = 200,
        save_steps: int = 200,

        fp16: bool = False,
        bf16: bool = True,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
    ) -> Trainer:
        # Prepare output directory
        output_dir = str(output_dir)

        do_eval = eval_dataset is not None

        # Se vuoi best model, tieni save_steps allineato a eval_steps (consiglio: uguale)
        if do_eval:
            save_steps = save_steps or eval_steps

        # Training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            eval_steps=eval_steps if do_eval else None,
            save_steps=save_steps,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            fp16=fp16 and torch.cuda.is_available(),
            bf16=bf16 and torch.cuda.is_available(),
            report_to=report_to,
        )

        # Data collator for seq2seq
        def _collate(features):
            # 1) Pad solo gli input (input_ids/attention_mask)
            #    (tokenizer.pad gestisce bene questi)
            inputs = [{"input_ids": f["input_ids"], "attention_mask": f.get("attention_mask")} for f in features]
            batch = self.tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
            )

            # 2) Pad labels MANUALMENTE alla stessa lunghezza di input_ids
            if "labels" in features[0]:
                max_len = batch["input_ids"].shape[1]
                labels = torch.full((len(features), max_len), -100, dtype=torch.long)

                for i, f in enumerate(features):
                    lab = f["labels"]
                    # lab deve essere lista di int
                    if isinstance(lab, torch.Tensor):
                        lab = lab.tolist()
                    lab_len = min(len(lab), max_len)
                    labels[i, :lab_len] = torch.tensor(lab[:lab_len], dtype=torch.long)

                batch["labels"] = labels

            return batch

        data_collator = _collate

        # Seq2SeqTrainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )


        # Start training
        trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )

        trainer.save_model(output_dir)                # save weights/config
        self.tokenizer.save_pretrained(output_dir)    # save tokenizer

        # Return trainer object
        return trainer

    @torch.inference_mode()
    def generate(self, questions: List[str], max_new_tokens: int = 128, num_beams: int = 1) -> List[str]:
        # Generate outputs for a list of input questions

        self.model.eval()
        
        # Format inputs
        inputs = [self.format_input(q) for q in questions]

        # Tokenize inputs
        enc = self.tokenizer(
            inputs,
            return_tensors="pt", # return as PyTorch tensors
            padding=True,
            truncation=True,
            max_length=self.config.max_source_length,
        ).to(self.device)

        # Generate outputs
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # taglia via il prompt: tieni solo i token generati
        gen_only = out[:, enc["input_ids"].shape[1]:]

        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)


    def evaluate_exact_match(
        self,
        raw_examples: List[Dict[str, Any]],
        *,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Evaluate exact match (EM) on a list of raw examples
        # raw_examples: list of dicts with "question" and "answer" fields
        # Returns a dict with number of examples, EM score, and some sample predictions 

        if limit is not None:
            raw_examples = raw_examples[:limit]

        # Extract questions
        questions = [ex["question"] for ex in raw_examples]

        # Extract gold answers that are final numbers only
        gold = [extract_final_answer(str(ex.get("answer", ""))) for ex in raw_examples]

        # Generate predictions
        generations = self.generate(questions, max_new_tokens=max_new_tokens, num_beams=num_beams)

        # Extract final numbers from predictions
        pred_nums = [extract_final_answer(g) for g in generations]

        # Compute exact match
        em = sum(int(exact_match(p, ga)) for p, ga in zip(pred_nums, gold)) / max(1, len(gold))
        
        return {
            "n": len(raw_examples),
            "exact_match": em,
            "pred_examples": list(zip(questions[:5], generations[:5], pred_nums[:5], gold[:5]))
        }

    # ----------------------------
    # Save / load
    # ----------------------------
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "StudentModel":
        path = Path(path)
        config = StudentModelConfig(model_name=str(path))
        obj = cls(config=config, device=device)
        return obj
