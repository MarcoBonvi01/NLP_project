"""
StudentModel: small seq2seq model wrapper for GSM8K distillation. 

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from functools import partial
from text.cleaning import clean_reasoning
from text.answer_parsing import extract_final_answer
from metrics.gsm8k_metrics import compute_gsm8k_metrics, exact_match

@dataclass
class StudentModelConfig:
    model_name: str = "google/flan-t5-base"

    # Model hyperparameters
    max_source_length: int = 512
    max_target_length: int = 256
    
    # Prompt templates (customize as needed)
    input_prefix: str = "Solve the problem:\n"

    # For answer-only training, we format target as: "Answer: <num>"
    answer_prefix: str = "Final answer: "

    # For CoT distillation, we format target as:
    reasoning_prefix: str = "Reasoning:\n"

class StudentModel:
    def __init__(self, config: Optional[StudentModelConfig] = None, device: Optional[str] = None):
        self.config = config or StudentModelConfig()
        # Load the tokenizer
        # the objective of the tokernizer is to convert text into tokens that the model can understand
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load the model
        # the model is a sequence-to-sequence model that generates text based on input sequences
        # utilize a pre-trained model from HuggingFace
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

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

            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config.max_source_length,
                truncation=True,
            )

            # Tokenize targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.config.max_target_length,
                    truncation=True,
                )

            # Replace pad token id's in labels by -100 to ignore in loss computation
            label_ids = labels["input_ids"]
            pad_id = self.tokenizer.pad_token_id
            label_ids = [[(t if t != pad_id else -100) for t in seq] for seq in label_ids]
            
            # Add labels to model inputs
            model_inputs["labels"] = label_ids

            # Return model inputs
            return model_inputs

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
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 8,
        gradient_checkpointing: bool = True,
        num_train_epochs: float = 1.0,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        label_smoothing_factor: float = 0.1,
        logging_steps: int = 50,
        eval_steps: int = 200,
        save_steps: int = 200,
        predict_with_generate: bool = True,
        fp16: bool = True,
        bf16: bool = False,
        seed: int = 42,
    ) -> Seq2SeqTrainer:
        # Train the model using HuggingFace's Seq2SeqTrainer
        # Set Seq2SeqTrainingArguments + DataCollatorForSeq2Seq + Seq2SeqTrainer and then launch training by trainer.train()

        # Prepare output directory
        output_dir = str(output_dir)

        # Training arguments
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir, # where to save model checkpoints and logs
            learning_rate=learning_rate, # learning rate for optimizer
            per_device_train_batch_size=per_device_train_batch_size, # batch size for training
            per_device_eval_batch_size=per_device_eval_batch_size, # batch size for evaluation
            num_train_epochs=num_train_epochs, # number of training epochs
            weight_decay=weight_decay, # weight decay for optimizer
            warmup_ratio=warmup_ratio, # warmup ratio for learning rate scheduler
            logging_steps=logging_steps, # log training info every N steps
            eval_strategy="steps" if eval_dataset is not None else "no", # evaluate every N steps if eval dataset is provided
            eval_steps=eval_steps if eval_dataset is not None else None, # evaluation frequency
            
            gradient_checkpointing=gradient_checkpointing, # enable gradient checkpointing to save memory
            
            # DISK CONTROL
            save_strategy="steps", # save model checkpoints every N steps
            save_steps=save_steps, # checkpoint saving frequency
            save_total_limit=1, #  limit total number of checkpoints to save
            load_best_model_at_end=True if eval_dataset is not None else False, # load best model at end of training
            
            seed=seed, # random seed for reproducibility
            report_to=[],  # keep notebooks clean by default
            metric_for_best_model="exact_match",
            greater_is_better=True,
            label_smoothing_factor=label_smoothing_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            

            # VRAM CONTROL
            fp16=fp16 and torch.cuda.is_available(), # enable fp16 if supported by GPU
            bf16=bf16 and torch.cuda.is_available(),  # enable bf16 if supported by GPU
            predict_with_generate=predict_with_generate, # enable generation during evaluation
        )

        # Data collator for seq2seq
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        # Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset, # training dataset
            eval_dataset=eval_dataset, # evaluation dataset
            data_collator=data_collator, # data collator for batching
            tokenizer=self.tokenizer, # tokenizer for decoding during evaluation
            compute_metrics=partial(compute_gsm8k_metrics, tokenizer=self.tokenizer)
        )

        # Start training
        trainer.train()

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

        # Decode outputs
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


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
        pred = [extract_final_answer(g) for g in generations]

        # Compute exact match
        em = sum(int(exact_match(pred, ga)) for pred, ga in zip(pred, gold)) / max(1, len(gold))
        
        return {
            "n": len(raw_examples),
            "exact_match": em,
            "pred_examples": list(zip(questions[:5], generations[:5], pred[:5], gold[:5]))
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
