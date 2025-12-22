# Knowledge Distillation for Step-by-Step Reasoning on GSM8K

## Project Overview

This project investigates **knowledge distillation** as a technique to transfer **step-by-step reasoning capabilities** from a large language model (teacher) to a significantly smaller model (student), using the **GSM8K** dataset as a controlled reasoning benchmark.

The core objective is to understand to what extent a compact model can internalize the **reasoning behavior** of a large language model, rather than merely reproducing correct answers. The project emphasizes the trade-offs between **model compactness, reasoning quality, generalization, and computational efficiency**.

---

## Objectives

- Distill reasoning knowledge from a large LLM into a small transformer model.
- Evaluate the effectiveness of **Chain-of-Thought (CoT) distillation** compared to standard supervised training.
- Measure performance degradation, efficiency gains, and reasoning faithfulness.
- Provide a reproducible experimental pipeline suitable for academic research.

---

## Dataset: GSM8K

GSM8K (Grade School Math 8K) is a dataset of natural language math word problems that require multi-step arithmetic reasoning. Each sample consists of:

- a textual problem description;
- a single correct numerical answer.

The dataset is particularly suitable for reasoning research because the correct output depends on a structured sequence of intermediate steps rather than surface-level pattern matching.

---

## 3. Teacher Model and Reasoning Supervision

### 3.1 Role of the Teacher Model

The **teacher model** is a large-scale language model (e.g., GPT-4) that exhibits strong reasoning capabilities. Its role extends beyond label generation: it serves as a **proxy expert**, providing rich supervisory signals that encode how a problem is solved.

From a theoretical standpoint, the teacher represents a highly expressive function that maps an input problem to:

- a sequence of intermediate reasoning steps;
- a final correct answer.

This additional structure transforms the learning problem from a simple input–output mapping into a **process imitation task**, which is central to modern reasoning-oriented distillation.

---

### 3.2 Chain-of-Thought Data Generation

For each GSM8K problem, the teacher model is prompted to generate:

1. an explicit step-by-step reasoning process;
2. the final numerical answer.

The generated dataset follows a structured format:

Input: Natural language math problem

Target: Step-by-step reasoning + final answer

This augmented dataset becomes the foundation for distillation. Unlike traditional supervised learning, where only the final answer is available, this setup provides **intermediate cognitive traces** that guide the student model during training.

Special care must be taken to:

- enforce consistent output formatting;
- avoid hallucinated or incorrect reasoning steps;
- ensure that the final answer matches the original GSM8K label.

---

### 3.3 Theoretical Motivation for Chain-of-Thought

Chain-of-Thought supervision provides several theoretical advantages:

- **Entropy reduction**: By decomposing a complex reasoning task into simpler steps, the conditional distribution learned by the student becomes easier to approximate.
- **Inductive bias injection**: The student is encouraged to follow structured reasoning paths rather than shortcut heuristics.
- **Privileged information**: CoT acts as additional training information unavailable at inference time, improving learning efficiency.

From a knowledge distillation perspective, CoT enables the transfer of **procedural knowledge**, not just declarative outcomes.

---

## 4. Student Model

The student model is a compact transformer architecture (e.g., TinyT5 or FLAN-T5-small) trained using supervised fine-tuning.

An encoder–decoder architecture is preferred because:

- the task is naturally formulated as text-to-text generation;
- reasoning steps and answers can be generated sequentially;
- it supports teacher forcing during training.

The student model represents a constrained hypothesis space, making it ideal for studying the limits of knowledge compression.

---

## 5. Knowledge Distillation Strategies

### 5.1 Standard Supervised Baseline

As a baseline, the student is trained using only:

- the GSM8K input problem;
- the ground-truth final answer.

This setup mirrors traditional fine-tuning and serves as a lower bound for reasoning performance. It allows us to isolate the benefits introduced by distillation-based supervision.

---

### 5.2 Chain-of-Thought Distillation

In the primary distillation setup, the student is trained to generate the **full reasoning trace produced by the teacher**, followed by the final answer.

Training uses a standard sequence-to-sequence language modeling loss. During optimization, the student learns to model the teacher’s reasoning distribution rather than the task distribution alone.

This approach encourages:

- internalization of reasoning structure;
- improved robustness to compositional complexity;
- better in-domain generalization.

---

### 5.3 Comparative Experimental Design

Multiple training configurations are considered:

1. Supervised training with final answers only.
2. Distillation using Chain-of-Thought supervision.
3. (Optional) Mixed setups combining both objectives.

This design enables controlled comparisons and ablation studies, clarifying whether improvements stem from:

- additional supervision;
- reasoning structure;
- or increased target sequence length.

---

## 6. Training Procedure

### 6.1 Data Preparation and Splitting

The dataset is split into training, validation, and test sets. Importantly:

- Chain-of-Thought annotations are used **only during training**;
- Evaluation is performed **only on the final answer**, ensuring fair comparison.

Input and target sequences are tokenized consistently across experiments. Padding and truncation strategies must preserve reasoning coherence.

Teacher forcing is used during training to stabilize convergence and reduce exposure bias.

---

## 7. Evaluation and Comparison

### 7.1 Quantitative Evaluation

The primary evaluation metric is **Exact Match Accuracy** on the final numerical answer. This ensures objective and reproducible assessment.

Additional metrics may include:

- arithmetic error rate;
- average reasoning length;
- step-level correctness (optional).

Evaluation must be performed on:

- the teacher model (upper bound);
- the baseline student;
- the distilled student.

---

### 7.2 Computational Efficiency Metrics

To evaluate the practical benefits of distillation, the following are measured:

- number of model parameters;
- inference latency per sample;
- memory consumption during inference.

These metrics allow direct comparison between reasoning quality and efficiency gains.

---

### 7.3 Qualitative Analysis

Beyond numerical scores, qualitative inspection is essential:

- comparison of reasoning traces between teacher and student;
- identification of systematic reasoning failures;
- analysis of error propagation across steps.

This analysis reveals _how_ and _why_ the student fails, not just _how often_.

---

### 7.4 Trade-off Analysis

The final comparison highlights the trade-offs between:

- reasoning fidelity and model size;
- performance and generalization;
- accuracy and computational cost.

The goal is to determine whether a distilled student can act as a **domain-specific reasoning expert** under resource constraints.

---

## Conclusion

This project provides a structured framework for studying reasoning-aware knowledge distillation. By leveraging GSM8K and Chain-of-Thought supervision, it explores the boundaries of compressing reasoning behavior into compact neural models, offering insights into both theoretical and practical aspects of modern language model distillation.

---

## References

- Hinton et al., _Distilling the Knowledge in a Neural Network_, 2015
- Wei et al., _Chain-of-Thought Prompting Elicits Reasoning_, 2022
- Jiao et al., _TinyBERT_, 2020
