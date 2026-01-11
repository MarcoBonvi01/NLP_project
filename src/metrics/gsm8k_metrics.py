from text.answer_parsing import extract_final_answer
from text.numeric import normalize_number
import numpy as np
from typing import Any, Dict

def exact_match(pred_text: str, gold_answer: str) -> bool:
    pred_norm = normalize_number(pred_text)
    gold_norm = normalize_number(gold_answer)

    if not pred_norm or not gold_norm:
        return False
    
    try:
        return float(pred_norm) == float(gold_norm)
    except:
        return pred_norm == gold_norm

def compute_gsm8k_metrics(eval_pred, tokenizer) -> Dict[str, Any]:
    """
    Compute exact match metrics for GSM8K.
    Fixed to handle invalid token IDs during generation.
    """
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Se preds è una tupla (loss, logits), prendi solo i logits
    if isinstance(preds, tuple):
        preds = preds[0]
    
    
    vocab_size = tokenizer.vocab_size
    
    # Converti a numpy se necessario
    if hasattr(preds, 'cpu'):
        preds = preds.cpu().numpy()
    
    # Clamp tutti i token IDs al range valido [0, vocab_size)
    preds = np.clip(preds, 0, vocab_size - 1).astype(np.int32)
    
    # ===== FIX: Gestisci anche token IDs strani nei labels =====
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    # Decodifica predictions con gestione errori
    try:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    except Exception as e:
        print(f"Warning: Error decoding predictions: {e}")
        # Fallback: decodifica uno alla volta saltando errori
        decoded_preds = []
        for pred_seq in preds:
            try:
                decoded = tokenizer.decode(pred_seq, skip_special_tokens=True)
                decoded_preds.append(decoded)
            except:
                decoded_preds.append("")  # Stringa vuota se fallisce
    
    # Labels: rimpiazza -100 con pad_token_id per decodifica
    labels_clean = []
    for seq in labels:
        clean_seq = [t if t != -100 else tokenizer.pad_token_id for t in seq]
        # Clamp anche i labels
        clean_seq = np.clip(clean_seq, 0, vocab_size - 1).astype(np.int32)
        labels_clean.append(clean_seq)
    
    try:
        decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
    except Exception as e:
        print(f"Warning: Error decoding labels: {e}")
        decoded_labels = [""] * len(labels_clean)
    
    # Estrai final answers
    pred_answers = [extract_final_answer(p) for p in decoded_preds]
    gold_answers = [extract_final_answer(g) for g in decoded_labels]
    
    # Calcola exact match
    matches = [exact_match(p, g) for p, g in zip(pred_answers, gold_answers)]
    em = sum(matches) / max(1, len(matches))
    
    return {
        "exact_match": em,
    }
