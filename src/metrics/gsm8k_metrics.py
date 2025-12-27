from src.text.answer_parsing import extract_final_answer
from src.text.numeric import normalize_number
from typing import Optional

def exact_match(pred_text: str, gold_answer: str) -> bool:
    pred_norm = normalize_number(pred_text)
    gold_norm = normalize_number(gold_answer)

    if not pred_norm or not gold_norm:
        return False
    
    try:
        return float(pred_norm) == float(gold_norm)
    except:
        return pred_norm == gold_norm

def compute_gsm8k_metrics(eval_pred, tokenizer) -> dict:
    preds, labels = eval_pred
    # preds può essere tuple
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # labels: rimpiazza -100 con pad_token_id per decodifica
    labels = [[(t if t != -100 else tokenizer.pad_token_id) for t in seq] for seq in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_ans = [extract_final_answer(x) for x in decoded_preds]
    gold_ans = [extract_final_answer(x) for x in decoded_labels]

    em = sum(p.strip() == g.strip() for p, g in zip(pred_ans, gold_ans)) / max(1, len(gold_ans))
    return {"exact_match": em}
