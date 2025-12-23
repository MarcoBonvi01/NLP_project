import re

def extract_after_delim(text: str) -> str:
    # prende l'ultima occorrenza di ####
    parts = text.split("####")
    if len(parts) >= 2:
        return parts[-1].strip()
    # fallback: ultima cifra in stringa
    m = re.findall(r"-?\d+\.?\d*", text)
    return m[-1] if m else ""

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    # preds può essere tuple
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # labels: rimpiazza -100 con pad_token_id per decodifica
    labels = [[(t if t != -100 else tokenizer.pad_token_id) for t in seq] for seq in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_ans = [extract_after_delim(x) for x in decoded_preds]
    gold_ans = [extract_after_delim(x) for x in decoded_labels]

    em = sum(p.strip() == g.strip() for p, g in zip(pred_ans, gold_ans)) / max(1, len(gold_ans))
    return {"exact_match": em}
