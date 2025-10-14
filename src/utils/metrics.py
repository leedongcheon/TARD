# src/utils/metrics.py
import re

def extract_ans_lines(text: str):
    lines = [ln.strip() for ln in text.splitlines()]
    outs = []
    for ln in lines:
        if ln.lower().startswith("ans:"):
            outs.append(ln[len("ans:"):].strip())
    if not outs and text.strip():
        outs = [text.strip()]
    return outs

def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
