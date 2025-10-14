import json, re, os, string
from copy import deepcopy
from tqdm import tqdm
import argparse

try:
    import torch
except Exception:
    torch = None 

def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())

def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)

def remove_duplicates(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _to_text(x):
    if isinstance(x, (list, tuple)):
        return "\n".join(map(str, x))
    return "" if x is None else str(x)

def get_pred(prediction, split=None):
    prediction = _to_text(prediction)  

    if split is not None:
        return [p for p in prediction.split(split) if p]

    res = [p for p in prediction.split("\n") if 'ans:' in p and 'none' not in p.lower()]
    if len(res) >= 1:
        res = [p for p in res if "ans: not available" not in p.lower()
                          and "ans: no information available" not in p.lower()]
    return remove_duplicates(res)


def eval_recall_corrected(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    matched = 0.
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / len(answer), matched, len(answer)

def eval_precision_corrected(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    num_pred = len(prediction)
    if num_pred == 0:
        return 0, 0, 0
    matched = 0.
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / num_pred, matched, num_pred

def eval_f1_corrected(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def eval_hit1_corrected(prediction, answer, double_check):
    if len(prediction) == 0:
        return 0
    for a in answer:
        if match(prediction[0], a):
            return 1
        elif double_check:
            if match(a, prediction[0].split('ans:')[-1].strip()):
                return 1
    return 0


def eval_results_corrected_compat(
    predict_file: str,
    cal_f1: bool = True,
    subset: bool = False,
    split: str | None = None,
    eval_hops: int = -1,
    dataset_name: str | None = None,   
    scored_triples_path: str | None = None,
):
    """
    Evaluate predictions with corrected metrics.
    
    Note: Hallucination scoring (hal_score) is disabled as triplets_dict 
    functionality is not implemented. Returns 0 for hal_score.
    """
    samples_to_eval = None
    
    if (dataset_name is None) and ("webqsp" in predict_file or "cwq" in predict_file):
        dataset_name = "webqsp" if "webqsp" in predict_file else "cwq"

    if dataset_name is not None and torch is not None:
        if scored_triples_path is None:
            if dataset_name == "webqsp":
                scored_triples_path = "./scored_triples/webqsp_240912_unidir_test.pth"
            else:
                scored_triples_path = "./scored_triples/cwq_240907_unidir_test.pth"
        try:
            samples_to_eval = torch.load(scored_triples_path, weights_only=False)
        except Exception:
            samples_to_eval = None

    # Aggregation buckets
    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    total_pred = 0
    total_answer = 0
    total_match = 0
    hal_score_list = []

    total_cnt = 0
    no_ans_cnt = 0
    stats = {
        'g_no_ans': 0, 'g_c': 0, 'g_w': 0, 
        'b_no_ans': 0, 'b_in_graph': 0, 'b_out_graph_c': 0, 'b_out_graph_w': 0,
        'total_ans': 0, 'total_g_samples': 0, 'total_b_samples': 0, 'total_samples': 0,
        'total_g_ans': 0, 'total_b_ans': 0,
        'g_c_out_graph': 0, "g_w_out_graph": 0, 'g_c_in_graph': 0, 'g_w_in_graph': 0
    }

    with open(predict_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Eval(corrected)"):
            try:
                data = json.loads(line)
            except Exception:
                print("[WARN] skip unparsable line:", line[:120])
                continue

            id_ = data.get('id')
            question = data.get('question', '')
            prediction_text = data.get('prediction', '')
            prediction_text = _to_text(prediction_text)
            answer = sorted(remove_duplicates(data.get('ground_truth', [])), key=len, reverse=True)

            if 'when' in question.lower() or 'what year' in question.lower():
                for idx in range(len(answer)):
                    if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                        answer[idx] = answer[idx].split('-')[0]
            
            double_check = any(k in question.lower() for k in [
                'when', 'what year', 'which year', 'where', 'sport', 
                "what countr", "language", 'nba finals', 'world series'
            ])

            if samples_to_eval is not None and id_ in samples_to_eval:
                if eval_hops > 0:
                    if eval_hops == 3:
                        if samples_to_eval[id_]['max_path_length'] is None or samples_to_eval[id_]['max_path_length'] < 3:
                            continue
                    elif samples_to_eval[id_]['max_path_length'] != eval_hops:
                        continue
                if subset and not samples_to_eval[id_]['a_entity_in_graph']:
                    continue

            if cal_f1:
                pred_list = get_pred(prediction_text, split)
                total_cnt += 1

                no_ans_flag = False
                if split == '\n':
                    if len(pred_list) == 0:
                        no_ans_cnt += 1
                        no_ans_flag = True
                else:
                    if len(pred_list) == 0 or 'ans:' not in prediction_text.lower() or \
                       "ans: not available" in prediction_text.lower() or \
                       "ans: no information available" in prediction_text.lower():
                        no_ans_cnt += 1
                        no_ans_flag = True

                precision_score, matched_1, num_pred = eval_precision_corrected(pred_list, answer, double_check)
                recall_score, matched_2, num_answer = eval_recall_corrected(pred_list, answer, double_check)
                f1_score = eval_f1_corrected(precision_score, recall_score)
                hit1 = eval_hit1_corrected(pred_list, answer, double_check)

                # Hallucination score disabled (triplets_dict not implemented)
                hal_score = 0

                assert matched_1 == matched_2
                total_pred += num_pred
                total_answer += num_answer
                total_match += matched_1

                hal_score_list.append(hal_score)
                f1_list.append(f1_score)
                precision_list.append(precision_score)
                recall_list.append(recall_score)
                hit_list.append(hit1)
                acc_list.append(recall_score)

    if len(hit_list) == 0:
        return [0]*13 + [None]

    avg_hit = sum(hit_list) * 100 / len(hit_list)
    avg_f1 = sum(f1_list) * 100 / len(f1_list)
    avg_precision = sum(precision_list) * 100 / len(precision_list)
    avg_recall = sum(recall_list) * 100 / len(recall_list)
    avg_hal_score = sum(hal_score_list) / len(hal_score_list)
    avg_hal_score = (avg_hal_score + 1.5) / (1 + 1.5) * 100

    num_exact_match = (sum(1 for x in f1_list if x == 1) / len(f1_list)) * 100
    num_totally_wrong = (sum(1 for x in recall_list if x == 0) / len(recall_list)) * 100

    micro_precision = total_match / total_pred if total_pred else 0.0
    micro_recall = total_match / total_answer if total_answer else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    return (avg_hit, avg_f1, avg_precision, avg_recall,
            num_exact_match, num_totally_wrong,
            micro_f1, micro_precision, micro_recall,
            total_cnt, no_ans_cnt, (no_ans_cnt/total_cnt if total_cnt else 0.0),
            avg_hal_score, stats)


def eval_acc_any(prediction_str, answer):
    matched = 0.
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    return matched / len(answer) if answer else 0.0

def eval_hit_any_style(prediction_str, answer, double_check):
    for a in answer:
        if "ans:" in prediction_str:
            all_pred = get_pred(prediction_str)
            for each_pred in all_pred:
                if match(each_pred, a):
                    return 1
                elif double_check and match(a, each_pred.split('ans:')[-1].strip()):
                    return 1
        else:
            if match(prediction_str, a):
                return 1
            elif double_check:
                for each_pred in prediction_str.split("\n"):
                    if match(a, each_pred):
                        return 1
    return 0

def f1_any_style(prediction_list, answer, double_check):
    if len(prediction_list) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = '\n'.join(prediction_list)
    all_pred = get_pred(prediction_str)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
        elif double_check:
            for each_pred in all_pred:
                if match(a, each_pred.split('ans:')[-1].strip()):
                    matched += 1
                    all_pred.remove(each_pred)
                    break
    precision = matched / len(prediction_list)
    recall = matched / len(answer) if answer else 0
    if precision + recall == 0:
        return 0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall

def eval_results_hit_any_compat(
    predict_file: str,
    cal_f1: bool = True,
    topk: int = -1,
    subset: bool = False,
    eval_hops: int = -1,
    dataset_name: str | None = None,
    scored_triples_path: str | None = None,
):
    acc_list, hit_list, f1_list, prec_list, rec_list = [], [], [], [], []

    samples_to_eval = None
    if (dataset_name is None) and ("webqsp" in predict_file or "cwq" in predict_file):
        dataset_name = "webqsp" if "webqsp" in predict_file else "cwq"
    
    if dataset_name is not None and torch is not None:
        if scored_triples_path is None:
            scored_triples_path = "./scored_triples/webqsp_240912_unidir_test.pth" if dataset_name=="webqsp" else "./scored_triples/cwq_240907_unidir_test.pth"
        try:
            samples_to_eval = torch.load(scored_triples_path, weights_only=False)
        except Exception:
            samples_to_eval = None

    with open(predict_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Eval(hit-any)"):
            try:
                data = json.loads(line)
            except Exception:
                continue
            
            id_ = data.get('id')
            q = data.get('question', '').lower()
            answer = sorted(data.get('ground_truth', []), key=len, reverse=True)
            pred_text = data.get('prediction', '')
            pred_text = _to_text(pred_text)
            
            if samples_to_eval is not None and id_ in samples_to_eval:
                if eval_hops > 0:
                    if eval_hops == 3:
                        if samples_to_eval[id_]['max_path_length'] is None or samples_to_eval[id_]['max_path_length'] < 3:
                            continue
                    elif samples_to_eval[id_]['max_path_length'] != eval_hops:
                        continue
                if subset and not samples_to_eval[id_]['a_entity_in_graph']:
                    continue

            double_check = any(k in q for k in [
                'when','what year','which year','where','sport',
                'what countr','language','nba finals','world series'
            ])

            if cal_f1:
                pred_list = pred_text.split("\n") if isinstance(pred_text, str) else pred_text
                f1_score, p, r = f1_any_style(pred_list, answer, double_check)
                f1_list.append(f1_score)
                prec_list.append(p)
                rec_list.append(r)
                prediction_str = '\n'.join(pred_list)
                acc = eval_acc_any(prediction_str, answer)
                hit = eval_hit_any_style(prediction_str, answer, double_check)
                acc_list.append(acc)
                hit_list.append(hit)
            else:
                prediction_str = pred_text if isinstance(pred_text, str) else '\n'.join(pred_text)
                acc_list.append(eval_acc_any(prediction_str, answer))
                hit_list.append(eval_hit_any_style(prediction_str, answer, double_check))

    if len(hit_list) == 0:
        return [0]*4

    avg_hit = sum(hit_list) * 100 / len(hit_list)
    avg_f1  = sum(f1_list) * 100 / len(f1_list)
    avg_p   = sum(prec_list) * 100 / len(prec_list)
    avg_r   = sum(rec_list) * 100 / len(rec_list)
    return avg_hit, avg_f1, avg_p, avg_r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate predictions.jsonl (corrected + hit-any)")
    ap.add_argument("--pred_file", required=True, help="Path to predictions.jsonl")
    args = ap.parse_args()

    # Corrected metrics
    (hit1, f1, prec, rec, em, tw, mi_f1, mi_p, mi_r,
     total_cnt, no_ans_cnt, no_ans_ratio, hal, stats) = eval_results_corrected_compat(
        predict_file=args.pred_file,
        cal_f1=True,
        subset=False,
        split=None,
        eval_hops=-1,
        dataset_name=None,   
        scored_triples_path=None,
    )
    print(f"[CORRECTED] Hit@1={hit1:.2f}, MacroF1={f1:.2f}, MacroP={prec:.2f}, MacroR={rec:.2f}, "
          f"EM={em:.2f}, TW={tw:.2f}, MicroF1={mi_f1*100:.2f}, MicroP={mi_p*100:.2f}, MicroR={mi_r*100:.2f}, "
          f"Total={total_cnt}, No-Ans={no_ans_cnt} ({no_ans_ratio*100:.2f}%), Hal={hal:.2f}")

    # Hit-any metrics
    hit_any, f1_any, p_any, r_any = eval_results_hit_any_compat(
        predict_file=args.pred_file,
        cal_f1=True,
        topk=-1,
        subset=False,
        eval_hops=-1,
        dataset_name=None,
        scored_triples_path=None,
    )
    print(f"[HIT-ANY] Hit={hit_any:.2f}, F1={f1_any:.2f}, P={p_any:.2f}, R={r_any:.2f}")