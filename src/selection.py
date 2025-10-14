# src/selection.py
import torch

def select_fixed_total_by_intents(logits_TI: torch.Tensor, k_each: int, total_N: int):
    T, I = logits_TI.shape
    total_N = int(min(max(1, total_N), T))
    k_eff = min(int(k_each), T)

    _, idx = torch.topk(logits_TI, k_eff, dim=0)
    intent_masks = torch.zeros_like(logits_TI, dtype=logits_TI.dtype)
    intent_masks.scatter_(0, idx, 1.0)

    gate = intent_masks.sum(dim=1)
    union_idx = torch.nonzero(gate > 0, as_tuple=True)[0]
    global_scores = logits_TI.max(dim=1).values

    if union_idx.numel() >= total_N:
        scores_u = global_scores[union_idx]
        topN_in_union = torch.topk(scores_u, k=total_N, largest=True).indices
        sel_indices = union_idx[topN_in_union]
    else:
        selected = torch.zeros(T, dtype=torch.bool, device=logits_TI.device)
        selected[union_idx] = True
        order = torch.argsort(global_scores, descending=True)
        need = total_N - union_idx.numel()
        add_candidates = order[~selected[order]][:need]
        sel_indices = torch.cat([union_idx, add_candidates], dim=0)

    return intent_masks, sel_indices


def gumbel_topk_batch(logits_TI, k, tau_gumbel=0.5):
    T, I = logits_TI.shape
    k_eff = min(int(k), int(T))
    if k_eff <= 0:
        return torch.zeros_like(logits_TI)
    u = torch.rand_like(logits_TI)
    g = -torch.log(-torch.log(u.clamp_min(1e-8)))
    scores = torch.sigmoid((logits_TI + g) / max(tau_gumbel, 1e-8))
    _, topk_idx = torch.topk(scores, k_eff, dim=0)
    hard = torch.zeros_like(scores)
    hard.scatter_(0, topk_idx, 1.0)
    sorted_scores, _ = torch.sort(scores, dim=0, descending=True)
    th = sorted_scores[k_eff-1, :].unsqueeze(0)
    soft = torch.sigmoid((scores - th) / (max(tau_gumbel, 1e-8) * 0.1))
    return (hard + soft - soft.detach()).to(dtype=logits_TI.dtype)


def per_intent_topk_no_noise(logits_TI, k_each):
    T, I = logits_TI.shape
    k_eff = min(int(k_each), int(T))
    _, idx = torch.topk(logits_TI, k_eff, dim=0)
    mask = torch.zeros_like(logits_TI)
    mask.scatter_(0, idx, 1.0)
    return mask


def get_per_intent_quota(num_intents, per_intent_total):
    I = max(1, int(num_intents))
    return max(1, int(per_intent_total) // I)
