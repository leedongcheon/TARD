import torch
import torch.nn.functional as F

def focal_loss_with_pos_weight(logits, targets, alpha=0.75, gamma=2.0):
    if targets.dim() == logits.dim() - 1:
        targets = targets.unsqueeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, targets.expand_as(logits), reduction='none')
    pt = torch.exp(-bce)
    alpha_t = torch.where(targets.expand_as(logits) == 1, alpha, 1 - alpha)
    focal = alpha_t * (1 - pt) ** gamma * bce
    return focal.mean()

def ranking_loss_vectorized(logits, targets, margin=2.0):
    if logits.dim() == 2:
        logits = logits.max(dim=1)[0]  
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return logits.sum() * 0
    pos_logits = logits[pos_mask] 
    neg_logits = logits[neg_mask]  
    diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)  
    loss = F.relu(margin - diff).mean()
    return loss

def _norm_over_intents(x, mode="max", temperature=1.0, use_softmax=False):
    if mode == "max":
        return x.max(dim=1)[0]
    if use_softmax:
        w = F.softmax(x / max(1e-6, temperature), dim=1) 
        return (w * x).sum(dim=1)
    else:
        p = torch.sigmoid(x)  
        union = 1.0 - torch.prod(1.0 - p, dim=1)  
        eps = 1e-6
        union = torch.clamp(union, eps, 1 - eps)
        return torch.log(union) - torch.log(1.0 - union)

def topk_decorrelation_loss(
    logits,
    k=100,
    eps=1e-6,
    *,
    mode="max",               
    temperature=1.0,            
    use_softmax=False,
    positive_mask=None          
):

    if logits.numel() == 0 or logits.size(1) <= 1:
        return logits.sum() * 0

    T, I = logits.shape
    k = min(k, T)

    if k < 2:
        return logits.sum() * 0

    if positive_mask is not None:
        idx_all = torch.arange(T, device=logits.device)
        idx = idx_all[positive_mask.bool()]
        if idx.numel() >= 1:
            logits_sel = logits[idx]
        else:
            logits_sel = logits
    else:
        logits_sel = logits

    reduced_per_triple = _norm_over_intents(logits_sel, mode=mode, temperature=temperature, use_softmax=use_softmax)
    
    num_available = reduced_per_triple.size(0)
    if num_available < 2:
        return logits.sum() * 0
    
    k_actual = min(k, num_available)
    topk_idx_local = torch.topk(reduced_per_triple, k=k_actual, largest=True)[1]
    X = logits_sel[topk_idx_local]  

    if X.size(0) < 2:
        return logits.sum() * 0

    X = torch.sigmoid(X)  
    X = X - X.mean(dim=0, keepdim=True)
    
    std = X.std(dim=0, keepdim=True, unbiased=False) + eps
    X = X / std

    C = (X.t() @ X) / X.size(0)  
    mask = ~torch.eye(C.size(0), device=C.device, dtype=torch.bool)
    return (C[mask] ** 2).mean()

def pattern_diversity_loss(
    scores,
    k=100,
    temperature=0.1,
    positive_mask=None,
    cap=None,
    auto_transpose=True
):

    if scores.dim() != 2:
        return scores.sum() * 0
    
    if auto_transpose and scores.size(0) > scores.size(1):
        scores = scores.T  
    
    I, T = scores.shape
    if I <= 1 or T == 0:
        return scores.sum() * 0

    k = min(k, T)
    topk_indices = torch.topk(scores, k, dim=-1).indices 
    union_indices = torch.unique(topk_indices.flatten())
    
    if union_indices.numel() == 0:
        return scores.sum() * 0

    if cap is not None and union_indices.numel() > cap:
        max_scores = scores[:, union_indices].max(dim=0)[0]
        top_cap = torch.topk(max_scores, k=cap, largest=True).indices
        union_indices = union_indices[top_cap]

    scores_subset = scores[:, union_indices]  

    if positive_mask is not None:
        pos_mask_subset = positive_mask[union_indices]
        scores_subset = scores_subset.clone()
        scores_subset[:, pos_mask_subset] = -100.0
    if scores_subset.size(1) < 2:
        return scores.sum() * 0

    P = F.softmax(scores_subset / max(1e-6, temperature), dim=-1) 

    P = P - P.mean(dim=1, keepdim=True)
    
    std = P.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    P = P / std
    
    C = (P @ P.t()) / P.size(1)  
    mask = ~torch.eye(I, device=C.device, dtype=torch.bool)
    return (C[mask].abs()).mean()