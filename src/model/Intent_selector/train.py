import os, csv, math, json, pickle, argparse, random, multiprocessing as mp, hashlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import build_unified_topic_datasets
from S2WTM import S2WTM_Flex


# ---------------- utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _as_str(x): return str(x) if not isinstance(x, str) else x

def safe_torch_load(path, map_location="cpu", prefer_weights_only=True):
    try:
        if prefer_weights_only:
            return torch.load(path, map_location=map_location, weights_only=True)
        return torch.load(path, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _load_any(p):
    try:
        return safe_torch_load(p, map_location="cpu", prefer_weights_only=False)
    except Exception:
        with open(p, "rb") as f:
            return pickle.load(f)

def load_id2q_from_pkl(pkl_path: str):
    data = _load_any(pkl_path)
    out = {}
    def push(k, v):
        if k is None or v is None: return
        k = _as_str(k)
        if k not in out: out[k] = v
    if isinstance(data, dict):
        src = data.get("data", data)
        if isinstance(src, dict):
            for k, d in src.items():
                if isinstance(d, dict):
                    q = d.get("q_text") or d.get("question") or d.get("text") or d.get("q"); push(k, q)
        else:
            for k, d in data.items():
                if isinstance(d, dict):
                    q = d.get("q_text") or d.get("question") or d.get("text") or d.get("q"); push(k, q)
                elif isinstance(d, str):
                    push(k, d)
    elif isinstance(data, (list, tuple)):
        for it in data:
            if isinstance(it, dict):
                k = it.get("id") or it.get("qid") or it.get("uid")
                q = it.get("q_text") or it.get("question") or it.get("text") or it.get("q")
                push(k, q)
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                push(it[0], it[1])
    if not out: raise RuntimeError(f"Could not find question texts in PKL file: {pkl_path}")
    return out

def load_id2emb_pack(pth_path: str):
    obj = safe_torch_load(pth_path, map_location="cpu", prefer_weights_only=False)
    if not isinstance(obj, dict): raise RuntimeError(f"[ERR] Unexpected format: {type(obj)}")
    return {_as_str(k): v for k, v in obj.items()}

def stack_texts_embs(id2q, id2pack):
    ids = sorted(set(id2q.keys()) & set(id2pack.keys()))
    assert ids, "No intersection of IDs found. Please check pkl/pth files."
    texts = [id2q[i] for i in ids]
    embs  = torch.stack([id2pack[i]["q_emb"].float().squeeze() for i in ids], dim=0)
    return texts, embs, ids

def _sha1_list(str_list):
    h = hashlib.sha1(); h.update("\n".join(str_list).encode("utf-8")); return h.hexdigest()

def apply_simple_stop(texts, stopset=None, min_token_len=2):
    try:
        from gensim.utils import simple_preprocess
    except Exception:
        import re
        def simple_preprocess(s, deacc=True):
            s = s.lower()
            if deacc: s = re.sub(r"[^a-z0-9\s_]+", " ", s)
            return [w for w in s.split() if w]
    if stopset is None:
        stopset = {
            "the","a","an","of","and","to","in","on","for","with","from","by",
            "is","are","was","were","be","been","being",
            "this","that","these","those",
        }
    out = []
    for t in texts:
        toks = simple_preprocess(t, deacc=True)
        toks = [w for w in toks if (w not in stopset and len(w) >= min_token_len)]
        out.append(" ".join(toks))
    return out

@torch.no_grad()
def build_G_from_dictionary(
    dictionary,
    device,
    model_name="Alibaba-NLP/gte-large-en-v1.5",
    cache_path=None,
    batch_size=256,
    max_length=8,
    pooling="cls",
    replace_underscore=True,
):

    id2token = {i: t for t, i in dictionary.token2id.items()}
    vocab_raw = [id2token[i] for i in range(len(id2token))]
    vocab_enc = [t.replace("_", " ") if replace_underscore else t for t in vocab_raw]

    meta_now = {
        "vocab_size": len(vocab_raw),
        "vocab_raw_sha1": _sha1_list(vocab_raw),
        "vocab_enc_sha1": _sha1_list(vocab_enc),
        "model_name": model_name,
        "max_length": max_length,
        "pooling": pooling,
        "normalize": True,
    }

    if cache_path and os.path.exists(cache_path):
        obj = safe_torch_load(cache_path, map_location=device, prefer_weights_only=False)
        if isinstance(obj, dict) and "G" in obj and "meta" in obj:
            Gc, meta = obj["G"], obj["meta"]
            same = (
                meta.get("vocab_size")==meta_now["vocab_size"] and
                meta.get("vocab_raw_sha1")==meta_now["vocab_raw_sha1"] and
                meta.get("vocab_enc_sha1")==meta_now["vocab_enc_sha1"] and
                meta.get("model_name")==meta_now["model_name"] and
                meta.get("max_length")==meta_now["max_length"] and
                meta.get("pooling")==meta_now["pooling"] and
                bool(meta.get("normalize", True)) is True
            )
            print(f"[G-cache] exact match = {same} (pooling={pooling})")
            if same:
                return F.normalize(Gc.to(device), p=2, dim=1)
            else:
                print("[INFO] G-cache meta mismatch → rebuild")
        else:
            print("[INFO] Unknown G-cache format → rebuild")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    enc = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    embs = []
    with torch.inference_mode():
        for s in range(0, len(vocab_enc), batch_size):
            batch = vocab_enc[s:s+batch_size]
            bt = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = enc(**bt)
            if pooling == "mean":
                last = out.last_hidden_state
                mask = bt["attention_mask"].unsqueeze(-1)
                vec = (last * mask).sum(1) / mask.sum(1).clamp_min(1)
            elif pooling == "pooler" and hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:  # "cls"
                vec = out.last_hidden_state[:, 0]
            vec = F.normalize(vec, p=2, dim=1)
            embs.append(vec.detach())
    G = torch.cat(embs, dim=0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"G": G.cpu(), "meta": meta_now}, cache_path)

    return F.normalize(G.to(device), p=2, dim=1)

def compute_idf_vec(dictionary, gamma=0.7):

    N = max(1, dictionary.num_docs)
    idf = torch.zeros(len(dictionary), dtype=torch.float32)
    for token_id, df in dictionary.dfs.items():
        val = math.log((N + 1.0) / (df + 1.0))
        idf[token_id] = max(0.0, val)
    if gamma is not None and gamma != 1.0:
        idf = idf.pow(float(gamma))
    idf = torch.clamp(idf, min=1e-6)
    return idf


class EarlyStopper:
    def __init__(self, patience=10, min_delta=5e-4):
        self.patience=patience; self.min_delta=min_delta
        self.best=float("inf"); self.count=0; self.should_stop=False
        self.best_state=None
    def step(self, metric, model):
        if metric < (self.best - self.min_delta):
            self.best = metric; self.count = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.wae.state_dict().items()}
        else:
            self.count += 1
            if self.count >= self.patience: self.should_stop=True


@torch.no_grad()
def eval_mean_total(val_loader, model, G, device, args, tau_rec, lam, idf_vec):
    model.wae.eval()
    totals = []
    for batch in val_loader:
        _, bows, embs_b = batch
        bows = bows.to(device); embs_b = embs_b.to(device)

        x_reconst, z = model.wae(embs_b)

        t = bows / (bows.sum(dim=1, keepdim=True) + 1e-8)
        qn  = F.normalize(embs_b, p=2, dim=1)
        sim = qn @ G.t()
        t_hat = F.softmax(sim / args.tau_g, dim=1)
        t = (1 - args.alpha_pbow) * t + args.alpha_pbow * t_hat

        w = idf_vec.to(device).unsqueeze(0)
        t = t * w
        t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-8)

        k = min(args.topk_mask_k, t.size(1)-1)
        if k > 0:
            _, idx = t.topk(k, dim=1)
            mask = torch.zeros_like(t).scatter(1, idx, 1.0)
            eps = 1e-6
            t = (t * mask) + eps
            t = t / t.sum(dim=1, keepdim=True)

        logits = x_reconst / tau_rec
        logp   = F.log_softmax(logits, dim=1)
        rec    = -(t * logp).sum(dim=1).mean()

        theta_prior = model.wae.sample(batch_size=z.size(0)).to(device)
        ot = model.wae.sp_swd_loss(z, theta_prior, num_projections=args.num_proj, device=device, p=2)

        total = rec + lam * ot
        totals.append(float(total.item()))
    model.wae.train()
    return float(np.mean(totals)) if totals else float("inf")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--train_pkl", required=True, type=str, help="Train question pickle file path")
    ap.add_argument("--train_pth", required=True, type=str, help="Train embedding pth file path")
    ap.add_argument("--val_pkl", required=True, type=str, help="Validation question pickle file path")
    ap.add_argument("--val_pth", required=True, type=str, help="Validation embedding pth file path")
    ap.add_argument("--g_cache", default="artifacts/gte/G.pt", type=str)
    ap.add_argument("--out", default="artifacts/output", type=str)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=42)
    
    ap.add_argument("--n_topic", type=int, default=16)
    ap.add_argument("--num_epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--num_proj", type=int, default=6000)
    ap.add_argument("--lambda_ot", type=float, default=1.8)
    ap.add_argument("--decoder_type", type=str, default="mlp", choices=["basis", "mlp"])
    ap.add_argument("--dropout", type=float, default=0.1)
    
    ap.add_argument("--dec_lr_mul", type=float, default=8.0)
    ap.add_argument("--dec_lr_max", type=float, default=7e-3)
    
    ap.add_argument("--tau_w", type=float, default=1.3)
    ap.add_argument("--theta_temp", type=float, default=0.65)
    ap.add_argument("--tau_rec", type=float, default=0.6)
    
    ap.add_argument("--alpha_pbow", type=float, default=0.80)
    ap.add_argument("--tau_g", type=float, default=0.04)
    
    ap.add_argument("--topk_mask_k", type=int, default=32)
    
    ap.add_argument("--g_pooling", type=str, default="mean", choices=["cls", "mean", "pooler"])
    
    ap.add_argument("--eta_min", type=float, default=3e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=5e-4)
    
    ap.add_argument("--dict_no_below", type=int, default=1)
    ap.add_argument("--dict_no_above", type=float, default=0.88)
    
    ap.add_argument("--idf_gamma", type=float, default=1.2)
    
    args = ap.parse_args()
    
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    
    train_id2q = load_id2q_from_pkl(args.train_pkl)
    train_id2pack = load_id2emb_pack(args.train_pth)
    train_texts, train_embs, train_ids = stack_texts_embs(train_id2q, train_id2pack)
    
    val_id2q = load_id2q_from_pkl(args.val_pkl)
    val_id2pack = load_id2emb_pack(args.val_pth)
    val_texts, val_embs, val_ids = stack_texts_embs(val_id2q, val_id2pack)
    
    train_texts_for_dict = apply_simple_stop(train_texts)
    val_texts_for_dict = apply_simple_stop(val_texts)
    
    train_ds, val_ds, dictionary = build_unified_topic_datasets(
        train_texts=train_texts_for_dict,
        test_texts=val_texts_for_dict,
        emb_train=train_embs,
        emb_test=val_embs,
        shared=True,
        dict_no_below=args.dict_no_below,
        dict_no_above=args.dict_no_above,
    )
    
    try:
        dictionary.save(os.path.join(args.out, "dictionary.dict"))
    except Exception as e:
        print(f"[WARN] Failed to save dictionary: {e}")
    
    id2tok = {int(v): k for k, v in dictionary.token2id.items()}
    vocab_list = [id2tok[i] for i in range(len(id2tok))]
    meta = {
        "vocab_size": len(vocab_list),
        "vocab_raw_sha1": _sha1_list(vocab_list),
        "note": "indices must align with beta.npy rows' V dimension and G.pt rows",
    }
    with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"id2tok": id2tok, "meta": meta}, f, ensure_ascii=False, indent=2)
    
    G = build_G_from_dictionary(
        dictionary, args.device,
        model_name="Alibaba-NLP/gte-large-en-v1.5",
        cache_path=args.g_cache, batch_size=256, max_length=8,
        pooling=args.g_pooling, replace_underscore=True
    )
    
    idf_vec = compute_idf_vec(dictionary, gamma=args.idf_gamma)
    
    model = S2WTM_Flex(
        mode="emb",
        emb_dim=train_ds.embs.shape[1],
        bow_dim=len(dictionary),
        n_topic=args.n_topic,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_projections=args.num_proj,
        beta=args.lambda_ot,
        loss_type="sph_sw",
        dist="unif_sphere",
        decoder_type=args.decoder_type,
        tau_w=args.tau_w,
        temperature=args.theta_temp,
        dropout=args.dropout,
        proj_type="linear",
        log_every=1000,
        learnable_temp=False,
    )
    
    with torch.no_grad():
        raw_tau = math.log(math.exp(args.theta_temp) - 1.0)
        model.wae._raw_temp.copy_(torch.tensor(raw_tau, device=model.wae._raw_temp.device))
    
    enc_params = list(model.wae.encoder.parameters()) + list(model.wae.proj.parameters())
    dec_params = [model.wae.topic_word_logits] if model.wae.decoder_type=="basis" else list(model.wae.decoder.parameters())
    
    dec_lr = min(args.lr * args.dec_lr_mul, args.dec_lr_max)
    opt = torch.optim.Adam(
        [{"params": enc_params, "lr": args.lr,   "weight_decay": 1e-5},
         {"params": dec_params, "lr": dec_lr,    "weight_decay": 0.0}],
        eps=1e-6,
    )
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=args.eta_min)
    
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=2, persistent_workers=True, prefetch_factor=2,
                           pin_memory=torch.cuda.is_available(),
                           collate_fn=train_ds.collate_fn_emb)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, persistent_workers=True, prefetch_factor=2,
                            pin_memory=torch.cuda.is_available(),
                            collate_fn=val_ds.collate_fn_emb)
    
    early = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    
    history = []
    best_epoch = -1
    model.wae.train()
    
    print("\n========== Training Started ==========\n")
    
    for ep in range(args.num_epochs):
        totals = []
        for batch in tr_loader:
            opt.zero_grad()
            _, bows, embs_b = batch
            bows = bows.to(args.device)
            embs_b = embs_b.to(args.device)
            
            x_reconst, z = model.wae(embs_b)
            
            t = bows / (bows.sum(dim=1, keepdim=True) + 1e-8)
            qn = F.normalize(embs_b, p=2, dim=1)
            sim = qn @ G.t()
            t_hat = F.softmax(sim / args.tau_g, dim=1)
            t = (1 - args.alpha_pbow) * t + args.alpha_pbow * t_hat
            
            w = idf_vec.to(args.device).unsqueeze(0)
            t = t * w
            t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-8)
            
            k = min(args.topk_mask_k, t.size(1)-1)
            if k > 0:
                _, idx = t.topk(k, dim=1)
                mask = torch.zeros_like(t).scatter(1, idx, 1.0)
                eps = 1e-6
                t = (t * mask) + eps
                t = t / t.sum(dim=1, keepdim=True)
            
            # CE loss
            logits = x_reconst / args.tau_rec
            logp = F.log_softmax(logits, dim=1)
            rec = -(t * logp).sum(dim=1).mean()
            
            # OT loss
            theta_prior = model.wae.sample(batch_size=z.size(0)).to(args.device)
            ot = model.wae.sp_swd_loss(z, theta_prior, num_projections=args.num_proj, device=args.device, p=2)
            
            total = rec + args.lambda_ot * ot
            total.backward()
            torch.nn.utils.clip_grad_norm_(enc_params, 5.0)
            torch.nn.utils.clip_grad_norm_(dec_params, 2.0)
            opt.step()
            
            totals.append(float(total.item()))
        
        mean_total_tr = float(np.mean(totals))
        mean_total_val = eval_mean_total(val_loader, model, G, args.device, args,
                                          args.tau_rec, args.lambda_ot, idf_vec)
        print(f"[epoch {ep+1}] train: {mean_total_tr:.6f} | val: {mean_total_val:.6f}")
        
        scheduler.step()
        
        history.append(mean_total_val)
        early.step(mean_total_val, model)
        if early.should_stop:
            print(f"\n[EARLY STOP] epoch {ep+1}, best val mean total={early.best:.6f}")
            best_epoch = ep + 1 - early.count
            if early.best_state is not None:
                model.wae.load_state_dict(early.best_state)
            break
    
    if best_epoch < 0:
        best_epoch = len(history) if history else args.num_epochs
    
    print(f"\n========== Training Completed ==========")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {early.best:.6f}")
    
    torch.save({"net": model.wae.state_dict(), "config": vars(args)}, os.path.join(args.out, "ckpt.pt"))
    
    theta = model.get_doc_topic_distribution(train_ds)
    beta = model.get_topic_word_dist(normalize=True)
    np.save(os.path.join(args.out, "theta_train.npy"), theta)
    np.save(os.path.join(args.out, "beta.npy"), beta)
    
    custom_stop = ["the","a","an","of","and","to","in","is","are"]
    topics = model.show_topic_words(dictionary, topK=10, hide_stopwords=custom_stop)
    
    with open(os.path.join(args.out, "topics_top10.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "top_words"])
        for k, words in enumerate(topics):
            w.writerow([k, " ".join(words)])
    
    with open(os.path.join(args.out, "topics_top10.jsonl"), "w", encoding="utf-8") as f:
        for k, words in enumerate(topics):
            f.write(json.dumps({"topic_id": k, "top_words": words}, ensure_ascii=False) + "\n")
    
    print(f"\nResults saved to: {args.out}")
    print(f"- Checkpoint: ckpt.pt")
    print(f"- Theta: theta_train.npy")
    print(f"- Beta: beta.npy")
    print(f"- Topics: topics_top10.csv / topics_top10.jsonl")

if __name__ == "__main__":
    main()