import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def build_G_from_dictionary(tokenizer, enc_model, dictionary, device="cuda", batch_size=64, normalize=True, max_length=32):

    id2token = {i: t for t, i in dictionary.token2id.items()}
    vocab = [id2token[i] for i in range(len(id2token))]
    embs = []
    enc_model.eval()

    for i in tqdm(range(0, len(vocab), batch_size), desc="Emb vocab (G)"):
        batch_tokens = vocab[i:i+batch_size]
        enc = tokenizer(
            batch_tokens,
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(device)
        out = enc_model(**enc)
        vec = out.last_hidden_state[:, 0]   # CLS
        if normalize:
            vec = F.normalize(vec, p=2, dim=1)
        embs.append(vec.detach().cpu())
    G = torch.cat(embs, dim=0)  
    return G


@torch.no_grad()
def compute_topic_embeddings(model, G):

    beta = model.get_topic_word_dist(normalize=True)   
    beta_t = torch.from_numpy(beta).float()            
    E_topic = beta_t @ G                               
    E_topic = F.normalize(E_topic, p=2, dim=1)
    return E_topic


@torch.no_grad()
def get_thetas(model, dataset, batch_size=256):
    return model.get_doc_topic_distribution(dataset)   


@torch.no_grad()
def evaluate_alignment(theta_np, E_topic, sent_embs, split_name="train"):

    theta = torch.from_numpy(theta_np).float()       
    mix = theta @ E_topic                              
    mix = F.normalize(mix, p=2, dim=1)
    sents = F.normalize(sent_embs, p=2, dim=1)

    cos = torch.sum(mix * sents, dim=1)              
    print(f"[Alignment/{split_name}] mean={cos.mean().item():.4f} | median={cos.median().item():.4f} | min={cos.min().item():.4f} | max={cos.max().item():.4f}")
    return cos


@torch.no_grad()
def evaluate_topic_diversity(E_topic):

    Et = F.normalize(E_topic, p=2, dim=1)
    sim = Et @ Et.T                    
    K = sim.size(0)
    off_diag = sim[~torch.eye(K, dtype=bool)]
    mean_sim = off_diag.mean().item()
    print(f"[Diversity] mean pairwise cosine = {mean_sim:.4f} (lower=more diverse)")
    return mean_sim


@torch.no_grad()
def evaluate_embedding_coherence(model, dictionary, G, topK=10):

    beta = model.get_topic_word_dist(normalize=True)  
    K, V = beta.shape
    beta_t = torch.from_numpy(beta).float()

    top_ids = torch.topk(beta_t, k=min(topK, V), dim=1).indices  
    G = F.normalize(G, p=2, dim=1)

    cos_list = []
    for k in range(K):
        idx = top_ids[k]                 
        W = G[idx]                       
        s = W @ W.T
        t = s[~torch.eye(s.size(0), dtype=bool)]
        if t.numel() > 0:
            cos_list.append(t.mean().item())
    mean_coh = float(np.mean(cos_list)) if cos_list else float("nan")
    print(f"[Embedding Coherence] mean top-{topK} word cosine = {mean_coh:.4f} (higher=more coherent)")
    return mean_coh


def evaluate_full_pipeline(
    model, dictionary, tokenizer, enc_model,
    train_ds, test_ds,
    embs, train_docs,
    device="cuda",
    vocab_batch_size=64,
    vocab_max_length=8,
    topK_coherence=10
):

    G = build_G_from_dictionary(
        tokenizer=tokenizer,
        enc_model=enc_model,
        dictionary=dictionary,
        device=device,
        batch_size=vocab_batch_size,
        normalize=True,
        max_length=vocab_max_length,
    )

    E_topic = compute_topic_embeddings(model, G)

    theta_train = model.get_doc_topic_distribution(train_ds)
    theta_test  = model.get_doc_topic_distribution(test_ds)

    N_train = len(train_docs)
    sent_embs_train = embs[:N_train].float()
    sent_embs_test  = embs[N_train:].float()

    _ = evaluate_alignment(theta_train, E_topic, sent_embs_train, split_name="train")
    _ = evaluate_alignment(theta_test,  E_topic, sent_embs_test,  split_name="test")

    _ = evaluate_topic_diversity(E_topic)
    _ = evaluate_embedding_coherence(model, dictionary, G, topK=topK_coherence)

    return {
        "G": G, "E_topic": E_topic,
        "theta_train": theta_train, "theta_test": theta_test,
        "sent_embs_train": sent_embs_train, "sent_embs_test": sent_embs_test
    }
