from typing import List, Optional, Tuple, Callable, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from gensim.corpora import Dictionary
from gensim.models import TfidfModel


class TopicDataset(Dataset):
    def __init__(self,
                 texts: List[str],
                 tfidf_sparse,
                 dictionary: Dictionary,
                 embs: Optional[torch.Tensor] = None):
        self.texts = texts
        self.dictionary = dictionary
        V = len(dictionary)

        bows = torch.zeros((len(texts), V), dtype=torch.float32)
        for i, vec in enumerate(tfidf_sparse):
            if len(vec) > 0:
                ids, vals = zip(*vec)
                bows[i, list(ids)] = torch.tensor(vals, dtype=torch.float32)
        self.bows = bows

        if embs is not None and isinstance(embs, np.ndarray):
            embs = torch.from_numpy(embs)
        self.embs = None if embs is None else embs.float().contiguous()
        if self.embs is not None:
            assert self.embs.size(0) == len(self.texts), "texts/embs length mismatch"

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = {"text": self.texts[idx], "bow": self.bows[idx]}
        if self.embs is not None: item["emb"] = self.embs[idx]
        return item

    @staticmethod
    def collate_fn_bow(batch):
        txts = [b["text"] for b in batch]
        bows = torch.stack([b["bow"] for b in batch], dim=0)
        return txts, bows

    @staticmethod
    def collate_fn_emb(batch):
        txts = [b["text"] for b in batch]
        bows = torch.stack([b["bow"] for b in batch], dim=0)
        embs = torch.stack([b["emb"] for b in batch], dim=0)
        return txts, bows, embs

    @staticmethod
    def get_collate_fn(mode: str):
        assert mode in ("bow", "emb")
        return TopicDataset.collate_fn_bow if mode == "bow" else TopicDataset.collate_fn_emb



def _tokenize_whitespace(texts: List[str]) -> List[List[str]]:
    return [t.split() for t in texts]

def _tfidf_sparse_list(tokens_list: List[List[str]], dictionary: Dictionary, tfidf_model: TfidfModel):
    bows = [dictionary.doc2bow(doc) for doc in tokens_list]
    tfidf_sparse = [tfidf_model[vec] for vec in bows]
    return tfidf_sparse

@torch.no_grad()
def compute_hf_embeddings(
    texts: List[str],
    hf_tokenizer,
    hf_model,
    batch_size: int = 256,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = hf_model.to(dev).eval()
    if max_length is None:
        max_len_cfg = getattr(getattr(hf_model, "config", None), "max_position_embeddings", None)
        max_length = int(max_len_cfg) if max_len_cfg is not None else 512

    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = hf_tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        out = hf_model(**enc)
        cls_emb = out.last_hidden_state[:, 0]
        cls_emb = F.normalize(cls_emb, p=2, dim=1)
        embs.append(cls_emb.cpu())
    return torch.cat(embs, dim=0)


def build_unified_topic_datasets(
    train_texts: List[str],
    test_texts: Optional[List[str]] = None,
    *,
    emb_train: Optional[np.ndarray] = None,
    emb_test: Optional[np.ndarray] = None,
    hf_tokenizer = None,
    hf_model = None,
    emb_batch_size: int = 256,
    emb_max_length: Optional[int] = None,
    emb_device: Optional[str] = None,
    tokenizer: Callable[[List[str]], List[List[str]]] = _tokenize_whitespace,
    shared: bool = False,
    dict_no_below: int = 5,
    dict_no_above: float = 0.5,
    dict_keep_n: Optional[int] = None,
):
    def _mk_dict(tokens_list):
        d = Dictionary(tokens_list)
        d.filter_extremes(no_below=dict_no_below, no_above=dict_no_above, keep_n=dict_keep_n)
        return d

    train_tokens = tokenizer(train_texts)
    test_tokens  = tokenizer(test_texts) if test_texts is not None else None

    if test_texts is None:
        dictionary = _mk_dict(train_tokens)
        corpus_fit = [dictionary.doc2bow(doc) for doc in train_tokens]
        tfidf_model = TfidfModel(corpus_fit)
        tfidf_train = _tfidf_sparse_list(train_tokens, dictionary, tfidf_model)

        if emb_train is None and hf_tokenizer is not None and hf_model is not None:
            emb_train_t = compute_hf_embeddings(train_texts, hf_tokenizer, hf_model,
                                                batch_size=emb_batch_size, max_length=emb_max_length, device=emb_device)
        else:
            emb_train_t = None if emb_train is None else (
                torch.from_numpy(emb_train).float() if isinstance(emb_train, np.ndarray) else emb_train.float()
            )

        train_ds = TopicDataset(train_texts, tfidf_train, dictionary, embs=emb_train_t)
        return train_ds, None, dictionary

    fit_tokens = (train_tokens + test_tokens) if shared else train_tokens
    dictionary = _mk_dict(fit_tokens)
    corpus_fit = [dictionary.doc2bow(doc) for doc in fit_tokens]
    tfidf_model = TfidfModel(corpus_fit)
    tfidf_train = _tfidf_sparse_list(train_tokens, dictionary, tfidf_model)
    tfidf_test  = _tfidf_sparse_list(test_tokens,  dictionary, tfidf_model)

    if emb_train is None and hf_tokenizer is not None and hf_model is not None:
        emb_train_t = compute_hf_embeddings(train_texts, hf_tokenizer, hf_model,
                                            batch_size=emb_batch_size, max_length=emb_max_length, device=emb_device)
    else:
        emb_train_t = None if emb_train is None else (
            torch.from_numpy(emb_train).float() if isinstance(emb_train, np.ndarray) else emb_train.float()
        )
    if emb_test is None and hf_tokenizer is not None and hf_model is not None:
        emb_test_t = compute_hf_embeddings(test_texts, hf_tokenizer, hf_model,
                                           batch_size=emb_batch_size, max_length=emb_max_length, device=emb_device)
    else:
        emb_test_t = None if emb_test is None else (
            torch.from_numpy(emb_test).float() if isinstance(emb_test, np.ndarray) else emb_test.float()
        )

    train_ds = TopicDataset(train_texts, tfidf_train, dictionary, embs=emb_train_t)
    test_ds  = TopicDataset(test_texts,  tfidf_test,  dictionary, embs=emb_test_t)
    return train_ds, test_ds, dictionary
