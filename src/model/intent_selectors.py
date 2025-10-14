from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
import numpy as np

def _safe_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def _load_G(g_cache, device, dtype=torch.bfloat16):
    obj = _safe_load(g_cache, device)
    if isinstance(obj, dict) and "G" in obj:
        G = obj["G"]
    else:
        G = obj
    return F.normalize(G.to(device=device, dtype=dtype), p=2, dim=1)

@dataclass
class SelectCfg:
    cum_threshold: float = 0.65
    min_k: int = 1
    max_k: int = 5
    normalize_rows: bool = True
    grad_to_beta: bool = False  

class IntentSelectorBridge:
    def __init__(self, run_dir: str, g_cache: str, device: str = None,
                 select_cfg: SelectCfg = SelectCfg(), dtype=torch.bfloat16):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = select_cfg
        self.dtype = dtype
        beta_np = np.load(os.path.join(run_dir, "beta.npy"))
        beta_init = torch.from_numpy(beta_np).to(self.device, dtype=self.dtype)  
        self.K, self.V = beta_init.shape

        ck = _safe_load(os.path.join(run_dir, "ckpt.pt"), self.device)
        self.ckpt = ck
        self.dec_type = ck["config"].get("decoder_type", "basis")

        self.G = _load_G(g_cache, self.device, dtype=self.dtype)  
        self.D = self.G.size(1)
        assert self.G.size(0) == self.V, f"G vocab({self.G.size(0)}) != beta V({self.V})"

        from Intent_selector.S2WTM import S2WTM_Flex
        self.selector = S2WTM_Flex(
            mode="emb", emb_dim=self.D, bow_dim=self.V,
            n_topic=self.K, decoder_type=self.dec_type,
            encode_dims=[self.D, 256, self.K], decode_dims=[self.K, 512, self.V],
            temperature=ck["config"].get("theta_temp", 0.8),
            proj_type="linear", dropout=0.0, batch_size=1,
            tau_w=ck["config"].get("tau_w", 1.3),
            basis_simplex=ck["config"].get("basis_simplex", False)
        ).to(self.device)
        self.selector.wae.load_state_dict(ck["net"], strict=False)
        self.selector.wae = self.selector.wae.to(dtype=self.dtype)
        self.selector.wae.eval()
        self.wae = self.selector.wae  

        self._learnable_beta = self.cfg.grad_to_beta
        if self._learnable_beta:
            print(f"[IntentBridge] Beta will be LEARNED (decoder_type={self.dec_type})")
            with torch.no_grad():
                if self.dec_type == "basis":
                    self._beta0 = torch.softmax(
                        self.wae.topic_word_logits.to(self.dtype) / self.wae.tau_w, dim=1
                    ).detach().clone()
                else: 
                    eye = torch.eye(self.K, device=self.device, dtype=self.dtype)
                    logits = self.wae.decode(eye).to(self.dtype)
                    self._beta0 = torch.softmax(logits, dim=1).detach().clone()
            self.T = None
            self.beta = None
            self._beta_param = True
        else:
            print(f"[IntentBridge] Beta is FIXED (loaded from beta.npy)")
            self.beta = F.normalize(beta_init, p=1, dim=1)  
            self.T = self._build_T(self.beta, self.G)       
            self._beta_param = False
            self._beta0 = None

    @torch.no_grad()
    def theta(self, q_emb: torch.Tensor) -> torch.Tensor:
        x = q_emb.to(self.device, dtype=self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.wae.encode(x)              
        z = F.normalize(z, p=2, dim=1)
        z = self.wae.proj(z)                 
        tau = self.wae.tau()
        return F.softmax(z / tau, dim=1)     

    def _build_T(self, beta: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        T = beta @ G
        return F.normalize(T, p=2, dim=1) if self.cfg.normalize_rows else T

    def beta_now(self, with_grad: bool = False) -> torch.Tensor:
        if self.dec_type == "basis":
            logits = self.wae.topic_word_logits.to(self.dtype) 
            if with_grad:
                beta = torch.softmax(logits / self.wae.tau_w, dim=1)
            else:
                with torch.no_grad():
                    beta = torch.softmax(logits / self.wae.tau_w, dim=1)
            return beta
        else:
            eye = torch.eye(self.K, device=self.device, dtype=self.dtype)
            if with_grad:
                out = self.wae.decode(eye).to(self.dtype)
            else:
                with torch.no_grad():
                    out = self.wae.decode(eye).to(self.dtype)
            beta = torch.softmax(out, dim=1)
            return beta

    def _ensure_T(self) -> torch.Tensor:
        if not self._beta_param:
            return self.T
        if self.dec_type == "basis":
            logits = self.wae.topic_word_logits.to(self.dtype) 
            beta = torch.softmax(logits / self.wae.tau_w, dim=1)
            return self._build_T(beta, self.G)
        elif self.dec_type == "mlp":
            eye = torch.eye(self.K, device=self.device, dtype=self.dtype)
            logits = self.wae.decode(eye).to(self.dtype)        
            beta = torch.softmax(logits, dim=1)
            return self._build_T(beta, self.G)
        else:
            raise ValueError(f"Unknown decoder_type: {self.dec_type}")

    @staticmethod
    def _pick(theta, thr, min_k, max_k):
        vals, idx = torch.sort(theta, dim=1, descending=True)  
        csum = torch.cumsum(vals, dim=1)
        n = int((csum[0] < float(thr)).sum().item()) + 1
        n = max(min_k, min(max_k, n))
        sel_idx = idx[0, :n].contiguous()
        sel_vals = vals[0, :n].contiguous()
        return sel_idx, sel_vals

    def select(self, q_emb: torch.Tensor):
        theta = self.theta(q_emb)  
        sel_idx, sel_vals = self._pick(theta, self.cfg.cum_threshold, self.cfg.min_k, self.cfg.max_k)
        T = self._ensure_T()       
        intent_embs = T.index_select(0, sel_idx)  
        return intent_embs, sel_idx, sel_vals

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.wae = self.wae.to(dtype=dtype)
        self.G = self.G.to(dtype=dtype)
        if hasattr(self, "beta") and self.beta is not None:
            self.beta = self.beta.to(dtype=dtype)
        if hasattr(self, "T") and self.T is not None:
            self.T = self.T.to(dtype=dtype)
        if hasattr(self, "_beta0") and self._beta0 is not None:
            self._beta0 = self._beta0.to(dtype=dtype)
        return self

    def get_beta_drift_loss(self, alpha: float = 0.1) -> torch.Tensor:
        if not self._learnable_beta or self._beta0 is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        beta_current = self.beta_now(with_grad=True) 
        kl = (self._beta0 * (self._beta0.clamp_min(1e-8).log() -
                             beta_current.clamp_min(1e-8).log())).sum()
        return (alpha * kl).to(self.dtype)
