# S2WTM.py
import math, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List
from WAE import WAE
import torch.nn as nn
class S2WTM_Flex(nn.Module):
    def __init__(self,
                 bow_dim: int,
                 n_topic: int = 20,
                 mode: str = 'bow',
                 emb_dim: Optional[int] = None,
                 device: Optional[str] = None,
                 taskname: Optional[str] = None,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 100,
                 log_every: int = 10,
                 beta: float = 1.0,          
                 loss_type: str = 'sph_sw',
                 num_projections: int = 256,
                 p: int = 2,
                 encode_dims=None,
                 decode_dims=None,
                 dropout: float = 0.5,
                 nonlin='relu',
                 dist: str = 'unif_sphere',
                 batch_size: int = 256,
                 temperature: float = 1.0,
                 learnable_temp: bool = False,
                 proj_type: str = 'linear',
                 use_latent_dropout: bool = False,
                 decoder_type: str = 'basis',
                 tau_w: float = 1.5,
                 basis_simplex: bool = False):
        super().__init__()
        assert mode in ('bow', 'emb')
        if mode == 'emb': assert emb_dim is not None

        self.mode = mode
        self.emb_dim = emb_dim
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.id2token = None
        self.taskname = taskname
        self.dropout = dropout
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.log_every = log_every
        self.beta = beta
        self.loss_type = loss_type
        self.num_projections = num_projections
        self.p = p

        if encode_dims is None:
            encode_dims = [bow_dim, 1024, 512, n_topic] if mode=="bow" else [emb_dim, 256, n_topic]
        if decode_dims is None:
            decode_dims = [n_topic, 512, bow_dim]

        self.wae = WAE(
            encode_dims=encode_dims,
            decode_dims=decode_dims,
            dropout=dropout,
            nonlin=nonlin,
            dist=dist,
            batch_size=self.batch_size,
            temperature=temperature,
            learnable_temp=learnable_temp,
            proj_type=proj_type,
            use_latent_dropout=use_latent_dropout,
            decoder_type=decoder_type,
            tau_w=tau_w,
            basis_simplex=basis_simplex,
        ).to(self.device)

        enc_params = list(self.wae.encoder.parameters()) + list(self.wae.proj.parameters())
        if decoder_type == "basis":
            dec_params = [self.wae.topic_word_logits]
        else:
            dec_params = list(self.wae.decoder.parameters())
        self.opt = torch.optim.Adam(
            [{"params": enc_params, "lr": self.learning_rate, "weight_decay": 1e-5},
             {"params": dec_params,  "lr": max(self.learning_rate*6, 5e-3), "weight_decay": 0.0}],
        )

    def train(self, train_data, verbose: bool = False, log_every_steps: int = 50,
              warmup_epochs: int = 0, warmup_dropout_p: float = 0.0,
              G: Optional[torch.Tensor] = None, alpha_align: float = 0.0,
              use_pseudo_bow: bool = True, lambda_pb: float = 0.8, tau_g: float = 0.1,
              tau_rec_start: float = 0.9, tau_rec_end: float = 0.7, tau_rec_warm_epochs: int = 2):

        if verbose:
            print(f"Settings: mode={self.mode} n_topic={self.n_topic} num_proj={self.num_projections} "
                  f"beta(SWD)={self.beta} decoder={self.wae.decoder_type} basis_simplex={self.wae.basis_simplex}")

        self.wae.train()
        self.id2token = {v: k for k, v in train_data.dictionary.token2id.items()}

        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2,
                            persistent_workers=True, prefetch_factor=2,
                            collate_fn=train_data.collate_fn_bow if self.mode=="bow" else train_data.collate_fn_emb,
                            pin_memory=torch.cuda.is_available())

        step = 0
        for ep in range(self.num_epochs):
            prog = min(1.0, ep / max(1, tau_rec_warm_epochs)) if tau_rec_warm_epochs>0 else 1.0
            tau_rec = tau_rec_start + (tau_rec_end - tau_rec_start) * prog

            for batch in loader:
                step += 1
                self.opt.zero_grad()

                if self.mode == "bow":
                    _, bows = batch; bows = bows.to(self.device); x_reconst, z = self.wae(bows)
                    embs = None
                else:
                    _, bows, embs = batch; bows = bows.to(self.device); embs = embs.to(self.device)
                    x_reconst, z = self.wae(embs)

                t = bows / (bows.sum(dim=1, keepdim=True) + 1e-8)

                if use_pseudo_bow and (embs is not None) and (G is not None):
                    qn = F.normalize(embs, p=2, dim=1); sim = qn @ G.t()
                    t_hat = F.softmax(sim / tau_g, dim=1)
                    t = (1 - lambda_pb) * t + lambda_pb * t_hat

                if self.wae.decoder_type == 'basis' and not self.wae.basis_simplex:
                    logits = x_reconst / tau_rec
                    logp   = F.log_softmax(logits, dim=1)
                    rec    = -(t * logp).sum(dim=1).mean()
                    phi_prob_for_debug = logp.exp()
                elif self.wae.decoder_type == 'basis' and self.wae.basis_simplex:
                    phi = x_reconst.clamp_min(1e-8)
                    rec = -(t * phi.log()).sum(dim=1).mean()
                    phi_prob_for_debug = phi
                else:
                    logp = F.log_softmax(x_reconst, dim=1)
                    rec  = -(t * logp).sum(dim=1).mean()
                    phi_prob_for_debug = logp.exp()

                theta_prior = self.wae.sample(batch_size=z.size(0)).to(self.device)
                ot = self.wae.sp_swd_loss(z, theta_prior, num_projections=self.num_projections, device=self.device, p=self.p)
                total = rec + self.beta * ot
                total.backward()
                enc_params = list(self.wae.encoder.parameters()) + list(self.wae.proj.parameters())
                if self.wae.decoder_type == 'basis':
                    dec_params = [self.wae.topic_word_logits]
                else:
                    dec_params = list(self.wae.decoder.parameters())
                torch.nn.utils.clip_grad_norm_(enc_params, 2.0)
                torch.nn.utils.clip_grad_norm_(dec_params, 1.0)
                self.opt.step()

                if log_every_steps and (step % log_every_steps == 0):
                    Ht   = -(t * (t.clamp_min(1e-9)).log()).sum(1).mean()
                    Hphi = -(phi_prob_for_debug * phi_prob_for_debug.clamp_min(1e-9).log()).sum(1).mean()
                    print(f"[ep {ep+1} | step {step}] rec={rec.item():.4f} ot={ot.item():.4f} total={total.item():.4f} "
                          f"| H(t)≈{Ht.item():.3f} H(φ)≈{Hphi.item():.3f} τ_rec={tau_rec:.2f}")

    @torch.no_grad()
    def get_doc_topic_distribution(self, dataset, n_samples: int = 1):
        self.wae.eval()
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2,
                            collate_fn=dataset.collate_fn_bow if self.mode == "bow" else dataset.collate_fn_emb)
        outs = []
        for _ in range(n_samples):
            thetas = []
            for batch in loader:
                x = (batch[1] if self.mode=='bow' else batch[2]).to(self.device)
                z = self.wae.encode(x)
                z = F.normalize(z, p=2, dim=1)
                z_proj = self.wae.proj(z)
                theta = F.softmax(z_proj / self.wae.tau(), dim=1)
                thetas.append(theta.cpu().numpy())
            outs.append(np.concatenate(thetas, axis=0))
        return np.mean(np.stack(outs, axis=0), axis=0)

    @torch.no_grad()
    def get_topic_word_dist(self, normalize: bool = True):
        self.wae.eval()
        if self.wae.decoder_type == 'basis':
            beta = torch.softmax(self.wae.topic_word_logits / self.wae.tau_w, dim=1)
            return beta.detach().cpu().numpy()
        else:
            eye = torch.eye(self.n_topic, device=self.device)
            logits = self.wae.decode(eye)
            if normalize: return F.softmax(logits, dim=1).detach().cpu().numpy()
            return logits.detach().cpu().numpy()

    @torch.no_grad()
    def show_topic_words(self, dictionary, topK: int = 15, hide_stopwords: Optional[List[str]] = None):
        self.wae.eval()
        if self.id2token is None and dictionary is not None:
            self.id2token = {v: k for k, v in dictionary.token2id.items()}
        if self.wae.decoder_type == 'basis':
            W = torch.softmax(self.wae.topic_word_logits / self.wae.tau_w, dim=1)
        else:
            eye = torch.eye(self.n_topic, device=self.device)
            W = F.softmax(self.wae.decode(eye), dim=1)

        _, indices = torch.topk(W, topK*5, dim=1) 
        indices = indices.cpu().tolist()
        stop = set(w.lower() for w in (hide_stopwords or []))
        topic_words = []
        for row in indices:
            words = []
            for idx in row:
                token = self.id2token[idx]
                if token.lower() in stop: continue
                words.append(token)
                if len(words) >= topK: break
            topic_words.append(words)
        return topic_words
    def compute_reconstruction_loss(self, bows, x_reconst, t):

        if self.wae.decoder_type == 'basis' and not self.wae.basis_simplex:
            logits = x_reconst / self.wae.tau()
            logp   = F.log_softmax(logits, dim=1)
            rec    = -(t * logp).sum(dim=1).mean()
        elif self.wae.decoder_type == 'basis' and self.wae.basis_simplex:
            phi = x_reconst.clamp_min(1e-8)
            rec = -(t * phi.log()).sum(dim=1).mean()
        else:
            logp = F.log_softmax(x_reconst, dim=1)
            rec  = -(t * logp).sum(dim=1).mean()
        return rec

    def compute_ot_loss(self, z, device):

        theta_prior = self.wae.sample(batch_size=z.size(0)).to(device)
        ot = self.wae.sp_swd_loss(z, theta_prior, num_projections=self.num_projections, device=device, p=self.p)
        return ot
    def compute_loss(self, bows, x_reconst, z, t, device):
        rec_loss = self.compute_reconstruction_loss(bows, x_reconst, t)
        
        ot_loss = self.compute_ot_loss(z, device)
        
        total_loss = rec_loss + self.beta * ot_loss
        return total_loss