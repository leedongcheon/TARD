import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions.von_mises_fisher import rand_von_mises_fisher

class WAE(nn.Module):

    def __init__(self, encode_dims, decode_dims, dropout, nonlin, dist, batch_size, temperature,
                 learnable_temp: bool = False, proj_type: str = 'linear', use_latent_dropout: bool = False,
                 decoder_type: str = 'basis', tau_w: float = 1.5, basis_simplex: bool = False):
        super().__init__()

        self.dist=dist
        self.batch_size = batch_size
        self.use_latent_dropout = use_latent_dropout
        self.z_dim = encode_dims[-1]
        self.nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[nonlin]
        self.dropout_feat = nn.Dropout(p=dropout)
        self.latent_dropout = nn.Dropout(p=dropout) if use_latent_dropout else nn.Identity()
        self.decoder_type = decoder_type
        self.tau_w = tau_w
        self.vocab_size = decode_dims[-1]
        self.basis_simplex = basis_simplex

        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i+1]) for i in range(len(encode_dims)-1)
        })

        if self.decoder_type == 'basis':
            self.topic_word_logits = nn.Parameter(torch.randn(self.z_dim, self.vocab_size) * 0.02)
        else:  # 'mlp'
            self.decoder = nn.ModuleDict({
                f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i+1], bias=(i < len(decode_dims)-2))
                for i in range(len(decode_dims)-1)
            })

        if proj_type == 'identity':
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(self.z_dim, self.z_dim, bias=False)
            with torch.no_grad(): self.proj.weight.copy_(torch.eye(self.z_dim))
        elif proj_type == 'original':
            self.proj = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim),
                nn.ReLU(),
                nn.Linear(self.z_dim, self.z_dim),
                nn.LayerNorm(self.z_dim)
            )
            for m in self.proj.modules():
                if isinstance(m, nn.Linear):
                    if m.bias is not None: nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=1e-3)
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim),
                nn.ReLU(),
                nn.Linear(self.z_dim, self.z_dim, bias=False)
            )
            for m in self.proj.modules():
                if isinstance(m, nn.Linear):
                    if m.bias is not None: nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=1e-3)

        raw = math.log(math.exp(float(temperature)) - 1.0)
        if learnable_temp:
            self._raw_temp = nn.Parameter(torch.tensor(raw))
        else:
            self.register_buffer("_raw_temp", torch.tensor(raw))

    def tau(self): return F.softplus(self._raw_temp) + 1e-4

    def encode(self, x):
        h = x
        n = len(self.encoder)
        for i, layer in enumerate(self.encoder.values()):
            h = layer(h)
            if i < n - 1: h = self.nonlin(self.dropout_feat(h))
        return h

    def decode(self, theta):
        if self.decoder_type == 'basis':
            if self.basis_simplex:
                beta = F.softmax(self.topic_word_logits / self.tau_w, dim=1)
                return theta @ beta
            else:
                logits = theta @ self.topic_word_logits 
                return logits
        else:
            h = theta
            n_layers = len(self.decoder)
            for i, layer in enumerate(self.decoder.values()):
                h = layer(h)
                if i < n_layers-1: h = self.nonlin(self.dropout_feat(h))
            return h

    def forward(self, x):
        z = self.encode(x)
        z_norm = F.normalize(z, p=2, dim=1)
        z_proj = self.proj(z_norm)
        z_proj = self.latent_dropout(z_proj)
        tau = self.tau()
        theta = F.softmax(z_proj / tau, dim=-1)
        x_reconst = self.decode(theta)
        return x_reconst, z_norm

    def sample(self, batch_size=None):
        n = batch_size if batch_size is not None else self.batch_size
        if self.dist=='vmf':
            mu = torch.eye(self.z_dim, dtype=torch.float)[0]; kappa=10
            X = rand_von_mises_fisher(mu, kappa=kappa, N=n)
            return torch.from_numpy(X).float()
        elif self.dist=='mixture_vmf':
            ps = np.ones(2*self.z_dim)/(2*self.z_dim)
            mus = torch.cat((torch.eye(self.z_dim, dtype=torch.float64), -torch.eye(self.z_dim, dtype=torch.float64)), 0)
            mus = F.normalize(mus, p=2, dim=-1)
            Z = np.random.multinomial(n, ps)
            idx = np.where(Z>0)[0]; nums = Z[idx]
            X = []
            for i, k in enumerate(nums):
                vmf = rand_von_mises_fisher(mus[idx[i]], kappa=10, N=int(k))
                X.append(torch.tensor(vmf, dtype=torch.float))
            return torch.cat(X, dim=0)
        else:  
            target_latent = torch.randn(n, self.z_dim)
            return F.normalize(target_latent, p=2, dim=-1)

    def sp_sliced_cost(self, Xs, Xt, U, p=2):
        Zs = torch.matmul(torch.transpose(U,1,2)[:,None], Xs[:,:,None]).reshape(U.size(0), Xs.size(0), 2)
        Zt = torch.matmul(torch.transpose(U,1,2)[:,None], Xt[:,:,None]).reshape(U.size(0), Xt.size(0), 2)
        Zs = F.normalize(Zs, p=2, dim=-1); Zt = F.normalize(Zt, p=2, dim=-1)
        ang_s = (torch.atan2(-Zs[:,:,1], -Zs[:,:,0]) + np.pi) / (2*np.pi)
        ang_t = (torch.atan2(-Zt[:,:,1], -Zt[:,:,0]) + np.pi) / (2*np.pi)
        ang_s, _ = torch.sort(ang_s, dim=1); ang_t, _ = torch.sort(ang_t, dim=1)
        return torch.mean((ang_s - ang_t).abs().pow(p))

    def sp_swd_loss(self, Xs, Xt, num_projections, device, p=2):
        d = Xs.shape[1]
        Z = torch.randn((num_projections,d,2), device=device)
        U, _ = torch.linalg.qr(Z)
        return self.sp_sliced_cost(Xs, Xt, U, p=p)
