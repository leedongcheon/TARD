import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')
    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)
    def message(self, x_j):
        return x_j

class DDE(nn.Module):
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())

    def forward(self, topic_entity_one_hot, edge_index, reverse_edge_index):
        result_list = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)

        return result_list

class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        num_intents=3
    ):
        super().__init__()

        self.emb_size = emb_size
        self.num_intents = num_intents

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)

        self.intent_embs = nn.Parameter(torch.randn(num_intents, emb_size))
        nn.init.orthogonal_(self.intent_embs)


        pred_in_size = 5 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        external_intents=None  
    ):

        device = entity_embs.device

        # Entity embeddings
        h_e = torch.cat([
            entity_embs,
            self.non_text_entity_emb(
                torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
        ], dim=0)

        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        h_r = relation_embs[r_id_tensor]
        h_head = h_e[h_id_tensor]
        h_tail = h_e[t_id_tensor]

        num_triples = len(h_r)

        if external_intents is not None:
            assert external_intents.size(-1) == self.emb_size, \
                f"external_intents dim {external_intents.size(-1)} != emb_size {self.emb_size}"
            intent_matrix = external_intents                   # [I, D]
        else:
            intent_matrix = self.intent_embs                   # [I_fixed, D]

        I = intent_matrix.size(0)

        all_logits = []
        for intent_idx in range(I):
            intent_vec = intent_matrix[intent_idx].unsqueeze(0).expand(num_triples, -1)

            h_triple = torch.cat([
                intent_vec,                      
                h_q.expand(num_triples, -1),     
                h_head,                          
                h_r,                             
                h_tail                          
            ], dim=1)

            logit = self.pred(h_triple)  
            all_logits.append(logit)

        all_logits = torch.cat(all_logits, dim=1)  

        return all_logits