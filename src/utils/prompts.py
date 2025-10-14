# src/utils/prompts.py
import torch
import torch.nn as nn

# -------- Prompts (그대로) --------
sys_prompt = (
    "Based on the triplets from a knowledge graph, please answer the given question. "
    "Triples marked with *** (strong agreement) or ** (moderate agreement) indicate "
    "consensus across multiple retrieval perspectives. "
    "Please keep the answers as simple as possible and return all the possible answers "
    'as a list, each with a prefix "ans:".'
)
cot_prompt = (
    'Format your above answers by listing each answer on a separate line, '
    'starting with the prefix "ans:".'
)

# -------- Marker / Fact text utils (그대로) --------
def get_agreement_marker(cnt):
    if cnt >= 3:
        return "*** "
    elif cnt == 2:
        return "** "
    else:
        return ""

def create_fact_texts_with_markers(entity_list, sample, sel_indices, overlap_cnt):
    fact_texts = []
    for j_idx, j in enumerate(sel_indices.tolist()):
        h_ent = entity_list[sample['h_id_list'][j]]
        rel = sample['relation_list'][sample['r_id_list'][j]]
        t_ent = entity_list[sample['t_id_list'][j]]
        marker = get_agreement_marker(overlap_cnt[j_idx].item())
        fact_texts.append(f"{marker}({h_ent}, {rel}, {t_ent})")
    return fact_texts

def create_fact_texts_without_markers(entity_list, sample, sel_indices):
    fact_texts = []
    for j in sel_indices.tolist():
        h_ent = entity_list[sample['h_id_list'][j]]
        rel = sample['relation_list'][sample['r_id_list'][j]]
        t_ent = entity_list[sample['t_id_list'][j]]
        fact_texts.append(f"({h_ent}, {rel}, {t_ent})")
    return fact_texts

class Consensus_aware_Prompt(nn.Module):
    def __init__(self, triple_hidden_size, llm_hidden_size, gate_boost=1.0):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(triple_hidden_size, llm_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size)
        )
        self.overlap_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, llm_hidden_size)
        )
        self.gate_boost = gate_boost

    def forward(self, triple_embeds, overlap_cnt, fact_texts, sel_indices,
                tokenizer, llm, device, gate_weights=None, disable_emphasis=False):
        module_dtype = self.projector[0].weight.dtype
        selected_embeds = triple_embeds[sel_indices]
        struct_projs = self.projector(selected_embeds)

        overlap_norm = overlap_cnt.to(dtype=module_dtype) / overlap_cnt.max().clamp_min(1.0)
        sort_idx = torch.argsort(overlap_cnt, descending=False)
        sorted_struct = struct_projs[sort_idx]
        sorted_overlap = overlap_norm[sort_idx]
        sorted_texts = [fact_texts[i] for i in sort_idx.tolist()]
        sorted_overlap_cnt = overlap_cnt[sort_idx]

        if disable_emphasis:
            overlap_bias = torch.zeros_like(sorted_struct)
            g_sel = None
        else:
            overlap_bias = self.overlap_encoder(sorted_overlap.unsqueeze(-1))
            overlap_bias[sorted_overlap_cnt < 2] = 0.0
            g_sel = None
            if gate_weights is not None:
                g_full = gate_weights.to(dtype=module_dtype)
                g_sel = (1.0 + self.gate_boost * g_full[sel_indices])[sort_idx]
                g_sel[sorted_overlap_cnt < 2] = 1.0

        enhanced_structs = sorted_struct + overlap_bias

        fact_blocks = []
        for i, (struct_emb, fact_text) in enumerate(zip(enhanced_structs, sorted_texts)):
            text_ids = tokenizer(fact_text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            text_embs = llm.get_input_embeddings()(text_ids)
            if (not disable_emphasis) and (g_sel is not None):
                s = g_sel[i].view(1, 1).to(text_embs.dtype)
                text_embs = text_embs * s
            fact_blocks.append(torch.cat([struct_emb.unsqueeze(0).unsqueeze(0), text_embs], dim=1))
        return torch.cat(fact_blocks, dim=1)
