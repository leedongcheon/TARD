import torch


def to_llm(x, llm):
    p = next(llm.parameters())
    return x.to(device=p.device, dtype=p.dtype)


def compute_triple_embeddings(
    h_id_tensor, r_id_tensor, t_id_tensor,
    entity_embs, num_non_text_entities,
    relation_embs, retriever, triple_proj, device
):
    proj_dtype = next(triple_proj[0].parameters()).dtype
    
    # Build full entity embeddings
    non_text_emb_cache = retriever.non_text_entity_emb(
        torch.LongTensor([0]).to(device)
    ).detach()
    
    full_entity_embs = torch.cat([
        entity_embs,
        non_text_emb_cache.expand(num_non_text_entities, -1)
    ], dim=0).to(proj_dtype)
    
    # Get h, r, t embeddings
    h_embs = full_entity_embs[h_id_tensor]
    r_embs = relation_embs[r_id_tensor].to(proj_dtype)
    t_embs = full_entity_embs[t_id_tensor]
    
    # Concatenate and project
    triple_inputs = torch.cat([h_embs, r_embs, t_embs], dim=1)
    triple_embeds = triple_proj(triple_inputs)
    
    return triple_embeds