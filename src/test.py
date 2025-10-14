import os, json, torch
from tqdm import tqdm

from src.utils.metrics import extract_ans_lines
from src.utils.prompts import sys_prompt, cot_prompt, create_fact_texts_with_markers
from src.utils.kg_embed import to_llm, compute_triple_embeddings
from src.selection import per_intent_topk_no_noise, select_fixed_total_by_intents
from src.setup import prepare_sample


@torch.no_grad()
def run_test_inference(args, config, dataloader, retriever, triple_proj, enhanced_prompt_module,
                       llm, tokenizer, device, bridge):

    llm.eval()
    retriever.eval()
    triple_proj.eval()
    enhanced_prompt_module.eval()
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    # Read config values
    test_cfg = config.get('test', {})
    train_cfg = config['training']
    
    gen_max_new_tokens = getattr(args, 'gen_max_new_tokens', None) or test_cfg.get('gen_max_new_tokens', 64)
    fixed_total_test = getattr(args, 'fixed_total_test', None) or test_cfg.get('fixed_total_test', 100)
    per_intent_total = train_cfg['per_intent_total']
    gate_boost = config['neural_prompt']['gate_boost']
    kg_wrapper = config['neural_prompt']['kg_wrapper']

    # Prepare answer lead tokens
    ans_lead_ids = tokenizer("Answer:\n", add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    ans_lead_embeds = to_llm(llm.get_input_embeddings()(ans_lead_ids), llm)
    results = []

    for idx, sample in enumerate(tqdm(dataloader, desc="test-generate")):
        if hasattr(args, 'limit_test') and args.limit_test and idx >= args.limit_test:
            break

        # Prepare sample
        (h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
         num_non_text_entities, relation_embs, topic_entity_one_hot, _, _) = prepare_sample(device, sample)

        # Convert to bfloat16
        q_emb = q_emb.to(device=device, dtype=torch.bfloat16)
        entity_embs = entity_embs.to(device=device, dtype=torch.bfloat16)
        relation_embs = relation_embs.to(device=device, dtype=torch.bfloat16)
        if torch.is_floating_point(topic_entity_one_hot):
            topic_entity_one_hot = topic_entity_one_hot.to(device=device, dtype=torch.bfloat16)

        # Intent selection
        intent_embs, _, _ = bridge.select(q_emb)
        if intent_embs is not None:
            intent_embs = intent_embs.to(device=device, dtype=torch.bfloat16)

        # Retrieval
        logits_TI = retriever(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            external_intents=intent_embs
        )

        T, I = logits_TI.shape
        k_each = max(1, int(per_intent_total) // max(1, I))
        raw_masks = per_intent_topk_no_noise(logits_TI, k_each).to(dtype=logits_TI.dtype)

        # Triple selection
        if fixed_total_test > 0:
            _, sel_indices = select_fixed_total_by_intents(
                logits_TI, k_each=k_each, total_N=fixed_total_test
            )
        else:
            gate_tmp = raw_masks.sum(dim=1) / max(1, I)
            sel_indices = torch.nonzero(gate_tmp > 0, as_tuple=True)[0]
            if sel_indices.numel() == 0:
                sel_indices = torch.topk(gate_tmp, k=1, largest=True).indices

        gate = torch.zeros(T, dtype=raw_masks.dtype, device=raw_masks.device)
        gate[sel_indices] = raw_masks[sel_indices].sum(dim=1) / max(1, I)
        overlap_cnt_sel = raw_masks[sel_indices].sum(dim=1)

        # Compute triple embeddings
        triple_embeds = compute_triple_embeddings(
            h_id_tensor, r_id_tensor, t_id_tensor,
            entity_embs, num_non_text_entities,
            relation_embs, retriever, triple_proj, device
        )
        scale = (1.0 + gate_boost * gate).to(triple_embeds.dtype).unsqueeze(1)
        triple_embeds = triple_embeds * scale

        # Create fact texts
        entity_list = sample['text_entity_list'] + sample['non_text_entity_list']
        fact_texts = create_fact_texts_with_markers(entity_list, sample, sel_indices, overlap_cnt_sel)

        # Create neural prompt
        neural_prompt = enhanced_prompt_module(
            triple_embeds, overlap_cnt_sel, fact_texts, sel_indices,
            tokenizer, llm, device, gate_weights=gate, disable_emphasis=False
        )

        # Prepare question
        qtext = sample['question'].strip()
        if not qtext.endswith('?'):
            qtext += '?'
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question:\n{qtext}\n{cot_prompt}"},
        ]
        chat_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(device)
        chat_embeds = to_llm(llm.get_input_embeddings()(chat_ids), llm)

        # KG wrapper tokens
        kg_start_ids = tokenizer(kg_wrapper['start'], add_special_tokens=False, return_tensors='pt').input_ids.to(device)
        kg_end_ids = tokenizer(kg_wrapper['end'], add_special_tokens=False, return_tensors='pt').input_ids.to(device)
        kg_start_embeds = to_llm(llm.get_input_embeddings()(kg_start_ids), llm)
        kg_end_embeds = to_llm(llm.get_input_embeddings()(kg_end_ids), llm)

        # Construct full prompt
        prompt_embeds = torch.cat([
            chat_embeds,
            kg_start_embeds,
            neural_prompt,
            kg_end_embeds,
            ans_lead_embeds
        ], dim=1)
        attn_mask = torch.ones(prompt_embeds.size()[:-1], dtype=torch.long, device=prompt_embeds.device)

        # Generate
        gen_out = llm.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attn_mask,
            max_new_tokens=gen_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        
        text_full = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
        preds = extract_ans_lines(text_full)

        # Save result
        out = {
            "id": sample.get("id", None),
            "question": qtext,
            "prediction": [f"ans: {p}" for p in preds],
            "ground_truth": [a for a in sample.get('a_entity', []) if isinstance(a, str)],
            "num_union_triples": int(sel_indices.numel()),
        }
        results.append(out)

    # Save predictions
    os.makedirs(os.path.dirname(args.test_output) or ".", exist_ok=True)
    with open(args.test_output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"[TEST] Saved {len(results)} predictions to {args.test_output}")