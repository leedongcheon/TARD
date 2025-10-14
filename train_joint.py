import os, gc, argparse, warnings, pathlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

from src.setup import set_seed, prepare_sample
from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_joint_retriever_llm
from src.model.Intent_aware_Retriever import Retriever
from src.model.intent_selectors import IntentSelectorBridge, SelectCfg
from src.utils.metrics import normalize_answer, extract_ans_lines
from src.selection import gumbel_topk_batch, per_intent_topk_no_noise, get_per_intent_quota
from src.test import run_test_inference
from src.utils.prompts import (
    sys_prompt, cot_prompt, Consensus_aware_Prompt,
    create_fact_texts_with_markers, create_fact_texts_without_markers
)
from src.losses import (
    focal_loss_with_pos_weight, ranking_loss_vectorized,
    topk_decorrelation_loss, pattern_diversity_loss
)
from src.utils.kg_embed import to_llm, compute_triple_embeddings


def compute_hit_score(prediction_text, ground_truth_list):
    pred_answers = extract_ans_lines(prediction_text)
    if len(pred_answers) == 0:
        return 0.0
    pred_set = {normalize_answer(a) for a in pred_answers}
    gold_set = {normalize_answer(a) for a in ground_truth_list}
    return 1.0 if len(pred_set & gold_set) > 0 else 0.0


@torch.no_grad()
def compute_answer_confidence(prefix_list, llm, tokenizer, ground_truth, device):
    """Compute confidence scores for different prompt configurations"""
    prev_use_cache = llm.config.use_cache
    prev_grad_ckpt = getattr(llm, 'is_gradient_checkpointing', False)
    
    llm.config.use_cache = True
    if hasattr(llm, 'gradient_checkpointing_disable'):
        llm.gradient_checkpointing_disable()
    
    try:
        results = []
        for prefix_embeds in prefix_list:
            attn_mask = torch.ones(prefix_embeds.size()[:-1], dtype=torch.long, device=device)
            gen_out = llm.generate(
                inputs_embeds=prefix_embeds, 
                attention_mask=attn_mask,
                max_new_tokens=64, 
                do_sample=False, 
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, 
                output_scores=True, 
                use_cache=True
            )
            pred_text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
            hit = compute_hit_score(pred_text, ground_truth)
            
            if hasattr(gen_out, 'scores') and len(gen_out.scores) > 0:
                scores = torch.stack(gen_out.scores)
                probs = F.softmax(scores[:, 0, :], dim=-1)
                generated_ids = gen_out.sequences[0, prefix_embeds.size(1):]
                if len(generated_ids) > 0 and len(generated_ids) <= len(probs):
                    token_confidences = probs[range(len(generated_ids)), generated_ids]
                    mean_confidence = token_confidences.mean().item()
                else:
                    mean_confidence = 0.5
            else:
                mean_confidence = 0.5
            
            results.append((hit, mean_confidence))
            del gen_out
            torch.cuda.empty_cache()
        
        return results
    finally:
        llm.config.use_cache = prev_use_cache
        if prev_grad_ckpt and hasattr(llm, 'gradient_checkpointing_enable'):
            llm.gradient_checkpointing_enable()


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    
    p.add_argument('-d', '--dataset', type=str, required=True,
                   choices=['webqsp', 'cwq'],
                   help='Dataset to use (webqsp or cwq)')
    
    p.add_argument('--config', type=str, default=None,
                   help='Config file path (default: configs/joint/{dataset}.yaml)')
    p.add_argument('--model_name', default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                   help='HuggingFace model name')
    p.add_argument('--device', default="cuda:0")
    p.add_argument('--save_dir', type=str, default=None,
                   help='Directory to save checkpoints (default: joint_{dataset}_run)')
    
    p.add_argument('--intent_run_dir', type=str, required=True,
                   help='Path to intent selector checkpoint directory')
    p.add_argument('--g_cache', type=str, required=True,
                   help='Path to G cache file')
    p.add_argument('--cum_threshold', type=float, default=0.3,
                   help='Cumulative threshold for intent selection')
    p.add_argument('--min_k', type=int, default=1,
                   help='Minimum number of intents to select')
    p.add_argument('--max_k', type=int, default=4,
                   help='Maximum number of intents to select')
    p.add_argument('--grad_to_beta', action='store_true',
                   help='Enable gradient flow to beta parameter')
    
    p.add_argument('--retriever_ckpt', type=str, default="",
                   help='Path to pretrained retriever checkpoint')
    p.add_argument('--load_selector_from_retriever_ckpt', action='store_true',
                   help='Load selector weights from retriever checkpoint')
    p.add_argument('--load_from_joint', action='store_true',
                   help='Resume from previous joint training checkpoint')
    p.add_argument('--joint_dir', type=str, default='',
                   help='Directory containing joint training checkpoints')
    
    p.add_argument('--end_to_end', action='store_true',
                   help='Enable end-to-end training (update retriever)')
    p.add_argument('--use_adaptive_dpo', action='store_true',
                   help='Use adaptive DPO training mode')
    
    p.add_argument('--do_test', action='store_true',
                   help='Run test inference after training')
    p.add_argument('--test_split', type=str, default='test',
                   help='Dataset split to use for testing')
    p.add_argument('--test_output', type=str, default=None,
                   help='Path to save test predictions (default: results/{dataset}_predictions.jsonl)')
    p.add_argument('--limit_test', type=int, default=0,
                   help='Limit number of test samples (0 = no limit)')
    
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed (overrides config)')
    p.add_argument('--epochs', type=int, default=None,
                   help='Number of training epochs (overrides config)')
    
    return p.parse_args()


def run_epoch(args, config, dataloader, retriever, triple_proj, enhanced_prompt_module,
              llm, tokenizer, optimizer, device, mode="train", bridge=None,
              epoch=0, enable_e2e=False):

    train_cfg = config['training']
    is_dpo_stage = args.use_adaptive_dpo
    train_selector_now = (mode == "train" and (epoch >= train_cfg['train_selector_after']))
    retriever_dtype = torch.bfloat16

    if mode == "train":
        if is_dpo_stage:
            retriever.eval()
            triple_proj.eval()
            enhanced_prompt_module.eval()
            bridge.selector.wae.eval()
            llm.train()
            for p in retriever.parameters(): 
                p.requires_grad = False
            for p in triple_proj.parameters(): 
                p.requires_grad = False
            for p in enhanced_prompt_module.parameters(): 
                p.requires_grad = False
            for p in bridge.selector.wae.parameters(): 
                p.requires_grad = False
        else:
            triple_proj.train()
            enhanced_prompt_module.train()
            llm.train()
            
            if train_selector_now:
                bridge.selector.wae.train()
                for p in bridge.selector.wae.parameters():
                    p.requires_grad = True
            else:
                bridge.selector.wae.eval()
                for p in bridge.selector.wae.parameters():
                    p.requires_grad = False
            
            if enable_e2e:
                retriever.train()
                for p in retriever.parameters():
                    p.requires_grad = True
            else:
                retriever.eval()
                for p in retriever.parameters():
                    p.requires_grad = False
    else:
        retriever.eval()
        triple_proj.eval()
        enhanced_prompt_module.eval()
        llm.eval()
        bridge.selector.wae.eval()

    losses, accum = [], 0.0
    gas = train_cfg['gradient_accumulation_steps'] if mode == "train" else 1
    
    ans_lead_ids = tokenizer("Answer:\n", add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    ans_lead_embeds = to_llm(llm.get_input_embeddings()(ans_lead_ids), llm)
    
    per_intent_total = train_cfg['per_intent_total']
    tau_gumbel = config['retriever']['selection']['tau_gumbel']
    gate_boost = config['neural_prompt']['gate_boost']

    for step, sample in enumerate(tqdm(dataloader, desc=f"{mode} E{epoch}", leave=False)):
        (h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
         num_non_text_entities, relation_embs, topic_entity_one_hot,
         target_triple_probs, a_entity_id_list) = prepare_sample(device, sample)

        q_emb = q_emb.to(device=device, dtype=retriever_dtype)
        entity_embs = entity_embs.to(device=device, dtype=retriever_dtype)
        relation_embs = relation_embs.to(device=device, dtype=retriever_dtype)
        if torch.is_floating_point(topic_entity_one_hot):
            topic_entity_one_hot = topic_entity_one_hot.to(device=device, dtype=retriever_dtype)

        intent_embs, sel_idx, _ = bridge.select(q_emb)
        if intent_embs is not None:
            intent_embs = intent_embs.to(device=device, dtype=retriever_dtype)

        with torch.set_grad_enabled(mode == "train" and enable_e2e and not is_dpo_stage):
            logits_TI = retriever(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot,
                external_intents=intent_embs
            )

        T, I = logits_TI.shape
        k_each = get_per_intent_quota(I, per_intent_total)
        
        if mode == "train" and not is_dpo_stage:
            intent_masks = gumbel_topk_batch(logits_TI, k=k_each, tau_gumbel=tau_gumbel)
        else:
            intent_masks = per_intent_topk_no_noise(logits_TI, k_each)
        intent_masks = intent_masks.to(dtype=logits_TI.dtype)

        gate = intent_masks.sum(dim=1) / max(1, I)
        sel_indices = torch.nonzero(gate > 0, as_tuple=True)[0]
        if sel_indices.numel() == 0:
            sel_indices = torch.topk(gate, k=1, largest=True).indices

        overlap_cnt = intent_masks[sel_indices].sum(dim=1)
        entity_list = sample['text_entity_list'] + sample['non_text_entity_list']

        triple_embeds = compute_triple_embeddings(
            h_id_tensor, r_id_tensor, t_id_tensor,
            entity_embs, num_non_text_entities,
            relation_embs, retriever, triple_proj, device
        )
        
        scale = (1.0 + gate_boost * gate).to(triple_embeds.dtype).unsqueeze(1)
        triple_embeds = triple_embeds * scale

        
        fact_texts_weighted = create_fact_texts_with_markers(entity_list, sample, sel_indices, overlap_cnt)
        fact_texts_plain = create_fact_texts_without_markers(entity_list, sample, sel_indices)

        
        neural_prompt_weighted = enhanced_prompt_module(
            triple_embeds, overlap_cnt, fact_texts_weighted, sel_indices,
            tokenizer, llm, device, gate_weights=gate, disable_emphasis=False
        )
        neural_prompt_plain = enhanced_prompt_module(
            triple_embeds, overlap_cnt, fact_texts_plain, sel_indices,
            tokenizer, llm, device, gate_weights=None, disable_emphasis=True
        )

        
        qtext = sample['question'].strip()
        if not qtext.endswith('?'):
            qtext += '?'
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question:\n{qtext}\n{cot_prompt}"},
        ]
        chat_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(device)
        chat_embeds = to_llm(llm.get_input_embeddings()(chat_ids), llm)

        
        kg_wrapper = config['neural_prompt']['kg_wrapper']
        kg_start_ids = tokenizer(kg_wrapper['start'], add_special_tokens=False, return_tensors='pt').input_ids.to(device)
        kg_end_ids = tokenizer(kg_wrapper['end'], add_special_tokens=False, return_tensors='pt').input_ids.to(device)
        kg_start_embeds = to_llm(llm.get_input_embeddings()(kg_start_ids), llm)
        kg_end_embeds = to_llm(llm.get_input_embeddings()(kg_end_ids), llm)

        
        inputs_prefix_weighted = torch.cat([
            chat_embeds, kg_start_embeds, neural_prompt_weighted, 
            kg_end_embeds, ans_lead_embeds
        ], dim=1)
        inputs_prefix_plain = torch.cat([
            chat_embeds, kg_start_embeds, neural_prompt_plain, 
            kg_end_embeds, ans_lead_embeds
        ], dim=1)

        
        answers = [a for a in sample.get('a_entity', []) if isinstance(a, str) and a.strip()]
        if len(answers) == 0:
            answers = ["not available"]

        
        chosen_prefix = None
        rejected_prefix = None
        if mode == "train" and is_dpo_stage:
            results = compute_answer_confidence(
                [inputs_prefix_weighted, inputs_prefix_plain], 
                llm, tokenizer, answers, device
            )
            (hit_w, conf_w), (hit_p, conf_p) = results
            hit_diff = hit_w - hit_p
            conf_diff = conf_w - conf_p
            
            dpo_threshold = train_cfg['dpo_threshold']
            
            if hit_diff > 0:
                chosen_prefix = inputs_prefix_weighted
                rejected_prefix = inputs_prefix_plain
            elif hit_diff < 0:
                chosen_prefix = inputs_prefix_plain
                rejected_prefix = inputs_prefix_weighted
            elif hit_w == 1.0 and hit_p == 1.0:
                if conf_diff > dpo_threshold:
                    chosen_prefix = inputs_prefix_weighted
                    rejected_prefix = inputs_prefix_plain
                elif conf_diff < -dpo_threshold:
                    chosen_prefix = inputs_prefix_plain
                    rejected_prefix = inputs_prefix_weighted

        if mode == "train":
            with amp.autocast(device_type='cuda', dtype=torch.bfloat16, cache_enabled=True):
                answers = answers[:30]
                gold_text = "\n".join([f"ans: {a}" for a in answers]) + tokenizer.eos_token
                gold_ids = tokenizer(
                    gold_text, add_special_tokens=False, 
                    truncation=True, max_length=64, return_tensors='pt'
                ).input_ids.to(device)
                gold_embeds = to_llm(llm.get_input_embeddings()(gold_ids), llm)

                inputs_full_weighted = torch.cat([inputs_prefix_weighted, gold_embeds], dim=1)
                attn_mask = torch.ones(inputs_full_weighted.size()[:-1], dtype=torch.long, device=device)
                labels = torch.full((1, inputs_full_weighted.size(1)), -100, device=device)
                start_idx = inputs_prefix_weighted.size(1)
                end_idx = start_idx + gold_ids.size(1)
                labels[0, start_idx:end_idx] = gold_ids[0]
                
                out = llm(inputs_embeds=inputs_full_weighted, attention_mask=attn_mask, labels=labels)
                lm_loss = out.loss

                dpo_loss = torch.tensor(0.0, device=device)
                if is_dpo_stage and chosen_prefix is not None:
                    def get_logprob(prefix, gold_ids_local):
                        ge = llm.get_input_embeddings()(gold_ids_local)
                        full = torch.cat([prefix, ge], dim=1)
                        am = torch.ones(full.size()[:-1], dtype=torch.long, device=device)
                        logits = llm(inputs_embeds=full, attention_mask=am).logits
                        logits_gold = logits[:, -gold_ids_local.size(1):, :]
                        log_probs = F.log_softmax(logits_gold, dim=-1)
                        token_lps = log_probs.gather(-1, gold_ids_local.unsqueeze(-1)).squeeze(-1)
                        return token_lps.mean()
                    
                    logp_chosen = get_logprob(chosen_prefix, gold_ids)
                    with torch.no_grad():
                        logp_rejected = get_logprob(rejected_prefix, gold_ids)
                    
                    dpo_beta = train_cfg['dpo_beta']
                    dpo_margin = train_cfg['dpo_margin']
                    log_ratio = logp_chosen - logp_rejected
                    dpo_loss = -F.logsigmoid(dpo_beta * (log_ratio - dpo_margin))

                if enable_e2e and not is_dpo_stage:
                    loss_cfg = config['retriever']['loss']
                    target = target_triple_probs.to(device=logits_TI.device, dtype=logits_TI.dtype)
                    
                    bce_loss = focal_loss_with_pos_weight(
                        logits_TI, target, 
                        alpha=loss_cfg['focal_alpha'],
                        gamma=loss_cfg['focal_gamma']
                    )
                    rank_loss_v = ranking_loss_vectorized(
                        logits_TI, target, 
                        margin=loss_cfg['ranking_margin']
                    )
                    decorr_loss = topk_decorrelation_loss(
                        logits_TI, 
                        k=loss_cfg['decorrelation_k']
                    )
                    
                    positive_mask = (target > 0.5)
                    pattern_loss = pattern_diversity_loss(
                        logits_TI, 
                        k=loss_cfg['pattern_k'], 
                        temperature=loss_cfg['pattern_temp'],
                        positive_mask=positive_mask, 
                        cap=None, 
                        auto_transpose=True
                    )
                    
                    ret_loss = bce_loss + train_cfg['rank_weight'] * rank_loss_v
                    div_loss = (train_cfg['diversity_weight'] * decorr_loss + 
                            train_cfg['pattern_div_weight'] * pattern_loss)
                else:
                    ret_loss = torch.tensor(0.0, device=device)
                    div_loss = torch.tensor(0.0, device=device)

                if is_dpo_stage:
                    total = (lm_loss + train_cfg['dpo_weight'] * dpo_loss) / gas
                else:
                    rho = train_cfg['rho']
                    total = (rho * ret_loss + (1 - rho) * lm_loss + 
                            div_loss + train_cfg['dpo_weight'] * dpo_loss) / gas
            total.backward()
            accum += float(total.item())
            
            if (step + 1) % gas == 0:
                if is_dpo_stage:
                    all_params = [p for p in llm.parameters() if p.requires_grad]
                else:
                    all_params = (
                        list(retriever.parameters()) + 
                        list(triple_proj.parameters()) +
                        list(enhanced_prompt_module.parameters()) +
                        list(bridge.selector.wae.parameters()) +
                        [p for p in llm.parameters() if p.requires_grad]
                    )
                
                max_grad_norm = config['optimizer']['max_grad_norm']
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(accum)
                accum = 0.0
        
        else:
            with torch.no_grad(), amp.autocast(device_type='cuda', dtype=torch.bfloat16, cache_enabled=True):
                answers = answers[:30]
                gold_text = "\n".join([f"ans: {a}" for a in answers]) + tokenizer.eos_token
                gold_ids = tokenizer(
                    gold_text, add_special_tokens=False, 
                    truncation=True, max_length=64, return_tensors='pt'
                ).input_ids.to(device)
                gold_embeds = to_llm(llm.get_input_embeddings()(gold_ids), llm)
                
                inputs_full_weighted = torch.cat([inputs_prefix_weighted, gold_embeds], dim=1)
                attn_mask_w = torch.ones((1, inputs_full_weighted.size(1)), dtype=torch.long, device=inputs_full_weighted.device)
                labels_w = torch.full((1, inputs_full_weighted.size(1)), -100, device=device)
                labels_w[0, inputs_prefix_weighted.size(1): inputs_prefix_weighted.size(1) + gold_ids.size(1)] = gold_ids[0]
                
                out = llm(inputs_embeds=inputs_full_weighted, attention_mask=attn_mask_w, labels=labels_w)
                total = out.loss
                losses.append(float(total.item()))

        del neural_prompt_weighted, neural_prompt_plain, triple_embeds
        if (step + 1) % (gas * 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if mode == "train" and accum > 0:
        if is_dpo_stage:
            all_params = [p for p in llm.parameters() if p.requires_grad]
        else:
            all_params = (
                list(retriever.parameters()) + 
                list(triple_proj.parameters()) +
                list(enhanced_prompt_module.parameters()) +
                list(bridge.selector.wae.parameters()) +
                [p for p in llm.parameters() if p.requires_grad]
            )
        
        max_grad_norm = config['optimizer']['max_grad_norm']
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(accum)

    avg = float(sum(losses) / max(1, len(losses)))
    return avg
def main():
    args = parse_args()
    
    if args.config is None:
        config_file = f'configs/joint/{args.dataset}.yaml'
    else:
        config_file = args.config
    
    if args.save_dir is None:
        args.save_dir = f"joint_{args.dataset}_run"
    
    if args.test_output is None:
        args.test_output = f'results/{args.dataset}_predictions.jsonl'
    
    config = load_yaml(config_file)
    
    seed = args.seed if args.seed is not None else config['env']['seed']
    set_seed(seed)
    
    device = torch.device(args.device)
    torch.set_num_threads(config['env']['num_threads'])
    
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    best_dir = os.path.join(args.save_dir, "best")
    pathlib.Path(best_dir).mkdir(parents=True, exist_ok=True)

    retriever_cfg = config['retriever']
    emb_size = retriever_cfg['emb_size']
    topic_pe = retriever_cfg['topic_pe']
    DDE_kwargs = retriever_cfg['DDE_kwargs']
    num_intents = retriever_cfg['num_intents']

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Config: {config_file}")
    print(f"Mode: {'DPO' if args.use_adaptive_dpo else 'Joint'} | E2E: {args.end_to_end}")
    print(f"Output: {args.save_dir}")
    print(f"{'='*60}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=os.getenv("HF_TOKEN"))
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    llm_cfg = config['llm']
    quant_cfg = llm_cfg['quantization']
    bnb = BitsAndBytesConfig(
        load_in_4bit=quant_cfg['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg['compute_dtype']),
        bnb_4bit_quant_type=quant_cfg['quant_type'],
        bnb_4bit_use_double_quant=quant_cfg['use_double_quant']
    )
    
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map={"": args.device},
        quantization_config=bnb, torch_dtype=torch.bfloat16,
        use_auth_token=os.getenv("HF_TOKEN")
    )
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    llm = prepare_model_for_kbit_training(llm)

    if not args.load_from_joint:
        lora_cfg = llm_cfg['lora']
        lora = LoraConfig(
            r=lora_cfg['r'],
            lora_alpha=lora_cfg['lora_alpha'],
            target_modules=lora_cfg['target_modules'],
            lora_dropout=lora_cfg['lora_dropout'],
            bias=lora_cfg['bias'],
            task_type="CAUSAL_LM"
        )
        llm = get_peft_model(llm, lora)

    retriever = Retriever(
        emb_size=emb_size,
        topic_pe=topic_pe,
        DDE_kwargs=DDE_kwargs,
        num_intents=num_intents
    ).to(device, dtype=torch.bfloat16)

    proj_cfg = config['triple_projection']
    d_mid = llm.config.hidden_size // proj_cfg['hidden_divisor']
    triple_proj = nn.Sequential(
        nn.Linear(emb_size * 3, d_mid), 
        nn.SiLU(),
        nn.Linear(d_mid, llm.config.hidden_size),
        nn.LayerNorm(llm.config.hidden_size)
    ).to(device, dtype=torch.bfloat16)

    prompt_cfg = config['neural_prompt']
    enhanced_prompt = Consensus_aware_Prompt(
        llm.config.hidden_size, 
        llm.config.hidden_size, 
        gate_boost=prompt_cfg['gate_boost']
    ).to(device, dtype=torch.bfloat16)

    if args.load_from_joint:
        if not args.joint_dir: 
            raise ValueError("--load_from_joint requires --joint_dir")
        
        retriever_path = os.path.join(args.joint_dir, "best/retriever_best.pth")
        if os.path.exists(retriever_path):
            ckpt = torch.load(retriever_path, map_location="cpu")
            retriever.load_state_dict(ckpt["retriever_state_dict"], strict=False)
            saved_intent_run_dir = ckpt.get('intent_run_dir', '').strip()
            saved_g_cache = ckpt.get('g_cache', '').strip()
            if not saved_intent_run_dir and args.intent_run_dir: 
                saved_intent_run_dir = args.intent_run_dir
            if not saved_g_cache and args.g_cache: 
                saved_g_cache = args.g_cache
        else:
            raise FileNotFoundError(f"Retriever checkpoint not found: {retriever_path}")

        for pth, module, key in [
            (os.path.join(args.joint_dir, "best/triple_proj_best.pth"), triple_proj, "triple_proj_state_dict"),
            (os.path.join(args.joint_dir, "best/enhanced_prompt_best.pth"), enhanced_prompt, "enhanced_prompt_state_dict"),
        ]:
            sd = torch.load(pth, map_location="cpu")
            module.load_state_dict(sd[key], strict=True)

        lora_path = os.path.join(args.joint_dir, "best/lora_best")
        llm = PeftModel.from_pretrained(llm, lora_path)

        if not saved_intent_run_dir or not saved_g_cache:
            raise ValueError("intent_run_dir/g_cache not found in checkpoint")
        run_dir_path = Path(saved_intent_run_dir).expanduser().resolve()
        if not run_dir_path.is_dir(): 
            raise FileNotFoundError(f"intent_run_dir not found: {run_dir_path}")

        bridge = IntentSelectorBridge(
            run_dir=str(run_dir_path), g_cache=saved_g_cache, device=args.device,
            select_cfg=SelectCfg(
                cum_threshold=args.cum_threshold, 
                min_k=args.min_k,
                max_k=args.max_k, 
                grad_to_beta=args.grad_to_beta
            )
        )
        selector_ckpt = os.path.join(args.joint_dir, "best/selector_best.pth")
        if os.path.exists(selector_ckpt):
            sd = torch.load(selector_ckpt, map_location="cpu")
            bridge.selector.wae.load_state_dict(sd["selector_state_dict"], strict=False)

    elif args.load_selector_from_retriever_ckpt:
        if not args.retriever_ckpt:
            raise ValueError("--load_selector_from_retriever_ckpt requires --retriever_ckpt")
        
        ckpt = torch.load(args.retriever_ckpt, map_location="cpu")
        model_sd = ckpt.get('model_state_dict', ckpt)
        retriever.load_state_dict(model_sd, strict=False)
        
        saved_intent_run_dir = (ckpt.get('intent_run_dir') or "").strip()
        saved_g_cache = (ckpt.get('g_cache') or "").strip()
        if args.intent_run_dir: 
            saved_intent_run_dir = args.intent_run_dir
        if args.g_cache: 
            saved_g_cache = args.g_cache
        
        run_dir_path = Path(saved_intent_run_dir).expanduser().resolve()
        if not run_dir_path.is_dir(): 
            raise FileNotFoundError(f"intent_run_dir not found: {run_dir_path}")

        bridge = IntentSelectorBridge(
            run_dir=str(run_dir_path), g_cache=saved_g_cache, device=args.device,
            select_cfg=SelectCfg(
                cum_threshold=args.cum_threshold, 
                min_k=args.min_k,
                max_k=args.max_k, 
                grad_to_beta=args.grad_to_beta
            )
        )
        if 'selector_state_dict' in ckpt:
            bridge.selector.wae.load_state_dict(ckpt['selector_state_dict'], strict=False)
    
    else:
        if args.retriever_ckpt:
            sd = torch.load(args.retriever_ckpt, map_location="cpu")
            model_sd = sd.get('model_state_dict', sd)
            retriever.load_state_dict(model_sd, strict=False)
        
        if not args.intent_run_dir or not args.g_cache:
            raise ValueError("Requires --intent_run_dir and --g_cache")
        
        bridge = IntentSelectorBridge(
            run_dir=args.intent_run_dir, g_cache=args.g_cache, device=args.device,
            select_cfg=SelectCfg(
                cum_threshold=args.cum_threshold, 
                min_k=args.min_k,
                max_k=args.max_k, 
                grad_to_beta=args.grad_to_beta
            )
        )

    bridge.selector.wae = bridge.selector.wae.to(dtype=torch.bfloat16)
    bridge.G = bridge.G.to(dtype=torch.bfloat16)
    if hasattr(bridge, 'beta') and bridge.beta is not None: 
        bridge.beta = bridge.beta.to(dtype=torch.bfloat16)
    if hasattr(bridge, 'T') and bridge.T is not None: 
        bridge.T = bridge.T.to(dtype=torch.bfloat16)

    opt_cfg = config['optimizer']
    lr_cfg = opt_cfg['learning_rates']
    
    if args.use_adaptive_dpo:
        param_groups = [
            {"params": [p for p in llm.parameters() if p.requires_grad], 
             "lr": lr_cfg['dpo_only']}
        ]
    else:
        param_groups = [
            {"params": retriever.parameters(), "lr": lr_cfg['retriever']},
            {"params": triple_proj.parameters(), "lr": lr_cfg['triple_proj']},
            {"params": enhanced_prompt.parameters(), "lr": lr_cfg['enhanced_prompt']},
            {"params": bridge.selector.wae.parameters(), "lr": lr_cfg['selector']},
            {"params": [p for p in llm.parameters() if p.requires_grad], "lr": lr_cfg['llm_lora']},
        ]
    
    optimizer = AdamW(param_groups, weight_decay=opt_cfg['weight_decay'])

    train_ds = RetrieverDataset(config=config, split='train')
    val_ds = RetrieverDataset(config=config, split='val')
    train_loader = DataLoader(
        train_ds[:20], batch_size=1, shuffle=True,
        collate_fn=collate_joint_retriever_llm,
        pin_memory=True, num_workers=2, 
        prefetch_factor=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds[:1], batch_size=1, shuffle=False,
        collate_fn=collate_joint_retriever_llm, 
        pin_memory=True
    )

    train_cfg = config['training']
    epochs = args.epochs if args.epochs is not None else train_cfg['epochs']
    patience_limit = train_cfg['patience']

    print(f"\n{'='*60}")
    print(f"Mode: {'DPO' if args.use_adaptive_dpo else 'Joint'} | E2E: {args.end_to_end}")
    print(f"Config: {config_file}")  
    print(f"Output: {args.save_dir}")
    print(f"{'='*60}\n")

    best_val, best_epoch, patience = float("inf"), 0, 0

    for ep in range(epochs):
        enable_e2e = args.end_to_end and (ep + 1) > train_cfg['freeze_retriever_epochs']

        tr_loss = run_epoch(
            args, config, train_loader, retriever, triple_proj, enhanced_prompt,
            llm, tok, optimizer, device, mode="train",
            bridge=bridge, epoch=ep+1, enable_e2e=enable_e2e
        )
        va_loss = run_epoch(
            args, config, val_loader, retriever, triple_proj, enhanced_prompt,
            llm, tok, None, device, mode="val",
            bridge=bridge, epoch=ep+1, enable_e2e=False
        )

        print(f"[Epoch {ep+1:3d}] Train: {tr_loss:.4f} | Val: {va_loss:.4f}")

        if va_loss < best_val:
            best_val, best_epoch, patience = va_loss, ep+1, 0
            Path(best_dir).mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "epoch": best_epoch, 
                "retriever_state_dict": retriever.state_dict(),
                "best_val_loss": best_val, 
                "config": config,
                "intent_run_dir": str(args.intent_run_dir), 
                "g_cache": args.g_cache,
            }, os.path.join(best_dir, "retriever_best.pth"))
            
            torch.save({
                "epoch": best_epoch, 
                "triple_proj_state_dict": triple_proj.state_dict()
            }, os.path.join(best_dir, "triple_proj_best.pth"))
            
            torch.save({
                "epoch": best_epoch, 
                "enhanced_prompt_state_dict": enhanced_prompt.state_dict()
            }, os.path.join(best_dir, "enhanced_prompt_best.pth"))
            
            torch.save({
                "epoch": best_epoch, 
                "selector_state_dict": bridge.selector.wae.state_dict()
            }, os.path.join(best_dir, "selector_best.pth"))
            
            llm.save_pretrained(os.path.join(best_dir, "lora_best"))
            print(f"         ✓ Best model saved (Val: {best_val:.4f})")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"[Early Stop] Patience {patience_limit} reached")
                break

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Training completed! Best Val: {best_val:.4f} @ Epoch {best_epoch}")
    print(f"Model saved: {best_dir}")
    print(f"{'='*60}\n")

    if args.do_test:
        print("[Test] Loading best weights...")
        
        ckpt = torch.load(os.path.join(best_dir, "retriever_best.pth"), map_location="cpu")
        retriever.load_state_dict(ckpt["retriever_state_dict"], strict=False)
        retriever.eval()

        sd = torch.load(os.path.join(best_dir, "triple_proj_best.pth"), map_location="cpu")
        triple_proj.load_state_dict(sd["triple_proj_state_dict"], strict=True)
        triple_proj.eval()

        sd = torch.load(os.path.join(best_dir, "enhanced_prompt_best.pth"), map_location="cpu")
        enhanced_prompt.load_state_dict(sd["enhanced_prompt_state_dict"], strict=False)
        enhanced_prompt.eval()

        selector_ckpt = os.path.join(best_dir, "selector_best.pth")
        if os.path.exists(selector_ckpt):
            sd = torch.load(selector_ckpt, map_location="cpu")
            bridge.selector.wae.load_state_dict(sd["selector_state_dict"], strict=False)
        bridge.selector.wae.eval()

        base_llm = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map={"": args.device},
            quantization_config=bnb, torch_dtype=torch.bfloat16,
            use_auth_token=os.getenv("HF_TOKEN")
        )
        llm = PeftModel.from_pretrained(base_llm, os.path.join(best_dir, "lora_best"))
        llm.eval()

        test_ds = RetrieverDataset(config=config, split=args.test_split, skip_no_path=False)
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            collate_fn=collate_joint_retriever_llm, 
            pin_memory=True
        )

        run_test_inference(args, config, test_loader, retriever, triple_proj, 
                          enhanced_prompt, llm, tok, device, bridge)
        
        try:
            from src.utils.evaluation import eval_results_corrected_compat, eval_results_hit_any_compat
            
            print("\n[EVAL] Running corrected metrics...")
            (hit1, f1, prec, rec, em, tw, mi_f1, mi_p, mi_r,
             total_cnt, no_ans_cnt, no_ans_ratio, hal, stats) = eval_results_corrected_compat(
                predict_file=args.test_output, cal_f1=True, subset=False, split=None,
                eval_hops=-1, dataset_name=None, scored_triples_path=None
            )
            print(f"[CORRECTED] Hit@1={hit1:.2f}, MacroF1={f1:.2f}, MacroP={prec:.2f}, MacroR={rec:.2f}, "
                  f"EM={em:.2f}, TW={tw:.2f}, MicroF1={mi_f1*100:.2f}, MicroP={mi_p*100:.2f}, MicroR={mi_r*100:.2f}, "
                  f"Total={total_cnt}, No-Ans={no_ans_cnt} ({no_ans_ratio*100:.2f}%), Hal={hal:.2f}")
            
            print("\n[EVAL] Running hit-any metrics...")
            hit_any, f1_any, p_any, r_any = eval_results_hit_any_compat(
                predict_file=args.test_output, cal_f1=True, topk=-1, subset=False,
                eval_hops=-1, dataset_name=None, scored_triples_path=None
            )
            print(f"[HIT-ANY] Hit={hit_any:.2f}, F1={f1_any:.2f}, P={p_any:.2f}, R={r_any:.2f}")
        except Exception as e:
            print(f"[Test] Evaluation skipped: {e}")


if __name__ == "__main__":
    main()