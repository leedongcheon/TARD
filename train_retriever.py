# train_retriever.py
"""
Retriever pre-training with Multi-Intent and Selector
"""
import os
import time
import numpy as np
import torch
from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_joint_retriever_llm
from src.model.Intent_aware_Retriever import Retriever
from src.model.intent_selectors import IntentSelectorBridge, SelectCfg
from src.setup import set_seed, prepare_sample

from src.losses import (
    focal_loss_with_pos_weight,
    ranking_loss_vectorized as ranking_loss,
    pattern_diversity_loss,
    topk_decorrelation_loss
)


@torch.no_grad()
def eval_epoch(config, device, data_loader, model, bridge):
    """Evaluation with intent selector"""
    model.eval()
    bridge.selector.wae.eval()
    
    recall_100_list = []
    retriever_dtype = torch.bfloat16
    
    for sample in tqdm(data_loader, desc="Evaluating", leave=False):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        q_emb = q_emb.to(device=device, dtype=retriever_dtype)
        entity_embs = entity_embs.to(device=device, dtype=retriever_dtype)
        relation_embs = relation_embs.to(device=device, dtype=retriever_dtype)
        if torch.is_floating_point(topic_entity_one_hot):
            topic_entity_one_hot = topic_entity_one_hot.to(device=device, dtype=retriever_dtype)

        intent_embs, _, _ = bridge.select(q_emb)
        if intent_embs is not None:
            intent_embs = intent_embs.to(device=device, dtype=retriever_dtype)
        
        all_intent_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            external_intents=intent_embs)
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1).cpu()
        if len(target_triple_ids) == 0:
            continue

        # Average across intents
        avg_logits = all_intent_logits.mean(dim=-1).cpu()
        sorted_ids = torch.argsort(avg_logits, descending=True)
        ranks = torch.empty_like(sorted_ids)
        ranks[sorted_ids] = torch.arange(len(ranks))
        
        recall_100 = (ranks[target_triple_ids] < 100).sum().item() / len(target_triple_ids)
        recall_100_list.append(recall_100)
    
    return np.mean(recall_100_list) if recall_100_list else 0.0

# train_retriever.py - train_epoch 함수 수정 부분만

def train_epoch(config, device, train_loader, model, bridge, optimizer, 
                epoch, args):
    """Training with intent selector"""
    model.train()
    
    train_selector_now = (epoch >= args.train_selector_after)
    if train_selector_now:
        bridge.selector.wae.train()
        for p in bridge.selector.wae.parameters():
            p.requires_grad = True
    else:
        bridge.selector.wae.eval()
        for p in bridge.selector.wae.parameters():
            p.requires_grad = False
    
    epoch_loss = 0
    epoch_task_loss = 0
    epoch_decorr_loss = 0
    epoch_pattern_loss = 0
    
    retriever_dtype = torch.bfloat16
    
    for sample in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        q_emb = q_emb.to(device=device, dtype=retriever_dtype)
        entity_embs = entity_embs.to(device=device, dtype=retriever_dtype)
        relation_embs = relation_embs.to(device=device, dtype=retriever_dtype)
        if torch.is_floating_point(topic_entity_one_hot):
            topic_entity_one_hot = topic_entity_one_hot.to(device=device, dtype=retriever_dtype)

        intent_embs, _, _ = bridge.select(q_emb)
        if intent_embs is not None:
            intent_embs = intent_embs.to(device=device, dtype=retriever_dtype)
        
        all_intent_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            external_intents=intent_embs)
        
        actual_num_intents = all_intent_logits.shape[1]
        target_triple_probs = target_triple_probs.to(device)
        target_expanded = target_triple_probs.unsqueeze(-1).expand(-1, actual_num_intents)
        
        # ✅ Task losses - READ FROM ARGS (not hardcoded)
        # Note: For retriever training, we use simple default values since
        # loss hyperparameters are not in retriever config
        bce_loss = focal_loss_with_pos_weight(
            all_intent_logits, target_expanded, 
            alpha=0.75,  # Default for retriever training
            gamma=2.0
        )
        rank_loss = ranking_loss(
            all_intent_logits, target_triple_probs, 
            margin=2.0  # Default for retriever training
        )
        task_loss = bce_loss + 0.5 * rank_loss
        
        # Regularization losses
        positive_mask = (target_triple_probs > 0.5)
        decorr_loss = topk_decorrelation_loss(
            all_intent_logits, k=args.decorr_k, mode=args.decorr_mode,
            temperature=args.decorr_temp, use_softmax=False, 
            positive_mask=positive_mask
        )
        
        pattern_loss = pattern_diversity_loss(
            all_intent_logits, k=args.pattern_k, 
            temperature=args.pattern_temp,
            positive_mask=positive_mask, cap=None, 
            auto_transpose=True
        )
        
        total_loss = (task_loss + 
                     args.decorr_weight * decorr_loss + 
                     args.pattern_div_weight * pattern_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        
        all_params = list(model.parameters()) + list(bridge.selector.wae.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 0.5)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_task_loss += task_loss.item()
        epoch_decorr_loss += decorr_loss.item()
        epoch_pattern_loss += pattern_loss.item()
    
    n = len(train_loader)
    return {
        'loss': epoch_loss / n,
        'task': epoch_task_loss / n,
        'decorr': epoch_decorr_loss / n,
        'pattern': epoch_pattern_loss / n,
    }


def main(args):
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    exp_name = f"{config['train']['save_prefix']}_intent_{ts}"
    os.makedirs(exp_name, exist_ok=True)

    train_set = RetrieverDataset(config=config, split='train')
    val_set = RetrieverDataset(config=config, split='val')

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, 
        collate_fn=collate_joint_retriever_llm
    )
    val_loader = DataLoader(
        val_set, batch_size=1, 
        collate_fn=collate_joint_retriever_llm
    )
    
    emb_size = train_set[0]['q_emb'].shape[-1]
    
    # Multi-Intent Retriever - using config values
    num_intents = config['retriever']['num_intents']
    model = Retriever(
        emb_size=emb_size,
        topic_pe=config['retriever']['topic_pe'],
        DDE_kwargs=config['retriever']['DDE_kwargs'],
        num_intents=num_intents
    ).to(device, dtype=torch.bfloat16)
    
    if args.retriever_ckpt:
        sd = torch.load(args.retriever_ckpt, map_location="cpu")
        model.load_state_dict(sd.get('model_state_dict', sd), strict=False)
        print(f"Loaded retriever from {args.retriever_ckpt}")
    
    # Intent Selector Bridge
    bridge = IntentSelectorBridge(
        run_dir=args.intent_run_dir,
        g_cache=args.g_cache,
        device=device,
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
    
    # Optimizer - using config values
    optimizer = Adam([
        {"params": model.parameters(), "lr": config['optimizer']['lr']},
        {"params": bridge.selector.wae.parameters(), "lr": config['optimizer']['selector_lr']},
    ], weight_decay=config['optimizer']['weight_decay'])

    print(f"\n{'='*60}")
    print(f"Training: {args.dataset} | Intents: {num_intents}")
    print(f"Output: {exp_name}")
    print(f"{'='*60}\n")

    best_recall = 0.0
    patience_counter = 0

    for epoch in range(config['train']['num_epochs']):
        # Training
        train_log = train_epoch(
            config, device, train_loader, model, bridge, optimizer, 
            epoch+1, args)
        
        # Validation
        val_recall = eval_epoch(config, device, val_loader, model, bridge)
        
        # Logging
        print(f"[Epoch {epoch+1:3d}] "
              f"Train Loss: {train_log['loss']:.4f} | "
              f"Val Recall@100: {val_recall:.4f}")
        
        # Save best model
        if val_recall > best_recall:
            best_recall = val_recall
            patience_counter = 0
            torch.save({
                'config': config,
                'model_state_dict': model.state_dict(),
                'selector_state_dict': bridge.selector.wae.state_dict(),
                'args': vars(args),
                'intent_run_dir': args.intent_run_dir,
                'g_cache': args.g_cache,
            }, os.path.join(exp_name, 'best.pth'))
            print(f"         ✓ Best model saved (Recall@100: {best_recall:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config['train']['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best Recall@100: {best_recall:.4f}")
    print(f"Model saved: {exp_name}/best.pth")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    
    # Required
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'])
    parser.add_argument('--intent_run_dir', type=str, required=True,
                        help='Path to Step 1 output (beta.npy, ckpt.pt)')
    parser.add_argument('--g_cache', type=str, required=True,
                        help='Path to G cache')
    
    # Intent configuration
    parser.add_argument('--cum_threshold', type=float, default=0.4)
    parser.add_argument('--min_k', type=int, default=1)
    parser.add_argument('--max_k', type=int, default=4)
    
    # Training control
    parser.add_argument('--train_selector_after', type=int, default=0,
                        help='Start training selector after this epoch')
    parser.add_argument('--grad_to_beta', action='store_true',
                        help='Enable gradient to beta parameter')
    parser.add_argument('--retriever_ckpt', type=str, default='',
                        help='Path to pretrained retriever checkpoint')
    
    # Loss weights
    parser.add_argument('--decorr_weight', type=float, default=0.1,
                        help='Weight for decorrelation loss')
    parser.add_argument('--pattern_div_weight', type=float, default=0.05,
                        help='Weight for pattern diversity loss')
    
    # Loss settings
    parser.add_argument('--decorr_k', type=int, default=100,
                        help='Top-k for decorrelation loss')
    parser.add_argument('--decorr_mode', type=str, default='union',
                        choices=['max', 'union'],
                        help='Mode for decorrelation loss')
    parser.add_argument('--decorr_temp', type=float, default=1.0,
                        help='Temperature for decorrelation loss')
    parser.add_argument('--pattern_k', type=int, default=100,
                        help='Top-k for pattern diversity loss')
    parser.add_argument('--pattern_temp', type=float, default=0.1,
                        help='Temperature for pattern diversity loss')
    
    args = parser.parse_args()
    main(args)