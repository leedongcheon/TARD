import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_sample(device, sample):
    h_id_tensor = torch.as_tensor(sample['h_id_list']).to(device=device, dtype=torch.long, non_blocking=True)
    r_id_tensor = torch.as_tensor(sample['r_id_list']).to(device=device, dtype=torch.long, non_blocking=True)
    t_id_tensor = torch.as_tensor(sample['t_id_list']).to(device=device, dtype=torch.long, non_blocking=True)

    q_emb              = torch.as_tensor(sample['q_emb']).to(device=device, dtype=torch.float32, non_blocking=True)
    entity_embs        = torch.as_tensor(sample['entity_embs']).to(device=device, dtype=torch.float32, non_blocking=True)
    relation_embs      = torch.as_tensor(sample['relation_embs']).to(device=device, dtype=torch.float32, non_blocking=True)
    topic_entity_one_hot = torch.as_tensor(sample['topic_entity_one_hot']).to(device=device, dtype=torch.float32, non_blocking=True)

    num_non_text_entities = len(sample['non_text_entity_list'])

    target_triple_probs = torch.as_tensor(sample['target_triple_probs']).to(device=device, dtype=torch.float32, non_blocking=True)
    a_entity_id_list = sample['a_entity_id_list']  

    return (h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            target_triple_probs, a_entity_id_list)