


# Overview
<img width="2050" height="1016" alt="Image" src="https://github.com/user-attachments/assets/8d3870e2-e67f-4b74-9835-b83d4a5bd594" />



## Datasets

We provide two benchmark datasets for multi-hop knowledge graph question answering (KGQA).

- `webqsp`
- `cwq`

## 1-1: Entity and Relation Embedding Pre-Computation

We pre-compute question, entity, and relation embeddings for all samples to prepare them for retriever training

### Installation

We use `gte-large-en-v1.5` for text encoder, hence the environment name.

```bash
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### Inference (Embedding Computation)

```bash
python emb.py -d D
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).


## 2-1: Intent Selector Training

We train the **Intent Selector** to learn topic-level intent representations based on pre-computed question, entity, and relation embeddings.


### Installation

We next switch to the TARD environment for training the Intent Selector.

```bash
conda create -n tard python=3.10 -y
conda activate tard
pip install -r requirements/retriever.txt

```

Train the Intent Selector to learn topic-level latent intents from the pre-computed embeddings:

### Training
cd Intent_selector

```bash
python train.py \
  --train_pkl ../data_files/<dataset>/processed/train.pkl \
  --train_pth ../data_files/<dataset>/emb/gte-large-en-v1.5/train.pth \
  --val_pkl   ../data_files/<dataset>/processed/val.pkl \
  --val_pth   ../data_files/<dataset>/emb/gte-large-en-v1.5/val.pth \
  --out       ../artifacts/intent_selector
```

## 2-2: Retriever Pre-Training

We pre-train the Multi-Intent Retriever jointly with the trained Intent Selector.
This step aligns the retriever with the intent-aware representation learned in the previous stage.

### Training

```bash
python train_retriever.py \
  -d D \
  --intent_run_dir artifacts/intent_selector \
  --g_cache artifacts/gte/G.pt
```

**Note:**  
`--intent_run_dir` must point to the output directory from the **Intent Selector** stage  
(containing files such as `beta.npy`, `ckpt.pt`, and `topics_top10.jsonl`).  
 
`--g_cache` specifies the cached word embedding matrix (`G.pt`) generated during that stage.

## 2-3: Joint Training ( Intent Selector + Retriever + LLM )

We jointly train the Retriever, LLM, and Intent Selector in an end-to-end fashion.
This step aligns all components within a unified reasoning framework.

### Training

```bash
python train_joint.py \
  -d D \
  --intent_run_dir artifacts/intent_selector \
  --g_cache artifacts/gte/G.pt \
  --retriever_ckpt retriever_intent_Feb12-03:40:12/best.pth \
  --end_to_end
```

## 3-1: Adaptive DPO

In the final stage, we refine the LLM using Adaptive Direct Preference Optimization (ADPO).
Unlike standard DPO, ADPO selectively applies preference learning only when the emphasized (weighted) context demonstrably improves answer quality over the plain context.

### Training

We jointly train the Retriever, LLM, and Intent Selector in an end-to-end fashion.
This step aligns all components within a unified reasoning framework.

```bash
python train_joint.py \
  -d D \
  --intent_run_dir artifacts/intent_selector \
  --g_cache artifacts/gte/G.pt \
  --retriever_ckpt retriever_intent_Feb12-03:40:12/best.pth \
  --use_adaptive_dpo \
  --confidence_metric hit@1 \
  --adaptive_threshold 0.05 \
  --do_test
```

