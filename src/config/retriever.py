import pydantic
import yaml
from typing import Optional, List, Dict, Any

from .base import EnvYaml

class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str


class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int


class RetrieverYaml(pydantic.BaseModel):
    emb_size: int
    topic_pe: bool
    DDE_kwargs: DDEYaml
    use_intent: bool
    num_intents: int 


class OptimizerYaml(pydantic.BaseModel):
    lr: float
    weight_decay: float  
    selector_lr: float 


class EvalYaml(pydantic.BaseModel):
    k_list: str


class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str


class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    optimizer: OptimizerYaml
    eval: EvalYaml
    train: RetrieverExpYaml


# ========== Joint Training Config ==========

class SelectionYaml(pydantic.BaseModel):
    tau_gumbel: float


class LossYaml(pydantic.BaseModel):
    focal_alpha: float
    focal_gamma: float
    ranking_margin: float
    decorrelation_k: int
    pattern_k: int 
    pattern_temp: float  


class JointRetrieverYaml(pydantic.BaseModel):
    emb_size: int
    topic_pe: bool
    num_intents: int
    DDE_kwargs: DDEYaml
    selection: SelectionYaml
    loss: LossYaml


class KGWrapperYaml(pydantic.BaseModel):
    start: str
    end: str


class MarkersYaml(pydantic.BaseModel):
    high: str
    medium: str
    low: str


class NeuralPromptYaml(pydantic.BaseModel):
    overlap_encoder_hidden: int
    dropout: float
    gate_boost: float  
    system_prompt: str
    cot_prompt: str
    kg_wrapper: KGWrapperYaml
    markers: MarkersYaml


class QuantizationYaml(pydantic.BaseModel):
    load_in_4bit: bool
    compute_dtype: str
    quant_type: str
    use_double_quant: bool


class LoRAYaml(pydantic.BaseModel):
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str


class LLMYaml(pydantic.BaseModel):
    quantization: QuantizationYaml
    lora: LoRAYaml


class TripleProjectionYaml(pydantic.BaseModel):
    hidden_divisor: int


class LearningRatesYaml(pydantic.BaseModel):
    retriever: float
    triple_proj: float
    enhanced_prompt: float
    selector: float
    llm_lora: float
    dpo_only: float


class JointOptimizerYaml(pydantic.BaseModel):
    weight_decay: float
    max_grad_norm: float
    learning_rates: LearningRatesYaml


class TrainingYaml(pydantic.BaseModel):
    epochs: int
    patience: int
    gradient_accumulation_steps: int
    per_intent_total: int
    freeze_retriever_epochs: int
    train_selector_after: int
    rho: float
    rank_weight: float
    diversity_weight: float
    pattern_div_weight: float
    dpo_beta: float
    dpo_weight: float
    dpo_threshold: float
    dpo_margin: float


class TestYaml(pydantic.BaseModel): 
    gen_max_new_tokens: int
    fixed_total_test: int
    limit_test: int


class JointTrainYaml(pydantic.BaseModel):
    env: EnvYaml  
    dataset: DatasetYaml
    retriever: JointRetrieverYaml
    neural_prompt: NeuralPromptYaml
    llm: LLMYaml
    triple_projection: TripleProjectionYaml
    optimizer: JointOptimizerYaml
    training: TrainingYaml  
    test: TestYaml  


def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task', None)
    
    if task is None:
        config = JointTrainYaml(**yaml_data).model_dump()
        return config
    
    elif task == 'retriever':
        config = RetrieverTrainYaml(**yaml_data).model_dump()
        config['eval']['k_list'] = [
            int(k) for k in config['eval']['k_list'].split(',')
        ]
        return config
    
    elif task == 'joint':  # ✅ 추가
        config = JointTrainYaml(**yaml_data).model_dump()
        return config
    
    else:
        raise ValueError(f"Unsupported task type: {task}")