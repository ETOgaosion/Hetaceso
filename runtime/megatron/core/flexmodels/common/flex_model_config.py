from dataclasses import dataclass

@dataclass
class FlexModelConfig():
    model_name: str = None 
    recompute_ops: list[int] = None
    flex_recompute_activations: list[int] = None 
    resharding_stages: list[int] = None
    # 这个参数在megatron中没有被使用
    scatter_gather_tensors_in_pipeline: bool = True