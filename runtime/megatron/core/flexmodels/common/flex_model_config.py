from dataclasses import dataclass

@dataclass
class FlexModelConfig():
    # 这个参数在megatron中没有被使用
    scatter_gather_tensors_in_pipeline: bool = True