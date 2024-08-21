import torch
from megatron.core.flexmodels.common.flex_ops import OpType
from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class FlexModule(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        op_type: OpType,
        op_index: int,
        op_name: str,
        prev_name: str,
        is_last_op=False,
        
    ):
        super().__init__(config=config)
        self.op_type = op_type
        self.op_name = op_name
        self.prev_name = prev_name
        self.op_index = op_index
        self.is_last_op = is_last_op

        self.tp_size = mpu.get_op_tp_size(op_index)
        self.dp_size = mpu.get_op_dp_size(op_index)

        self.input_tensors_info = {}
        self.output_tensors_info = {}
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}
        self.shared_weights_info = {}

        ## resharding
        self.output_extra_specs = None
        self.output_extra_mats_info = None
        self.required_input_extra_specs = {}
        self.input_extra_mats = None
        self.new_input_extra_tensors = {}
        self.tmp_buffer = None
        self.elementwise = False
        self.input_mats = None
        self.input_extra_mats = None

        ## for profiling
        self.weight_size = 0
