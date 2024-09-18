from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.flexmodels.common.flex_model_config import FlexModelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from typing import Dict, Literal, Optional, Tuple, Union
from megatron.core.transformer.enums import AttnMaskType, ModelType
import torch
from torch import Tensor
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_rank,
)
from megatron.core.parallel_state import get_op_start_index, get_op_end_index
from megatron.core.flexmodels.common.flex_ops import (
    FlexEmbeddingInfo,
    OpType,
    FlexLayerNormMlpDropoutInfo,
    FlexLayerNormSelfAttentionDropoutInfo,
)
from megatron.core.flexmodels.common.flex_ops import gen_op
from megatron.core.flexmodels.common.flex_model import get_flex_model


class FlexGPTModel(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        flex_config: FlexModelConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        profiling=False,
    ) -> None:
        super().__init__(config)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.rotary_base = rotary_base
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.model_type = ModelType.encoder_or_encoder

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if not profiling:
            num_layers = self.config.num_layers

            global op_start_index, op_end_index
            pipeline_rank = get_pipeline_model_parallel_rank()
            virtual_pipeline_rank = get_virtual_pipeline_model_parallel_rank()
            op_start_index = get_op_start_index(pipeline_rank, virtual_pipeline_rank)
            op_end_index = get_op_end_index(pipeline_rank, virtual_pipeline_rank)
            num_ops = op_end_index - op_start_index
            assert num_ops >= 0

            full_op_list = self.gen_oplist()
            current_op_list = []
            for i in range(op_start_index, op_end_index):
                current_op_list.append(gen_op(full_op_list[i]))

            self.language_model = get_flex_model(
                config,
                flex_config,
                full_model_op_list=current_op_list,
                pre_process=pre_process,
                post_process=post_process,
            )

        else:
            self.full_op_list = self.gen_oplist()

    def gen_oplist(self) -> list:
        op_list = []

        if self.pre_process:
            op_list.append(
                FlexEmbeddingInfo(
                    op_type=OpType.EMBEDDING,
                    op_index=0,
                    op_name="dec-embedding",
                    prev_name=None,
                    config=self.config,
                    vocab_size=self.vocab_size,
                    max_sequence_length=self.max_sequence_length,
                    position_embedding_type=self.position_embedding_type,
                    rotary_percent=self.rotary_percent,
                    rotary_base=self.rotary_base,
                    seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                )
            )
        prev_name = "dec-embedding"
        num_ops_per_transformer = 2
        for i in range(self.config.num_layers):
            op_list.extend(
                [
                    FlexLayerNormSelfAttentionDropoutInfo(
                        op_type=OpType.LAYER_NORM_SELF_ATTENTION_DROPOUT,
                        op_index=i * num_ops_per_transformer + 0,
                        op_name="dec-self-attention",
                        prev_name=prev_name,
                        config=self.config,
                        submodules=self.transformer_layer_spec,
                        layer_number=i + 1,
                    ),
                    FlexLayerNormMlpDropoutInfo(
                        op_type=OpType.LAYER_NORM_MLP_DROPOUT,
                        op_index=i * num_ops_per_transformer + 1,
                        op_name="dec-mlp",
                        prev_name="dec-self-attention",
                        config=self.config,
                        submodules=self.transformer_layer_spec,
                        layer_number=i + 1,
                    ),
                ]
            )
            prev_name = "dec-mlp"

        # TODO: 这个LayerNorm可以和post process合到一起
        # op_list.append(FlexLayerNormInfo(op_type= OpType.LAYER_NORM, op_index=1 + self.config.num_layers * num_ops_per_transformer + 0, prev_name=prev_name))
        # TODO: post process
        # if self.post_process:
        #     if self.config.defer_embedding_wgrad_compute:
        #         # The embedding activation buffer preserves a reference to the input activations
        #         # of the final embedding projection layer GEMM. It will hold the activations for
        #         # all the micro-batches of a global batch for the last pipeline stage. Once we are
        #         # done with all the back props for all the microbatches for the last pipeline stage,
        #         # it will be in the pipeline flush stage. During this pipeline flush we use the
        #         # input activations stored in embedding activation buffer and gradient outputs stored
        #         # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
        #         self.embedding_activation_buffer = []
        #         self.grad_output_buffer = []
        #     else:
        #         self.embedding_activation_buffer = None
        #         self.grad_output_buffer = None
        #     # TODO: post process层如何和embedding层同步信息
        #     # op_list.append(ColumnParallelLinearInfo())
        # if self.pre_process or self.post_process:
        #     self.setup_embeddings_and_output_layer()
        return op_list

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert (
            len(input_tensor) == 1
        ), "input_tensor should only be length 1 for gpt/bert"
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_tensors, input_extra_tensors):
        lm_output = self.language_model(input_tensors, input_extra_tensors)
        return lm_output
