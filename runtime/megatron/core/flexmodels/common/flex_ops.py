from enum import Enum
from megatron.core.flexmodels.common.flex_module import FlexModule
from typing import Dict, Literal, Optional, Tuple, Union
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core import InferenceParams, parallel_state, tensor_parallel
import torch
from torch import Tensor
from dataclasses import dataclass
from megatron.core.transformer.spec_utils import build_module, ModuleSpec
from runtime.megatron.core.transformer.transformer_layer import (
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor


class OpType(Enum):
    EMBEDDING = 1
    LAYER_NORM_SELF_ATTENTION_DROPOUT = 2
    LAYER_NORM_MLP_DROPOUT = 3


class OpInfo:
    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        op_index: int,
        input_width: int,
        in_channels: int,
        num_classes,
        prev_name,
        output_width,
    ) -> None:
        self.op_type = op_type
        self.op_name = op_name
        self.op_index = op_index
        self.input_width = input_width
        self.in_channels = in_channels
        self.num_calsses = num_classes
        self.prev_name = prev_name
        self.output_width = output_width


@dataclass
class FlexEmbeddingInfo:
    op_type: OpType
    op_index: int
    op_name: str
    prev_name: str
    config: TransformerConfig
    vocab_size: int
    max_sequence_length: int
    position_embedding_type: Literal["learned_absolute", "rope"] = ("learned_absolute",)
    num_tokentypes: int = 0
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None


class FlexEmbedding(FlexModule):
    def __init__(
        self,
        op_type: OpType,
        op_index: int,
        op_name: str,
        prev_name: str,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal["learned_absolute", "rope"],
        num_tokentypes: int,
        rotary_percent: float,
        rotary_base: int,
        seq_len_interpolation_factor: Optional[float],
    ):
        super().__init__(config, op_type, op_index, op_name, prev_name)
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.position_embedding_type = position_embedding_type

        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=position_embedding_type,
            num_tokentypes=num_tokentypes,
        )

        if self.position_embedding_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

    def forward(
        self,
        input_tensors: Dict,
        input_extra_tensors: Dict,
        output_extra_tensors: Dict,
        profiling=False,
    ):
        output_tensors = {}
        input_ids: Tensor = input_tensors["input_ids"]
        position_ids: Tensor = input_tensors["position_ids"]
        inference_params: InferenceParams = input_tensors["inference_params"]

        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        rotary_pos_emb = None
        if self.position_embedding_type == "rope":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, None, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        output_tensors["hidden_states"] = decoder_input
        output_tensors["rotary_pos_emb"] = rotary_pos_emb

        return output_tensors


@dataclass
class FlexLayerNormSelfAttentionDropoutInfo:
    op_type: OpType
    op_index: int
    op_name: str
    prev_name: str
    config: TransformerConfig
    submodules: TransformerLayerSubmodules
    layer_number: int = 1
    hidden_droupout: float = None


class FlexLayerNormSelfAttentionDropout(FlexModule):
    """
    input_layernorm + self_attention + self_attn_bda.

    """

    def __init__(
        self,
        op_type: OpType,
        op_index: int,
        op_name: str,
        prev_name: str,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int,
        hidden_dropout: float,
    ):
        """
        Args:
            layer_number: The global number of transformer layer, start with 1.
        """
        super().__init__(config, op_type, op_index, op_name, prev_name)
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.hidden_dropout = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
        )
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        input_tensors: Dict,
        input_extra_tensors: Dict,
        output_extra_tensors: Dict,
        profiling=False,
    ):
        output_tensors = {}
        hidden_states: Tensor = input_tensors["hidden_states"]
        attention_mask: Tensor = input_tensors["attention_mask"]
        context: Tensor = input_tensors.get("context", None)
        rotary_pos_emb = input_tensors.get("rotary_pos_emb", None)
        inference_params = input_tensors.get("inference_params", None)
        packed_seq_params = input_tensors.get("packed_seq_params", None)

        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        output_tensors["hidden_states"] = hidden_states
        output_tensors["context"] = context
        return output_tensors


@dataclass
class FlexLayerNormMlpDropoutInfo:
    op_type: OpType
    op_index: int
    op_name: str
    prev_name: str
    config: TransformerConfig
    submodules: TransformerLayerSubmodules
    layer_number: int = 1
    hidden_droupout: float = None

class FlexLayerNormMlpDropout(FlexModule):
    def __init__(
        self,
        op_type: OpType,
        op_index: int,
        op_name: str,
        prev_name: str,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int,
        hidden_dropout: float,
    ):
        super().__init__(config, op_type, op_index, op_name, prev_name)

        self.layer_nummber = layer_number
        self.hidden_dropout = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        input_tensors: Dict,
        input_extra_tensors: Dict,
        output_extra_tensors: Dict,
        profiling=False,
    ):
        output_tensors = {}
        hidden_states: Tensor = input_tensors["hidden_states"]
        context: Tensor = input_tensors.get("context", None)
        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        output_tensors["hidden_states"] = output
        output_tensors["context"] = context

        return output_tensors
