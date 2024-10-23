from enum import Enum
from typing import Dict, Literal, Optional, Tuple, Union

import torch.distributed
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
from megatron.core.transformer.transformer_layer import (
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.tensor_parallel import vocab_parallel_cross_entropy, VocabParallelEmbedding
from megatron.core.models.common.language_module.language_module import parallel_lm_logits
class OpType(Enum):
    EMBEDDING = 1
    LAYER_NORM_SELF_ATTENTION_DROPOUT = 2
    LAYER_NORM_MLP_DROPOUT = 3
    LAYER_NORM_POST_PROCESS = 4

@dataclass
class OpInfo:
    op_type: OpType
    op_index: int
    op_name: str
    prev_name: str



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

        ## for profiling
        self.weight_size = 0

        # shapes
        self.seq_length = config.seq_length
        self.micro_batch_size = config.micro_batch_size
        self.hidden_size = config.hidden_size
        # [s, b, h]
        self.hidden_state_size = [
            config.seq_length,
            config.micro_batch_size,
            config.hidden_size,
        ]

        self.input_tensors_info = {}
        self.output_tensors_info = {}
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}
        self.shared_weights_info={}
    
    def get_shared_tensor(self, grads=False):
        tensor_dict = {}
        for key in sorted(self.shared_weights_info):
            if key == "word_embeddings":
                if grads:
                    tensor_dict[key] = self.embedding.word_embeddings.weight.main_grad
                else:
                    tensor_dict[key] = self.embedding.word_embeddings.weight.data
        return tensor_dict

    def set_shared_tensor(self, new_data, grads=False):
        for key in sorted(self.shared_weights_info):
            if key == "word_embeddings":
                if grads:
                    self.embedding.word_embeddings.weight.main_grad = new_data[key][0]
                else:
                    self.embedding.word_embeddings.weight.data = new_data[key][0]


@dataclass
class OpInfo:
    op_type: OpType
    op_index: int
    op_name: str
    prev_name: str

@dataclass
class FlexEmbeddingInfo(OpInfo):
    config: TransformerConfig
    num_tokentypes: int = 0
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
        num_tokentypes: int,
        rotary_base: int,
        seq_len_interpolation_factor: Optional[float],
    ):
        super().__init__(config, op_type, op_index, op_name, prev_name)
        self.padded_vocab_size = config.padded_vocab_size
        self.max_sequence_length = config.max_position_embeddings
        self.position_embedding_type = config.position_embedding_type

        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.padded_vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
            num_tokentypes=num_tokentypes,
        )

        self.embedding.word_embeddings.weight.shared_embedding = True

        if self.position_embedding_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=config.rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        self.weight_size = (
            config.padded_vocab_size * config.hidden_size
        ) / self.tp_size + config.max_position_embeddings * config.hidden_size

        self.input_tensors_info = {
            'input_ids': {'shape': [self.micro_batch_size, self.seq_length], 'tp_split_dim': -1, 'dp_split_dim': -1}, 
            'position_ids': {'shape': [self.micro_batch_size, self.seq_length,], 'tp_split_dim': -1, 'dp_split_dim': -1}
        }
        
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1,
                "dp_split_dim": 1,
            }
        }
        
        self.shared_weights_info = {
            "word_embeddings": {
                "root": True,
                "sharing_with_ops": [config.num_layers * 2 + 1],
                "shape": [config.padded_vocab_size, config.hidden_size],
                "tp_split_dim": 0,
                "dp_split_dim": -1,
            }
        }

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
        inference_params: InferenceParams = input_tensors.get("inference_params", None)

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
class FlexLayerNormSelfAttentionDropoutInfo(OpInfo):
    config: TransformerConfig
    submodules: TransformerLayerSubmodules
    layer_number: int = 1
    hidden_dropout: float = None


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

        qkv_projection_size = config.kv_channels * config.num_attention_heads
        qkv_weight = config.hidden_size * qkv_projection_size * 3 / self.tp_size
        dense_weight = (
            (config.kv_channels * config.num_attention_heads) * config.hidden_size
        ) / self.tp_size
        self.weight_size = qkv_weight + dense_weight
        self.input_tensors_info = {'hidden_states': {'shape': self.hidden_state_size, 'tp_split_dim': -1, 'dp_split_dim': 1}}
        self.output_tensors_info = {'hidden_states': {'shape': self.hidden_state_size, 'tp_split_dim': -1, 'dp_split_dim': 1}}
        self.input_extra_tensors_info = {
            "attention_mask": {
                "shape": [
                    config.micro_batch_size // self.dp_size,
                    1,
                    config.seq_length,
                    config.seq_length,
                ],
                "tp_split_dim": -1,
                "dp_split_dim": -1,
                "recv_from": 0,
            }
        }
    def forward(
        self,
        input_tensors: Dict | list,
        input_extra_tensors: Dict,
        output_extra_tensors: Dict,
        profiling=False,
    ):
        output_tensors = {}
        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]

        rotary_pos_emb = input_tensors.get("rotary_pos_emb", None)
        inference_params = input_tensors.get("inference_params", None)
        packed_seq_params = input_tensors.get("packed_seq_params", None)

        attention_mask: Tensor = input_extra_tensors["attention_mask"]
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
        return output_tensors


@dataclass
class FlexLayerNormMlpDropoutInfo(OpInfo):
    config: TransformerConfig
    submodules: TransformerLayerSubmodules
    layer_number: int = 1
    hidden_dropout: float = None


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

        gemm_1_weight = config.hidden_size * config.ffn_hidden_size / self.tp_size
        gemm_2_weight = config.ffn_hidden_size * config.hidden_size / self.tp_size
        self.weight_size = gemm_1_weight + gemm_2_weight

        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1,
                "dp_split_dim": 1,
            }
        }

    def forward(
        self,
        input_tensors: Dict | list,
        input_extra_tensors: Dict,
        output_extra_tensors: Dict,
        profiling=False,
    ):
        output_tensors = {}
        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]
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
        return output_tensors


@dataclass
class FlexLayerNormPostProcessInfo(OpInfo):
    config: TransformerConfig
    submodules: TransformerLayerSubmodules
    parallel_output: bool = True
    num_tokentypes: int = 0


class FlexLayerNormPostProcess(FlexModule):
    def __init__(
        self,
        op_type: OpType,
        op_index: int,
        op_name: str,
        prev_name: str,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        parallel_output: bool,
        num_tokentypes: int,
    ):
        super().__init__(config, op_type, op_index, op_name, prev_name)

        self.hidden_dropout = self.config.hidden_dropout

        self.final_layernorm = build_module(
            submodules.final_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        
        if config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None
        
        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy
        
        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config=config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=False,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )
        
        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            position_embedding_type=config.position_embedding_type,
            num_tokentypes=num_tokentypes,
        )
        self.embedding.word_embeddings.weight.data.fill_(0)
        self.embedding.word_embeddings.weight.shared = True

        self.embedding.word_embeddings.weight.shared_embedding = True

        self.weight_size = config.padded_vocab_size * config.hidden_size / self.tp_size

        self.input_tensors_info = {'hidden_states': {'shape': self.hidden_state_size, 'tp_split_dim': -1, 'dp_split_dim': 1}}
        self.output_tensors_info = {'output_tensor': {'shape': [1], 'tp_split_dim': -1, 'dp_split_dim': -1}}
        self.input_extra_tensors_info = {
            "labels": {
                "shape": [
                    config.micro_batch_size // self.dp_size,
                    config.seq_length
                ],
                "tp_split_dim": -1,
                "dp_split_dim": 0,
                "recv_from": 0,
            }
        }
        self.shared_weights_info = {
            "word_embeddings": {
                "root": False,
                "sharing_with_ops": [0],
                "shape": [config.padded_vocab_size, config.hidden_size],
                "tp_split_dim": 0,
                "dp_split_dim": -1,
            }
        }
        
    def forward(
        self,
        input_tensors: Dict | list,
        input_extra_tensors: Dict,
        output_extra_tensors: Dict,
        profiling=False,
    ):
        output_tensors = {}

        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]

        # Optional Layer norm post the cross-attention.
        final_layernorm_output = self.final_layernorm(hidden_states)

        # always post process
        weights = self.embedding.word_embeddings.weight

        output, _ = self.output_layer(final_layernorm_output, weights)
        
        labels = input_extra_tensors["labels"]

        if labels is None:
            output_tensors["output_tensor"] = output.transpose(0, 1).contiguous()
            return output_tensors
        else:
            labels = labels.transpose(0, 1).contiguous()
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
            loss = loss.transpose(0, 1).contiguous()

        output_tensors["output"] = loss

        return output_tensors


def gen_op(
    op_info: (
        FlexEmbeddingInfo
        | FlexLayerNormSelfAttentionDropoutInfo
        | FlexLayerNormMlpDropoutInfo
        | FlexLayerNormPostProcessInfo
    ),
):
    op = None
    if op_info.op_type == OpType.EMBEDDING:
        op = FlexEmbedding(**vars(op_info))
    elif op_info.op_type == OpType.LAYER_NORM_SELF_ATTENTION_DROPOUT:
        op = FlexLayerNormSelfAttentionDropout(**vars(op_info))
    elif op_info.op_type == OpType.LAYER_NORM_MLP_DROPOUT:
        op = FlexLayerNormMlpDropout(**vars(op_info))
    elif op_info.op_type == OpType.LAYER_NORM_POST_PROCESS:
        op = FlexLayerNormPostProcess(**vars(op_info))
    else:
        raise ValueError(f"Unknown op type: {op_info.op_type}")
    return op
