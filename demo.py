import torch
import torchvision
import torchKQI
import pandas as pd
import os
from transformers import LlamaConfig, BertConfig, T5Config, Gemma2Config, OpenAIGPTConfig, GPT2Config, Qwen2Config, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


Llama_2_7b_hf = {
  "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "tie_word_embeddings": False,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0.dev0",
  "use_cache": True,
  "vocab_size": 32000
}

Llama_2_13b_hf = {
  "_name_or_path": "meta-llama/Llama-2-13b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 13824,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "num_key_value_heads": 40,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "tie_word_embeddings": False,
  "torch_dtype": "float16",
  "transformers_version": "4.32.0.dev0",
  "use_cache": True,
  "vocab_size": 32000
}

Llama_2_70b_hf = {
  "_name_or_path": "meta-llama/Llama-2-70b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": 28672,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "tie_word_embeddings": False,
  "torch_dtype": "float16",
  "transformers_version": "4.32.0.dev0",
  "use_cache": True,
  "vocab_size": 32000
}

Meta_Llama_3_8B = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "rope_theta": 500000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.0.dev0",
  "use_cache": True,
  "vocab_size": 128256
}

bert_base_uncased = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": True,
  "vocab_size": 30522
}

bert_large_uncased = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": True,
  "vocab_size": 30522
}

t5_small = {
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 512,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "initializer_factor": 1.0,
  "is_encoder_decoder": True,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_heads": 8,
  "num_layers": 6,
  "output_past": True,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "task_specific_params": {
    "summarization": {
      "early_stopping": True,
      "length_penalty": 2.0,
      "max_length": 200,
      "min_length": 30,
      "no_repeat_ngram_size": 3,
      "num_beams": 4,
      "prefix": "summarize: "
    },
    "translation_en_to_de": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to German: "
    },
    "translation_en_to_fr": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to French: "
    },
    "translation_en_to_ro": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to Romanian: "
    }
  },
  "vocab_size": 32128
}

t5_base = {
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "d_ff": 3072,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "initializer_factor": 1.0,
  "is_encoder_decoder": True,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": True,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "task_specific_params": {
    "summarization": {
      "early_stopping": True,
      "length_penalty": 2.0,
      "max_length": 200,
      "min_length": 30,
      "no_repeat_ngram_size": 3,
      "num_beams": 4,
      "prefix": "summarize: "
    },
    "translation_en_to_de": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to German: "
    },
    "translation_en_to_fr": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to French: "
    },
    "translation_en_to_ro": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to Romanian: "
    }
  },
  "vocab_size": 32128
}

t5_large = {
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "d_ff": 4096,
  "d_kv": 64,
  "d_model": 1024,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "initializer_factor": 1.0,
  "is_encoder_decoder": True,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_heads": 16,
  "num_layers": 24,
  "output_past": True,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "task_specific_params": {
    "summarization": {
      "early_stopping": True,
      "length_penalty": 2.0,
      "max_length": 200,
      "min_length": 30,
      "no_repeat_ngram_size": 3,
      "num_beams": 4,
      "prefix": "summarize: "
    },
    "translation_en_to_de": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to German: "
    },
    "translation_en_to_fr": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to French: "
    },
    "translation_en_to_ro": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to Romanian: "
    }
  },
  "vocab_size": 32128
}

gemma_2_2b = {
  "architectures": [
    "Gemma2ForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": 50.0,
  "bos_token_id": 2,
  "cache_implementation": "hybrid",
  "eos_token_id": 1,
  "final_logit_softcapping": 30.0,
  "head_dim": 256,
  "hidden_act": "gelu_pytorch_tanh",
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 2304,
  "initializer_range": 0.02,
  "intermediate_size": 9216,
  "max_position_embeddings": 8192,
  "model_type": "gemma2",
  "num_attention_heads": 8,
  "num_hidden_layers": 26,
  "num_key_value_heads": 4,
  "pad_token_id": 0,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "torch_dtype": "float32",
  "transformers_version": "4.42.4",
  "use_cache": True,
  "vocab_size": 256000
}

gemma_2_9b = {
  "architectures": [
    "Gemma2ForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": 50.0,
  "bos_token_id": 2,
  "cache_implementation": "hybrid",
  "eos_token_id": 1,
  "final_logit_softcapping": 30.0,
  "head_dim": 256,
  "hidden_act": "gelu_pytorch_tanh",
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "gemma2",
  "num_attention_heads": 16,
  "num_hidden_layers": 42,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "sliding_window_size": 4096,
  "torch_dtype": "float32",
  "transformers_version": "4.42.0.dev0",
  "use_cache": True,
  "vocab_size": 256000
}

gpt = {
  "afn": "gelu",
  "architectures": [
    "OpenAIGPTLMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "openai-gpt",
  "n_ctx": 512,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 512,
  "n_special": 0,
  "predict_special_tokens": True,
  "resid_pdrop": 0.1,
  "summary_activation": None,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": True,
  "summary_type": "cls_index",
  "summary_use_proj": True,
  "task_specific_params": {
    "text-generation": {
      "do_sample": True,
      "max_length": 50
    }
  },
  "vocab_size": 40478
}

gpt2 = {
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": None,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": True,
  "summary_type": "cls_index",
  "summary_use_proj": True,
  "task_specific_params": {
    "text-generation": {
      "do_sample": True,
      "max_length": 50
    }
  },
  "vocab_size": 50257
}

Qwen1_5_110B = {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": 49152,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.0",
  "use_cache": True,
  "use_sliding_window": False,
  "vocab_size": 152064
}

Qwen2_7B = {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 131072,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.2",
  "use_cache": True,
  "use_sliding_window": False,
  "vocab_size": 152064
}

Qwen2_72B = {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": 29568,
  "max_position_embeddings": 131072,
  "max_window_layers": 80,
  "model_type": "qwen2",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": True,
  "use_sliding_window": False,
  "vocab_size": 152064
}

Yi_1_5_6B = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 4,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": None,
  "rope_theta": 5000000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.2",
  "use_cache": True,
  "vocab_size": 64000
}

Yi_1_5_34B = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 7168,
  "initializer_range": 0.02,
  "intermediate_size": 20480,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 56,
  "num_hidden_layers": 60,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": None,
  "rope_theta": 5000000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.2",
  "use_cache": True,
  "vocab_size": 64000
}

deepseek_llm_7b_base = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 30,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": None,
  "rope_theta": 10000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.33.1",
  "use_cache": True,
  "vocab_size": 102400
}


def task_ImageClassification():
    x = torch.randn(1, 3, 224, 224)

    model_fns = [
        torchvision.models.alexnet,
        torchvision.models.convnext_tiny, torchvision.models.convnext_small, torchvision.models.convnext_base, torchvision.models.convnext_large,
        torchvision.models.densenet121, torchvision.models.densenet161, torchvision.models.densenet169, torchvision.models.densenet201,
        torchvision.models.efficientnet_b0, torchvision.models.efficientnet_b1, torchvision.models.efficientnet_b2, torchvision.models.efficientnet_b3, torchvision.models.efficientnet_b4, torchvision.models.efficientnet_b5, torchvision.models.efficientnet_b6, torchvision.models.efficientnet_b7, torchvision.models.efficientnet_v2_s, torchvision.models.efficientnet_v2_m, torchvision.models.efficientnet_v2_l,
        torchvision.models.googlenet,
        torchvision.models.inception_v3,
        # torchvision.models.maxvit_t,
        torchvision.models.mnasnet0_5, torchvision.models.mnasnet0_75, torchvision.models.mnasnet1_0, torchvision.models.mnasnet1_3,
        torchvision.models.mobilenet_v2,
        torchvision.models.mobilenet_v3_large, torchvision.models.mobilenet_v3_small,
        torchvision.models.regnet_y_400mf, torchvision.models.regnet_y_800mf, torchvision.models.regnet_y_1_6gf, torchvision.models.regnet_y_3_2gf, torchvision.models.regnet_y_8gf, torchvision.models.regnet_y_16gf, torchvision.models.regnet_y_32gf, torchvision.models.regnet_y_128gf, torchvision.models.regnet_x_400mf, torchvision.models.regnet_x_800mf, torchvision.models.regnet_x_1_6gf, torchvision.models.regnet_x_3_2gf, torchvision.models.regnet_x_8gf, torchvision.models.regnet_x_16gf, torchvision.models.regnet_x_32gf,
        torchvision.models.resnet18, torchvision.models.resnet34, torchvision.models.resnet50, torchvision.models.resnet101, torchvision.models.resnet152,
        torchvision.models.resnext50_32x4d, torchvision.models.resnext101_32x8d, torchvision.models.resnext101_64x4d,
        torchvision.models.wide_resnet50_2, torchvision.models.wide_resnet101_2,
        torchvision.models.shufflenet_v2_x0_5, torchvision.models.shufflenet_v2_x1_0, torchvision.models.shufflenet_v2_x1_5, torchvision.models.shufflenet_v2_x2_0,
        torchvision.models.squeezenet1_0, torchvision.models.squeezenet1_1,
        torchvision.models.swin_t, torchvision.models.swin_s, torchvision.models.swin_b,  # torchvision.models.swin_v2_t, torchvision.models.swin_v2_s, torchvision.models.swin_v2_b,
        torchvision.models.vgg11, torchvision.models.vgg11_bn, torchvision.models.vgg13, torchvision.models.vgg13_bn, torchvision.models.vgg16, torchvision.models.vgg16_bn, torchvision.models.vgg19, torchvision.models.vgg19_bn,
        torchvision.models.vit_b_16, torchvision.models.vit_b_32, torchvision.models.vit_l_16, torchvision.models.vit_l_32, torchvision.models.vit_h_14
    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


def task_SemanticSegmentation():
    x = torch.randn(1, 3, 224, 224)

    model_fns = [
        torchvision.models.segmentation.deeplabv3_mobilenet_v3_large, torchvision.models.segmentation.deeplabv3_resnet50, torchvision.models.segmentation.deeplabv3_resnet101,
        torchvision.models.segmentation.deeplabv3_resnet50,
        torchvision.models.segmentation.deeplabv3_resnet101,
    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x, lambda model, x: model(x)['out']).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


def task_ObjectDetection():
    x = torch.randn(1, 3, 224, 224)

    model_fns = [
        # torchvision.models.detection.fasterrcnn_resnet50_fpn, torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn, torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn, torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        torchvision.models.detection.fcos_resnet50_fpn,
        torchvision.models.detection.retinanet_resnet50_fpn, torchvision.models.detection.retinanet_resnet50_fpn_v2,
        torchvision.models.detection.ssd300_vgg16,
        torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x, lambda model, x: model(x)[0]['boxes']).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


def task_VideoClassification():
    x = torch.randn(1, 3, 3, 224, 224)

    model_fns = [
        # torchvision.models.video.mvit_v1_b, torchvision.models.video.mvit_v2_s,
        torchvision.models.video.r3d_18, torchvision.models.video.mc3_18, torchvision.models.video.r2plus1d_18,
        # torchvision.models.video.s3d,
        # torchvision.models.video.swin3d_t, torchvision.models.video.swin3d_s, torchvision.models.video.swin3d_b

    ]

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for model_fn in model_fns:
        if model_fn.__name__ in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            model = model_fn().eval()
            kqi = torchKQI.KQI(model, x, lambda model, x: model(x)['out']).item()
            result = pd.DataFrame([[model_fn.__name__, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[model_fn.__name__, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


def task_LLM():
    llm_configs = {
        "Llama_2_7b_hf": (Llama_2_7b_hf, LlamaConfig),
        "Llama_2_13b_hf": (Llama_2_13b_hf, LlamaConfig),
        "Llama_2_70b_hf": (Llama_2_70b_hf, LlamaConfig),
        "Meta_Llama_3_8B": (Meta_Llama_3_8B , LlamaConfig),
        "bert_base_uncased": (bert_base_uncased, BertConfig),
        "bert_large_uncased": (bert_large_uncased, BertConfig),
        "t5_small": (t5_small, T5Config),
        "t5_base": (t5_base, T5Config),
        "t5_large": (t5_large, T5Config),
        "gemma_2_2b": (gemma_2_2b, Gemma2Config),
        "gemma_2_9b": (gemma_2_9b, Gemma2Config),
        "gpt": (gpt, OpenAIGPTConfig),
        "gpt2": (gpt2, GPT2Config),
        "Qwen1_5_110B": (Qwen1_5_110B, Qwen2Config),
        "Qwen2_7B": (Qwen2_7B, Qwen2Config),
        "Qwen2_72B": (Qwen2_72B, Qwen2Config),
        "Yi_1_5_6B": (Yi_1_5_6B, LlamaConfig),
        "Yi_1_5_34B": (Yi_1_5_34B, LlamaConfig),
        "deepseek_llm_7b_base": (deepseek_llm_7b_base, LlamaConfig),
    }

    results_file = 'model_results.csv'
    errors_file = 'model_errors.csv'

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['Model Name', 'KQI']).to_csv(results_file, index=False)
    if not os.path.exists(errors_file):
        pd.DataFrame(columns=['Model Name', 'Error']).to_csv(errors_file, index=False)

    for llm_name, llm_config in llm_configs.items():
        if llm_name in pd.read_csv(results_file)['Model Name'].values:
            continue
        try:
            config = llm_config[1].from_dict(llm_config[0])
            x = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings))
            model = AutoModel.from_config(config).eval()
            callback_func = lambda model, x: model(x).logits if isinstance(model(x), CausalLMOutputWithPast) else model(x).last_hidden_state
            kqi = torchKQI.KQI(model, x, callback_func).item()
            result = pd.DataFrame([[llm_name, kqi]], columns=['Model Name', 'KQI'])
            result.to_csv(results_file, mode='a', header=False, index=False)
        except Exception as e:
            error = pd.DataFrame([[llm_name, str(e)]], columns=['Model Name', 'Error'])
            error.to_csv(errors_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    task_ImageClassification()
    task_SemanticSegmentation()
    task_ObjectDetection()
    task_VideoClassification()
    task_LLM()
