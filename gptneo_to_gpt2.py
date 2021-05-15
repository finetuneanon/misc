# convert GPT-Neo Huggingface Transformers models to GPT2 with fp16 weights
import torch
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer, GPT2Config
from transformers.file_utils import cached_path, WEIGHTS_NAME, hf_bucket_url
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input model", type=str, required=True)
parser.add_argument("-o", "--output", help="output model", type=str, required=True)
args = parser.parse_args()

def load_as_gpt2(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.n_embd = config.hidden_size
    config.n_ctx = config.max_position_embeddings
    config.n_head = config.num_heads
    config.n_layer = config.num_layers
    config.model_type = "gpt2"
    config.n_positions = config.max_position_embeddings
    config.scale_attn_weights = False

    if os.path.isdir(model_name_or_path):
        resolved_archive_file = str(Path(model_name_or_path) / "pytorch_model.bin")
    else:
        archive_file = hf_bucket_url(model_name_or_path, filename=WEIGHTS_NAME)
        resolved_archive_file = cached_path(archive_file)
    checkpoint = torch.load(resolved_archive_file)
    keys = list(checkpoint.keys())
    converted = {}

    def to_conv1d(weights):
        weights.data = weights.data.T.contiguous()
        return weights

    for key in keys:
        if key not in checkpoint:
            continue
        data = checkpoint[key]
        if 'attention.bias' in key:
            del data
            continue
        key = key.replace('attn.attention', 'attn').replace('out_proj', 'c_proj')
        if 'attn.k_proj.' in key:
            if 'weight' not in key:
                raise KeyError("not weights: " + key)
            k_data = (data)
            q_key = key.replace('k_proj', 'q_proj').replace('attn', 'attn.attention')
            q_data = (checkpoint[q_key])
            v_key = key.replace('k_proj', 'v_proj').replace('attn', 'attn.attention')
            v_data = (checkpoint[v_key])
            key = key.replace('k_proj', 'c_attn')
            if key in converted:
                raise KeyError("key exists: " + key)
            converted[key] = to_conv1d(torch.cat([q_data, k_data, v_data], dim=0))
            del checkpoint[q_key]
            del checkpoint[v_key]
            del data
            continue
        if 'attn.q_proj.' in key or 'attn.v_proj.' in key:
            continue
        if 'mlp.c_' in key and '.weight' in key:
            data = to_conv1d(data)
        if 'attn.c_proj' in key and '.weight' in key:
            data = to_conv1d(data)
        if key in converted:
            raise KeyError("key exists: " + key)
        converted[key] = data
    del checkpoint

    max_positions = config.n_ctx
    window_size = config.window_size
    bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(1, 1, max_positions, max_positions).bool()
    local_bias = bias ^ torch.tril(bias, -window_size)
    for i in range(config.n_layer):
        if config.attention_layers[i] == "local":
            converted[f"transformer.h.{i}.attn.bias"] = local_bias
        elif config.attention_layers[i] == "global":
            converted[f"transformer.h.{i}.attn.bias"] = bias
        else:
            raise AttributeError("unknown attention type: " + config.attention_layers[i])

    gpt2_config = GPT2Config()
    fields = [a for a in dir(gpt2_config) if not a.startswith('__') and not callable(getattr(gpt2_config, a))]
    for field in fields:
        if hasattr(config, field):
            try:
                setattr(gpt2_config, field, getattr(config, field))
            except:
                pass

    model = GPT2LMHeadModel(gpt2_config)
    model.load_state_dict(converted, strict=False)

    keys = list(converted.keys())
    for key in keys:
        del converted[key]
    del converted

    return model, gpt2_config

print("loading model: " + args.input)
model, config = load_as_gpt2(args.input)
model = model.half()

print("saving gpt2 model: " + args.output)
try:
    os.mkdir(args.output)
except:
    pass
config.to_json_file(args.output + "/config.json")
torch.save(model.state_dict(), args.output + "/pytorch_model.bin")
