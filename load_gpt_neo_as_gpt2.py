# function to load gpt-neo models using the huggingface transformers GPT2 implementation
import torch
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer, GPT2Config
from transformers.file_utils import cached_path, WEIGHTS_NAME, hf_bucket_url
from pathlib import Path
import os

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

if __name__ == "__main__":
    print("half precision can give different results, but float precision matches\n")

    model, config = load_as_gpt2("EleutherAI/gpt-neo-125M")
    model = model.cuda().eval().float()
    tokenizer = AutoTokenizer.from_pretrained('gpt2', fast=True)

    ids = tokenizer.encode("In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.", return_tensors='pt').cuda()

    torch.manual_seed(0)
    generated_ids = model.generate(ids, do_sample=True, temperature=1.2, use_cache=True, max_length=600, pad_token_id=50256)[0]
    print("loaded as GPT2: " + tokenizer.decode(generated_ids) + "\n")

    from transformers import GPTNeoForCausalLM
    neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").cuda().eval().float()

    torch.manual_seed(0)
    generated_ids = neo_model.generate(ids, do_sample=True, temperature=1.2, use_cache=True, max_length=600, pad_token_id=50256)[0]
    print("loaded as GPT-Neo: " + tokenizer.decode(generated_ids) + "\n")
    del neo_model

    print("saving as vanilla gpt2 model")
    import os
    try:
        os.mkdir("gpt2-125M-neo")
    except:
        pass
    config.to_json_file("gpt2-125M-neo/config.json")
    torch.save(model.state_dict(), "gpt2-125M-neo/pytorch_model.bin")

    print("loading vanilla gpt2 model")
    model = GPT2LMHeadModel.from_pretrained("gpt2-125M-neo").cuda().eval().float()

    torch.manual_seed(0)
    generated_ids = model.generate(ids, do_sample=True, temperature=1.2, use_cache=True, max_length=600, pad_token_id=50256)[0]
    print("loaded as vanilla GPT2: " + tokenizer.decode(generated_ids) + "\n")
