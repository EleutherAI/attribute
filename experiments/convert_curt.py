# %%
from IPython import get_ipython
from pathlib import Path

if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    ip.run_line_magic("env", "CUDA_VISIBLE_DEVICES=6")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.chdir(Path(__file__).parent.parent)

model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_hf = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sparsify.config import SparseCoderConfig
from safetensors.torch import load_file, save_file
from attribute.caching import TranscodedModel
import json

# clt_name = "untied_global_batchtopk_jumprelu"
# clt_name = "tied_per_target_skip_global_batchtopk_jumprelu"
# clt_name = "untied-layerwise-tokentopk"
# clt_name = "untied_layerwise_batchtopk_jumprelu"
clt_name = "tied-per-target-layerwise-token-topk"
clt_file_name = clt_name
if clt_name in ("untied-layerwise-tokentopk", "tied-per-target-layerwise-token-topk"):
    clt_file_name = "clt_checkpoint_97689"
st_file = hf_hub_download(repo_id="ctigges/gpt2-clts",
                          filename=f"{clt_name}/{clt_file_name}.safetensors")
cfg_file = hf_hub_download(repo_id="ctigges/gpt2-clts",
                          filename=f"{clt_name}/cfg.json")
cfg = json.load(open(cfg_file))
output_path = Path(f"results/gpt2-curt-clt-{clt_name}")
#%%
transcoder_weights = load_file(st_file, device="cpu")
#%%
norm_stats = hf_hub_download(repo_id="ctigges/gpt2-clts",
                             filename=f"norm_stats.json")
norm_stats = json.load(open(norm_stats))
#%%
from tqdm import trange
import json

for layer_idx in trange(cfg["num_layers"]):
    norm_stat = norm_stats[str(layer_idx)]
    stat_in = norm_stat["inputs"]
    in_mean, in_std = torch.tensor(stat_in["mean"]), torch.tensor(stat_in["std"])
    stat_out = norm_stat["targets"]
    out_mean, out_std = torch.tensor(stat_out["mean"]), torch.tensor(stat_out["std"])

    enc_bias = transcoder_weights[f"encoder_module.encoders.{layer_idx}.bias_param"]
    if cfg["activation_fn"] == "topk":
        threshold = torch.zeros_like(enc_bias)
    else:
        threshold = transcoder_weights["theta_manager.log_threshold"][layer_idx]
        threshold = threshold.exp()
    enc_bias = enc_bias - threshold
    post_enc = threshold

    enc_weight = transcoder_weights[f"encoder_module.encoders.{layer_idx}.weight"]

    enc_weight = enc_weight / in_std
    enc_bias = enc_bias - in_mean @ enc_weight.T

    decoders = {}
    n_targets = cfg["num_layers"] - layer_idx
    W_decs = []
    b_decs = []
    per_target = cfg["decoder_tying"] == "per_target"
    for source_idx in range(layer_idx + 1 if not per_target else 1):
        source_key = f"{source_idx}->{layer_idx}" if not per_target else f"{layer_idx}"
        W_dec = transcoder_weights[f"decoder_module.decoders.{source_key}.weight"].T.contiguous()
        W_dec = W_dec * out_std
        W_decs.append(W_dec)
        b_dec = transcoder_weights[f"decoder_module.decoders.{source_key}.bias_param"]
        b_decs.append(b_dec)

    state_dict = {}

    if cfg["skip_connection"]:
        state_dict["W_skip"] = transcoder_weights[f"decoder_module.skip_weights.{layer_idx}"] / in_std[None, :] * out_std[:, None]
        out_mean = out_mean - in_mean @ state_dict["W_skip"].T

    b_dec = torch.stack(b_decs, dim=0).sum(dim=0)
    b_dec = b_dec * out_std + out_mean
    state_dict |= {
        "encoder.weight": enc_weight,
        "encoder.bias": enc_bias,
        "W_dec": torch.cat(W_decs, dim=0),
        "b_dec": b_dec,
        **{f"post_encs.{i}": post_enc.clone() for i in range(n_targets)},
    }
    layer_path = output_path / f"h.{layer_idx}.mlp"
    layer_path.mkdir(parents=True, exist_ok=True)
    cfg_path = layer_path / "cfg.json"
    weights_path = layer_path / "sae.safetensors"
    save_file(state_dict, weights_path)
    num_latents, d_in = cfg["num_features"], cfg["d_model"]
    config = SparseCoderConfig(
        activation="topk",
        k=128 if cfg["activation_fn"] != "topk" else cfg["topk_k"],
        n_targets=n_targets,
        n_sources=layer_idx + 1 if not per_target else 0,
        num_latents=num_latents,
        transcode=True,
        skip_connection="W_skip" in state_dict,
        coalesce_topk="per-layer" if not per_target else "concat",
        topk_coalesced=False,
        divide_cross_layer=False,
        normalize_io=False
    )
    json.dump(config.to_dict() | dict(d_in=d_in), cfg_path.open("w"))
#%%
model_hooked = TranscodedModel(
    model_hf,
    output_path,
    # "../e2e/checkpoints/gpt2-sweep/bs16-lr2e-4-btopk-clt-noskip-ef128-k16-adam8",
    # "../e2e/checkpoints/gpt2-sweep/bs8-lr3e-4-tied-ef128-k16",
    device=device,
    pre_ln_hook=True,
)
#%%
import numpy as np
result = model_hooked("<|endoftext|>In another moment, down went Alice after it, never once considering how in the world she was to get out again.");
avg_fvu = np.mean([fvu for fvu in result.error_magnitudes])
total_l0 = np.sum([l0 for l0 in result.l0_per_layer])
avg_fvu, total_l0

# %%
