#%%
from IPython import get_ipython
if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    ip.run_line_magic("env", "CUDA_VISIBLE_DEVICES=7")
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
import torch
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from sparsify import SparseCoderConfig
from attribute.caching import TranscodedModel
from safetensors.torch import load_file, save_file
torch.set_grad_enabled(False)
# %%
model_name = "gpt2"
repo_id = "mntss/sparsecoder-sweep-gpt2"
branch = "clt-relu-sp10-1b"
# branch = "clt-relu-sp8"
# branch = "relu-sp8-noskip"
# branch = "relu-sp6-skip"
model_hf = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cpu"})
transcoder_path = Path(snapshot_download(repo_id, revision=branch))
output_path = Path(f"results/gpt2-mntss-transcoder-{branch}")
# %%
n_layers = 12
for layer_idx in range(n_layers):
    clt = True
    try:
        encoder = load_file(transcoder_path / f"encoder_layer_{layer_idx}.safetensors", device="cpu")
        decoder = load_file(transcoder_path / f"decoder_layer_{layer_idx}.safetensors", device="cpu")
    except FileNotFoundError:
        clt = False
        initial_state = load_file(transcoder_path / f"layer_{layer_idx}.safetensors", device="cpu")
        encoder = {
            "W_enc": initial_state["W_enc"],
            "b_enc": initial_state["b_enc"],
        }
        decoder = {
            f"W_dec_from_{layer_idx}": initial_state["W_dec"],
            "b_dec": initial_state["b_dec"],
        }
        if "W_skip" in initial_state:
            decoder["W_skip"] = initial_state["W_skip"]
    sources = [
        decoder[f"W_dec_from_{source_idx}"].T
        for source_idx in range(
            0 if clt else layer_idx,
            layer_idx + 1
        )
    ]
    state_dict = {
        "W_dec": torch.cat(sources, dim=0),
        "encoder.weight": encoder["W_enc"].contiguous(),
        "b_dec": decoder["b_dec"],
        "encoder.bias": encoder["b_enc"],
    }
    if "W_skip" in decoder:
        state_dict["W_skip"] = decoder["W_skip"]
    layer_path = output_path / f"h.{layer_idx}.mlp"
    layer_path.mkdir(parents=True, exist_ok=True)
    cfg_path = layer_path / "cfg.json"
    weights_path = layer_path / "sae.safetensors"
    save_file(state_dict, weights_path)
    num_latents, d_in = state_dict["encoder.weight"].shape
    config = SparseCoderConfig(
        k=128,
        num_latents=num_latents,
        transcode=True,
        skip_connection="W_skip" in state_dict,
        coalesce_topk="per-layer" if clt else "none",
        n_targets=n_layers - layer_idx if clt else 0,
        n_sources=len(sources),
    )
    json.dump(config.to_dict() | dict(d_in=d_in), cfg_path.open("w"))
# %%
model = TranscodedModel(
    model_hf,
    output_path,
    device="cuda",
    pre_ln_hook=True,
)
#%%
import numpy as np
result = model("In another moment, down went Alice after it, never once considering how in the world she was to get out again.")
avg_fvu = np.mean([fvu for fvu in result.error_magnitudes])
total_l0 = np.sum([l0 for l0 in result.l0_per_layer])
avg_fvu, total_l0
