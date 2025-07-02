# %%
from IPython import get_ipython
from pathlib import Path

if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-2-2b"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_hf = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
from tqdm import tqdm
from huggingface_hub import snapshot_download
from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import SparseCoder
import sys
sys.path.append("..")
from attribute.caching import TranscodedModel

model_name = "google/gemma-2-2b"
repo_id = "mntss/skip-transcoders-gemma-2-2b"
branch = "no-skip"
transcoder_path = Path(snapshot_download(repo_id, revision=branch))
output_path = Path(f"results/gemma-mntss-{branch}")
#%%
import re
from safetensors.torch import load_file, save_file
import json

for st_file in tqdm(list(transcoder_path.glob("*.safetensors"))):
    layer_idx = int(re.match(r"layer_(\d+)\.safetensors", st_file.name).group(1))
    state_dict = load_file(st_file, device="cpu")
    state_dict["W_dec"] = state_dict["W_dec"].T.contiguous()
    state_dict["encoder.bias"] = state_dict.pop("b_enc")
    state_dict["encoder.weight"] = state_dict.pop("W_enc")

    layer_path = output_path / f"layers.{layer_idx}.mlp"
    layer_path.mkdir(parents=True, exist_ok=True)
    cfg_path = layer_path / "cfg.json"
    weights_path = layer_path / "sae.safetensors"
    save_file(state_dict, weights_path)
    num_latents, d_in = state_dict["W_dec"].shape
    config = SparseCoderConfig(
        activation="topk",
        k=128,
        num_latents=num_latents,
        transcode=True,
        skip_connection="W_skip" in state_dict,
    )
    json.dump(config.to_dict() | dict(d_in=d_in), cfg_path.open("w"))
#%%
model_hooked = TranscodedModel(
    model_hf,
    output_path,
    device=device,
    pre_ln_hook=True,
    post_ln_hook=True,
    offload=True
)
#%%
model_hooked("In another moment, down went Alice after it, never once considering how in the world she was to get out again.");
#%%
