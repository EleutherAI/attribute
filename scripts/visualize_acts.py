#%%
%load_ext autoreload
%autoreload 2
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from natsort import natsorted
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from collections import defaultdict
from pathlib import Path
import os
from safetensors.torch import load_file

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
name = "bs8-lr3e-4-tied-ef128-k16"
weights_path = f"../e2e/checkpoints/gpt2-sweep/{name}"
w_decs = []
for layer_idx in range(12):
    w_dec = load_file(f"{weights_path}/h.{layer_idx}.mlp/sae.safetensors")
    w_decs.append(w_dec["W_dec"])
#%%
import io, base64
from matplotlib import pyplot as plt
raw_dir = Path(f"./results/gpt2-sweep-cache/{name}/latents")

feat_tensor = torch.arange(0, 1000)
alignments = {}
alignment_pngs = {}
for feat in feat_tensor.tolist():
    w_dec_arr = torch.stack([
        w_dec[feat]
        for w_dec in w_decs
    ], dim=0)
    w_decs_normed = w_dec_arr / w_dec_arr.norm(dim=1, keepdim=True)
    alignments[feat] = (w_decs_normed @ w_decs_normed.T).pow(2)
    plt.figure(figsize=(3, 3))
    plt.imshow(alignments[feat])
    plt.xticks(np.arange(len(w_decs)))
    plt.yticks(np.arange(len(w_decs)))
    bytes_io = io.BytesIO()
    plt.savefig(bytes_io)
    png_raw = base64.b64encode(bytes_io.getvalue()).decode("utf-8")
    alignment_pngs[feat] = f"<img src='data:image/png;base64,{png_raw}' />"
    plt.close()
#%%
latents = {
    module_name.name: feat_tensor
    for module_name in raw_dir.glob("*")
}
ds = LatentDataset(
    raw_dir,
    SamplerConfig(), ConstructorConfig(),
    modules=natsorted(latents.keys()),
    latents=latents,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
#%%
bar = tqdm()
latent_layer = defaultdict(list)
async for i in ds:
    module_name = i.latent.module_name
    layer_idx = int(module_name.split(".")[-2])
    h1 = f"<h1>{i.latent.latent_index} #{layer_idx}</h1>"
    html = f"{h1}{i.display(tokenizer, do_display=False, example_source='train')}"
    latent_layer[i.latent.latent_index].append(html)
    bar.update(1)
#%%
import shutil
import numpy as np
save_dir = Path(f"results/htmls/{name}")
shutil.rmtree(save_dir, ignore_errors=True)
save_dir.mkdir(parents=True, exist_ok=True)
for latent_idx, htmls in sorted(latent_layer.items()):
    if len(htmls) < 2:
        continue
    if latent_idx not in alignment_pngs:
        continue
    alignment = alignments[latent_idx]
    off_axis = (alignment * (1 - np.eye(len(alignment)))).sum()
    off_axis_ratio = off_axis / len(alignment)
    if off_axis_ratio < 0.1:
        continue
    combined = alignment_pngs[latent_idx] + "<br>" + "<br>".join(htmls)
    open(save_dir / f"{latent_idx}.html", "w").write(combined)
# %%
