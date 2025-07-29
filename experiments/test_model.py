#%%
from IPython import get_ipython
from pathlib import Path

if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    ip.run_line_magic("env", "CUDA_VISIBLE_DEVICES=0")

import os
os.chdir(Path(__file__).parent.parent)
from attribute.caching import TranscodedModel
model_hooked = TranscodedModel(
    "gpt2",
    # "../e2e/checkpoints/gpt2-sweep/bs2-lr2e-4-clt-noskip-ef128-k16",
    "../e2e/checkpoints/gpt2-sweep/bs8-lr2e-4-tied-no-affine-ef128-k16",
    # "results/gpt2-mntss-transcoder-clt-relu-sp10-1b", pre_ln_hook=True,
    device="cuda",
)
#%%
import numpy as np
result = model_hooked("<|endoftext|>In another moment, down went Alice after it, never once considering how in the world she was to get out again. The quick brown fox jumped over the lazy dog.");
avg_fvu = np.mean([fvu for fvu in result.error_magnitudes])
total_l0 = np.sum([l0 for l0 in result.l0_per_layer])
avg_fvu, total_l0
#%%
out = result.mlp_outputs[2]
out.error.norm(dim=-1), out.original.norm(dim=-1)
#%%
