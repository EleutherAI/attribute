#%%
from IPython import get_ipython
if (ipy := get_ipython()) is not None:
    ipy.run_line_magic("load_ext", "autoreload")
    ipy.run_line_magic("autoreload", "2")
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# name = "k16-tied-filter-long"
# name = "k16-source-tied"
# name = "bs16-lr2e-4-tied-no-affine-ef128-k16-reset"
# name = "gpt2-curt-clt-tied_per_target_skip_global_batchtopk_jumprelu"
name = "gpt2-curt-clt-tied-per-target-layerwise-token-topk"
# weights_path = f"../e2e/checkpoints/gpt2-sweep/{name}"
weights_path = f"EleutherAI/{name}"
# weights_path = "../e2e/checkpoints/gpt2-sweep/bs16-lr2e-4-tied-no-affine-ef128-k16"
cache_path = f"results/gpt2-sweep-cache/{name}/latents"
render_to = "results/tied-comparison"
# %%
base_js_path = "http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/"
js_files = [
    "feature_examples/init-feature-examples-list.js",
    "feature_examples/init-feature-examples-logits.js",
    "feature_examples/init-feature-examples.js",
    "feature_examples/feature-examples.css",
    "util.js",
    "style.css",
    "feature-view.html",
]

for js_file in js_files:
    url = base_js_path + js_file
    local_path = os.path.join(render_to, js_file)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    import urllib.request
    print(f"Downloading {url} to {local_path}")
    urllib.request.urlretrieve(url, local_path)
    if js_file == "util.js":
        source = "https://d1fk9w8oratjix.cloudfront.net/"
        target = "./"
        with open(local_path, "r") as f:
            content = f.read()
        content = content.replace(source, target)
        with open(local_path, "w") as f:
            f.write(content)
#%%
from attribute.caching import TranscodedModel
model = TranscodedModel(
    "gpt2",
    weights_path,
)
#%%
from attribute.saving import cache_features, cache_contexts
for layer in range(12):
    nodes = []
    for feature in range(128):
        nodes.append((layer, feature))
    cache_features(model, render_to, nodes, scan=name + f"-{layer}", use_logit_bias=True, use_cantor=False)
    await cache_contexts(model, cache_path, render_to, nodes, scan=name + f"-{layer}", use_cantor=False)
# %%
import random
import webbrowser
import subprocess

PORT = random.randint(8000, 9000)
url = f"http://localhost:{PORT}/feature-view.html?model={name}-1"
try:
    pop = subprocess.Popen(["python", "-m", "http.server", str(PORT)], cwd=render_to)
    print(f"Serving at {url}")
    webbrowser.open(url)
    pop.wait()
finally:
    pop.terminate()

# %%


# %%
from pathlib import Path
from safetensors.torch import load_file
import torch
from natsort import natsorted
st_files = Path(cache_path).glob("**/*.safetensors")

for st_file in natsorted(st_files):
    loaded = load_file(st_file)
    vals = loaded["locations"][:, 2]
    start, end = map(int, st_file.stem.split("_"))
    print(st_file.parent.name, len(torch.unique(vals.long())) / (end - start))
# %%
# Group by file name: for each file name, collect (layer, locations) pairs across all model subfolders
from pathlib import Path
from safetensors.numpy import load_file
from collections import defaultdict
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

model_dirs = [d for d in Path(cache_path).iterdir() if d.is_dir()]
file_names = set()
for model_dir in model_dirs:
    file_names.update(f.name for f in model_dir.glob("*.safetensors"))

# For each file name, gather (layer, locations) pairs
file_locations = defaultdict(list)
for fname in file_names:
    for model_dir in model_dirs:
        st_file = model_dir / fname
        if st_file.exists():
            layer = model_dir.name
            loaded = load_file(st_file)
            file_locations[fname].append((layer, loaded["locations"]))
#%%
high_iou_density = []
for fname, layer_locs in file_locations.items():
    for feature_idx in trange(1000):
        total_intersections = defaultdict(int)
        total_counts = defaultdict(int)
        for layer1, locs1 in layer_locs:
            locs1 = locs1[locs1[:, 2] == feature_idx]
            total_counts[layer1] += locs1.shape[0]
            for layer2, locs2 in layer_locs:
                locs2 = locs2[locs2[:, 2] == feature_idx]
                l1idx = int(layer1.split(".")[1])
                l2idx = int(layer2.split(".")[1])
                if l1idx >= l2idx:
                    continue
                import numpy as np

                arr1 = locs1[:, :2].astype(np.uint64)
                arr2 = locs2[:, :2].astype(np.uint64)
                arr1 = arr1[:, 0] * 16384 + arr1[:, 1]
                arr2 = arr2[:, 0] * 16384 + arr2[:, 1]
                common = np.intersect1d(arr1, arr2, assume_unique=True)

                total_intersections[(layer1, layer2)] += len(common)

        iou_matrix = np.zeros((12, 12))
        for layer1, layer2 in total_intersections.keys():
            intersection = total_intersections[(layer1, layer2)]
            union = total_counts[layer1] + total_counts[layer2] - intersection
            l1idx = int(layer1.split(".")[1])
            l2idx = int(layer2.split(".")[1])
            if union == 0:
                continue
            iou_matrix[l1idx, l2idx] = intersection / union

        passes_filter = iou_matrix.max() > 0.1
        if not passes_filter:
            continue
        high_iou_density.append(feature_idx)
        plt.title(f"Feature {feature_idx}")
        plt.xlabel("Layer")
        plt.ylabel("Layer")
        plt.imshow(iou_matrix, vmin=0, vmax=1)
        plt.colorbar(label="IoU")
        plt.show()
    break

#%%
len(high_iou_density) / max(high_iou_density)
#%%
high_iou_density
