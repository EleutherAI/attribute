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
name = "gpt2-curt-clt-tied_per_target_skip_global_batchtopk_jumprelu"
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
for st_file in Path(cache_path).glob("**/*.safetensors"):
    loaded = load_file(st_file)
    vals = loaded["locations"][:, 2]
    start, end = map(int, st_file.stem.split("_"))
    print(st_file.parent.name, len(torch.unique(vals.long())) / (end - start))
# %%
