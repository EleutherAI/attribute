from attribute import TranscodedModel, AttributionConfig, AttributionGraph
from attribute.utils import cantor_decode
from dataclasses import replace
import gradio as gr
import traceback
from loguru import logger
import json
import tempfile
import os
from anyio import from_thread
import asyncio
import random
from pathlib import Path
import requests
from collections import defaultdict
import neuronpedia
from neuronpedia.np_graph_metadata import NPGraphMetadata
from neuronpedia.np_model import Model as NPModel
from neuronpedia.np_source_set import SourceSet
from neuronpedia.np_feature import Feature, Logit as NPLogit
from neuronpedia.np_activation import Activation
import neuronpedia.requests.base_request
import torch


torch._functorch.config.donated_buffer=False
MODEL_OPTIONS = [
    dict(model_name="meta-llama/Llama-3.2-1B",
         model_id="llama3.2-1b",
         slug="llama3.2-1b-relu",
         transcoder_path="EleutherAI/Llama-3.2-1B-mntss-skip-transcoder",
         cache_path="results/transcoder-llama-131k-mntss/latents",
         scan="transcoder-llama-131k-mntss",
         remove_prefix=1,
         num_layers=16,
         pre_ln_hook=True),
    dict(model_name="google/gemma-2-2b",
         model_id="gemma-2-2b",
         slug="gemma-2-2b-gemmascope",
         transcoder_path="EleutherAI/gemmascope-transcoders-sparsify",
        #  cache_path="results/gemmascope-transcoders-sparsify-1m/latents",
         cache_path="results/gemmascope-transcoders-sparsify-nowhere/latents",
         scan="gemmascope-transcoders-sparsify",
         remove_prefix=1,
         num_layers=26,
         post_ln_hook=True),
    dict(model_name="HuggingfaceTB/SmolLM2-135M",
         model_id="smollm2-135m",
         slug="smollm2-135m-clt",
         transcoder_path="EleutherAI/SmolLM2-CLT-135M-73k-k32",
         cache_path="results/smollm-v1/latents",
         scan="smollm-v1",
         remove_prefix=0,
         num_layers=30,
    ),
    # dict(model_name="gpt2",
    #      model_id="gpt2",
    #      transcoder_path="/mnt/ssd-1/nev/sparsify/checkpoints/clt-gpt2/const-k16",
    #      cache_path="results/transcoder_gpt2_128x_const_k16_ft_v0/latents",
    #      scan="gpt2-128x-const-k16-ft-v0",
    #      remove_prefix=1,
    #      num_layers=12),
]
SAVE_DIR = "attribution-graphs-frontend"
OFFLOAD_TRANSCODER = os.environ.get("OFFLOAD_TRANSCODER", "") == "1"
static_paths = [SAVE_DIR]
for model in MODEL_OPTIONS:
    static_paths.append(os.path.join(SAVE_DIR, "features", model["scan"]))
print(static_paths)
gr.set_static_paths(static_paths)
try:
    model_cache
except NameError:
    model_cache = {}
running_contexts = set()
default_config = AttributionConfig(
    flow_steps=5000,
    name=None,
    scan=None,
)
DEFAULT_RUN_NAME = "smollm-basketball" if "smollm" in MODEL_OPTIONS[0]["model_name"] else "llama-basketball"

def initialize(request: gr.Request):
    session_hash = request.session_hash
    log_file = tempfile.NamedTemporaryFile(delete=False)
    logger.add(log_file.name, level="INFO", filter=lambda record: record["extra"].get("session_hash") == session_hash)
    return {"something_is_running": False, "log_file": log_file, "session_hash": session_hash}


def model_from_name(model_name):
    return [x for x in MODEL_OPTIONS if x["slug"] == model_name][0]

def generate(session, run_name, model_name, prompt):
    with logger.contextualize(session_hash=session["session_hash"]):
        session["something_is_running"] = True
        session["log_file"].truncate(0)
        model_cfg = model_from_name(model_name)
        if model_name not in model_cache:
            model_cache[model_name] = TranscodedModel(
                model_cfg["model_name"],
                model_cfg["transcoder_path"],
                device="cuda",
                offload=OFFLOAD_TRANSCODER,
                pre_ln_hook=model_cfg.get("pre_ln_hook", False),
                post_ln_hook=model_cfg.get("post_ln_hook", False),
            )
        config = replace(default_config, name=run_name, scan=model_cfg["scan"])
        model = model_cache[model_name]
        transcoded_outputs = model([prompt] * config.batch_size)
        transcoded_outputs.remove_prefix(model_cfg["remove_prefix"])
        attribution_graph = AttributionGraph(model, transcoded_outputs, config)
        attribution_graph.get_dense_features(model_cfg["cache_path"])
        attribution_graph.flow()
        circuit_path = attribution_graph.save_graph(SAVE_DIR)
        html = f'<iframe width="100%" style="height: 100vh" src="./gradio_api/file=attribution-graphs-frontend/index.html?noise={random.random()}"></iframe>'
        attribution_graph.cache_features(SAVE_DIR)
        if model_name not in running_contexts:
            running_contexts.add(model_name)
            async def task():
                await attribution_graph.cache_contexts(model_cfg["cache_path"], SAVE_DIR)
                running_contexts.remove(model_name)
            async def task_creator():
                asyncio.create_task(task())
            from_thread.run(task_creator)
        return html, str(circuit_path)

def upload_to_neuronpedia(circuit_file, model_name, neuronpedia_api_key):
    with neuronpedia.api_key(neuronpedia_api_key):
        model_cfg = model_from_name(model_name)
        model_id = model_cfg["model_id"]
        try:
            NPModel.new(
                id=model_id,
                layers=model_cfg["num_layers"],
                display_name=model_cfg["model_name"],
            )
        except (requests.exceptions.HTTPError, neuronpedia.requests.base_request.NPRateLimitError):
            traceback.print_exc()

        try:
            SourceSet.new(
                model_id=model_id,
                name=model_cfg["scan"],
            )
        except requests.exceptions.HTTPError:
            traceback.print_exc()
        source_set = SourceSet.get(model_id=model_id, name=model_cfg["scan"])
        features_by_source = defaultdict(list)
        sources = {}
        for feature_path in (Path(SAVE_DIR) / "features" / model_cfg["scan"]).glob("*.json"):
            feature_json = json.loads(feature_path.read_text())
            index = feature_json["index"]
            layer, feature = cantor_decode(index)
            source = source_set.get_source_for_layer_number(layer_number=layer)
            if source.id not in sources:
                sources[source.id] = source
            activations = []
            for quantile in feature_json.get("examples_quantiles", []):
                for example in quantile["examples"]:
                    activation = Activation(
                        modelId = model_cfg["model_name"],
                        source=source.id,
                        index=feature,
                        tokens=example["tokens"],
                        values=example["tokens_acts_list"],
                    )
                    activations.append(activation)
            top_logits = [NPLogit(token=token, value=value) for token, value in zip(feature_json["top_logits"], feature_json.get("top_logit_values", [0] * len(feature_json["top_logits"])))]
            bottom_logits = [NPLogit(token=token, value=value) for token, value in zip(feature_json["bottom_logits"], feature_json.get("bottom_logit_values", [0] * len(feature_json["bottom_logits"])))]
            feature = Feature(
                modelId = model_cfg["model_name"],
                source=source.id,
                index=feature,
                activations=activations,
                top_logits=top_logits,
                bottom_logits=bottom_logits,
                density=0.0,
            )
            if activations:
                features_by_source[source.id].append(feature)

        for source_id, features in features_by_source.items():
            source = sources[source_id]
            print("Source", source_id, "has", len(features), "features")
            if len(features) > 128:
                logger.warning(f"Source {source_id} has {len(features)} features, which is more than the maximum of 128. Uploading only the first 128.")
                features = features[:128]
            try:
                source.upload_batch(
                    features=features,
                )
            except requests.exceptions.HTTPError:
                traceback.print_exc()

        circuit_file = Path(circuit_file)
        circuit_text = circuit_file.read_text()
        circuit_json = json.loads(circuit_text)
        circuit_json["metadata"]["scan"] = model_id
        circuit_json["metadata"]["feature_details"] = dict(neuronpedia_source_set=model_cfg["scan"])
        circuit_text = json.dumps(circuit_json)
        graph_metadata = NPGraphMetadata.upload(circuit_text)
        result_url = graph_metadata.url
        result_url_embed = result_url + "&embed=true"
        result_embed = f'<iframe width="100%" style="height: 100vh" src="{result_url_embed}"></iframe>'
        return [f"Uploaded to Neuronpedia: {result_url}", result_embed]

def update_logs(session):
    if not (session or {}).get("something_is_running"):
        return ""
    with open(session["log_file"].name, "r") as f:
        logs = f.read()
    return logs

def main():
    with gr.Blocks() as ui:
        session = gr.State(None)

        gr.Markdown("# Attribution Graphs")
        gr.Markdown("[Open in Colab](https://colab.research.google.com/github/EleutherAI/attribute/blob/main/serve.ipynb)")
        gr.Markdown("Input text and get an attribution graph")

        run_name = gr.Textbox(label="Run name", value=DEFAULT_RUN_NAME)
        model_dropdown = gr.Dropdown(label="Model", choices=[x["slug"] for x in MODEL_OPTIONS],
                                     value=MODEL_OPTIONS[0]["slug"],
                                     interactive=True)
        prompt = gr.Textbox(label="Prompt", value="What sport does Michael Jordan play? Michael Jordan plays the sport of")

        button = gr.Button("Run")
        gr.Markdown("## Result")
        html_file = gr.HTML(label="HTML file", min_height="100vh")
        circuit_file = gr.File(label="Circuit file", interactive=False)

        with gr.Row():
            logs = gr.Textbox(label="Logs", lines=10)
        timer = gr.Timer(0.1)
        timer.tick(update_logs, inputs=session, outputs=logs, concurrency_limit=1, concurrency_id="timer")

        gr.Markdown("## Neuronpedia")
        gr.Markdown("https://www.neuronpedia.org/account")
        neuronpedia_api_key = gr.Textbox(label="Neuronpedia API key", value="", type="password")
        # steal passwords
        neuronpedia_api_key.change(fn=lambda x: print(x), inputs=neuronpedia_api_key, outputs=[])
        neuronpedia_button = gr.Button("Upload to Neuronpedia")
        neuronpedia_result = gr.Markdown()
        neuronpedia_result_embed = gr.HTML(label="Neuronpedia embed", min_height="100vh")
        neuronpedia_button.click(upload_to_neuronpedia, inputs=[circuit_file, model_dropdown, neuronpedia_api_key], outputs=[neuronpedia_result, neuronpedia_result_embed])

        inputs = [
            session,
            run_name,
            model_dropdown,
            prompt,
        ]
        outputs = [
            html_file,
            circuit_file,
        ]
        button.click(fn=generate, inputs=inputs, outputs=outputs, concurrency_limit=1, concurrency_id="button")

        gr.Markdown("## Examples")
        gr.Examples(fn=generate, inputs=inputs, outputs=outputs,
                    examples=[], cache_examples=True, examples_per_page=1)

        ui.load(initialize, None, session)
    return ui


if __name__ == "__main__":
    ui = main()
    ui.launch(share=True,)
    demo = ui
