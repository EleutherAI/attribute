import os
from pathlib import Path
import json
from collections import defaultdict

import torch
from loguru import logger
from tqdm.auto import tqdm
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset

from .utils import cantor
from .caching import TranscodedModel


def make_latent_dataset(model: TranscodedModel, cache_path: os.PathLike, module_latents: dict[str, torch.Tensor]):
    if model.pre_ln_hook and module_latents:
        module_latents = {k.replace(".mlp", f".{model.mlp_layernorm_name}"): v for k, v in module_latents.items()}
    return LatentDataset(
        cache_path,
        SamplerConfig(n_examples_train=10, train_type="top", n_examples_test=0),
        ConstructorConfig(center_examples=True, example_ctx_len=16, n_non_activating=0),
        modules=list(module_latents.keys()) if module_latents else None,
        latents=module_latents,
    )


def cache_features(model: TranscodedModel, save_dir: os.PathLike, nodes: list[int, int], *, scan: str, use_logit_bias: bool = False, use_cantor: bool = True):
    save_dir = Path(save_dir)
    logit_weight = model.logit_weight
    logit_bias = model.logit_bias
    for layer, feature in tqdm(nodes, desc="Caching features"):
        feature_dir = save_dir / "features" / scan

        with torch.no_grad(), torch.autocast("cuda"):
            try:
                logger.disable("attribute.caching")
                dec_weight = model.w_dec_i(layer, feature)
            finally:
                logger.enable("attribute.caching")
            logits = logit_weight @ dec_weight
            del dec_weight
            if use_logit_bias:
                logits += logit_bias
            top_logits = logits.topk(10).indices.tolist()
            bottom_logits = logits.topk(10, largest=False).indices.tolist()
        top_logits = [model.decode_token(i) for i in top_logits]
        bottom_logits = [model.decode_token(i) for i in bottom_logits]

        feat_idx = cantor(layer, feature) if use_cantor else feature
        feature_vis = dict(
            index=feat_idx,
            bottom_logits=bottom_logits,
            top_logits=top_logits,
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_path = feature_dir / f"{feat_idx}.json"
        if feature_path.exists():
            try:
                feature_vis = json.loads(feature_path.read_text()) | feature_vis
            except json.JSONDecodeError:
                pass
        feature_path.write_text(json.dumps(feature_vis))


async def cache_contexts(
    model: TranscodedModel,
    cache_path: os.PathLike,
    save_dir: os.PathLike,
    nodes: list[int, int],
    *,
    scan: str,
    use_cantor: bool = True,
):
    cache_path = Path(cache_path)
    save_dir = Path(save_dir)
    feature_paths = {}
    module_latents = defaultdict(list)
    dead_features = set()
    for layer_idx, feature_idx in nodes:
        feature_dir = save_dir / "features" / scan
        feature_dir.mkdir(parents=True, exist_ok=True)
        feat_idx = cantor(layer_idx, feature_idx) if use_cantor else feature_idx
        feature_path = feature_dir / f"{feat_idx}.json"
        if feature_path.exists():
            if "examples_quantiles" in json.loads(feature_path.read_text()):
                continue
        feature_paths[(layer_idx, feature_idx)] = feature_path
        module_latents[model.temp_hookpoints_mlp[layer_idx]].append(feature_idx)
        dead_features.add((layer_idx, feature_idx))

    module_latents = {k: torch.tensor(v) for k, v in module_latents.items()}
    module_latents = {k: v[torch.argsort(v)] for k, v in module_latents.items()}

    ds = make_latent_dataset(model, cache_path, module_latents)

    bar = tqdm(total=sum(map(len, module_latents.values())))
    def process_feature(feature):
        layer_idx = int(feature.latent.module_name.split(".")[-2])
        feature_idx = feature.latent.latent_index
        dead_features.discard((layer_idx, feature_idx))

        feature_path = feature_paths[(layer_idx, feature_idx)]

        if not feature_path.exists():
            logger.warning(f"Feature L{layer_idx}-{feature_idx} does not exist")
            return
        feature_vis = json.loads(feature_path.read_text())
        examples_quantiles = feature_vis.get("examples_quantiles", None)

        if examples_quantiles is None:
            examples_quantiles = defaultdict(list)
            for example in feature.train:
                examples_quantiles[example.quantile].append(dict(
                    is_repeated_datapoint=False,
                    train_token_index=len(example.tokens) - 1,
                    tokens=[model.decode_token(i) for i in example.tokens.tolist()],
                    tokens_acts_list=example.activations.tolist(),
                ))
            examples_quantiles = [
                dict(
                    quantile_name=f"Quantile {i}",
                    examples=examples_quantiles[i],
                ) for i in sorted(examples_quantiles.keys())
            ]
        feature_vis["examples_quantiles"] = examples_quantiles

        feature_path.write_text(json.dumps(feature_vis))
        bar.update(1)
        bar.refresh()

    async for feature in ds:
        process_feature(feature)
    bar.close()

    if len(dead_features) > 0:
        dead_features = sorted(dead_features)
        logger.info(f"Dead features: {dead_features[:10]}{'...' if len(dead_features) > 10 else ''}")
