import gc
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
from jaxtyping import Array, Float, Int
from sparsify import SparseCoder, CrossLayerRunner
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from transformers.models.llama import LlamaPreTrainedModel
from transformers.models.gpt_neo import GPTNeoPreTrainedModel
from transformers.models.gpt2 import GPT2PreTrainedModel
from transformers.models.qwen2 import Qwen2PreTrainedModel
from transformers.models.gemma2 import Gemma2PreTrainedModel
from loguru import logger


LlamaLike = LlamaPreTrainedModel | Qwen2PreTrainedModel | Gemma2PreTrainedModel
GPT2Like = GPT2PreTrainedModel | GPTNeoPreTrainedModel


@dataclass
class MLPOutputs:
    activation: Float[Array, "batch seq_len k"]
    source_activation: Float[Array, "batch seq_len k"]
    location: Int[Array, "batch seq_len k"]
    error: Float[Array, "batch seq_len hidden_size"]
    source_error: Float[Array, "batch seq_len hidden_size"]
    l0: float


@dataclass
class TranscodedOutputs:
    input_ids: Int[Array, "batch seq_len"]
    original_input_ids: Int[Array, "batch seq_len"]
    mlp_outputs: dict[int, MLPOutputs]
    last_layer_activations: Float[Array, "batch seq_len hidden_size"]
    first_layer_activations: Float[Array, "batch seq_len hidden_size"]
    logits: Float[Array, "batch seq_len vocab_size"]

    error_magnitudes: list[float]
    l0_per_layer: list[float]

    @property
    def seq_len(self):
        return self.input_ids.shape[1]

    @property
    def batch_size(self):
        return self.input_ids.shape[0]

    def remove_prefix(self, remove_prefix: int):
        if remove_prefix > 0:
            self.input_ids = self.input_ids[:, remove_prefix:]
            # only ever accessed w/ [-1], removing BOS doesn't matter
            # transcoded_outputs.last_layer_activations = transcoded_outputs.last_layer_activations[:, 1:]
            self.logits = self.logits[:, remove_prefix:]
            for k, mlp_output in self.mlp_outputs.items():
                mlp_output.activation = mlp_output.activation[:, remove_prefix:]
                mlp_output.location = mlp_output.location[:, remove_prefix:]
                mlp_output.error = mlp_output.error[:, remove_prefix:]
                assert mlp_output.location.shape[1] == self.input_ids.shape[1]
                # we don't remove BOS from source nodes because we take gradients to them


class TranscodedModel(object):
    @torch.no_grad()
    def __init__(
        self,
        model_name: str | os.PathLike | PreTrainedModel,
        transcoder_path: os.PathLike,
        hookpoint_fn=None,
        device="cuda",
        pre_ln_hook: bool = False,
        post_ln_hook: bool = False,
        offload: bool = False,
    ):
        logger.info(f"Loading model {model_name} on device {device}")
        self.device = device
        if isinstance(model_name, PreTrainedModel):
            model = model_name
            model_name = model.name_or_path
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": device},
                torch_dtype=torch.bfloat16,
            )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.offload = offload

        if transcoder_path is None:
            logger.warning("No transcoder path provided. Returning model without hooks.")
            return
        if hookpoint_fn is None:
            def hookpoint_fn(hookpoint):
                if isinstance(model, LlamaLike):
                    return hookpoint.replace("model.layers.", "layers.")
                elif isinstance(model, GPT2Like):
                    return hookpoint.replace("transformer.h.", "h.")
                else:
                    logger.warning(f"Unknown model type: {type(model)}. Using default hookpoint.")
                    return hookpoint
        self.hookpoints_mlp = [f"{self.layer_prefix}.{i}.mlp" for i in range(self.num_layers)]
        if post_ln_hook and isinstance(model, Gemma2PreTrainedModel):
            self.hookpoints_mlp_post = [
                f"{self.layer_prefix}.{i}.post_feedforward_layernorm"
                for i in range(self.num_layers)
            ]
        else:
            self.hookpoints_mlp_post = self.hookpoints_mlp
        self.temp_hookpoints_mlp = [
            hookpoint_fn(hookpoint) for hookpoint in self.hookpoints_mlp
        ]
        logger.info(f"Loading transcoders from {transcoder_path}")
        transcoder_path = Path(transcoder_path)
        self.transcoders = {}
        self.transcoder_loaders = {}
        self.offloaded_encoder_indices = {}
        self.offloaded_decoder_indices = {}
        def load_transcoder(hookpoint, temp_hookpoint):
            if not transcoder_path.joinpath(temp_hookpoint).exists():
                sae = SparseCoder.load_from_hub(
                    str(transcoder_path),
                    temp_hookpoint,
                    device=device,
                )
            else:
                sae = SparseCoder.load_from_disk(
                    transcoder_path / temp_hookpoint,
                    device=device,
                )
            sae.requires_grad_(False)
            self.transcoders[hookpoint] = sae
        for hookpoint, temp_hookpoint in zip(self.hookpoints_mlp, self.temp_hookpoints_mlp):
            loader = partial(load_transcoder, hookpoint, temp_hookpoint)
            if offload:
                self.transcoder_loaders[hookpoint] = loader
            else:
                loader()
        self.hookpoints_layer = [f"{self.layer_prefix}.{i}" for i in range(self.num_layers)]
        self.hookpoints_ln = [f"{self.layer_prefix}.{i}.{self.mlp_layernorm_name}" for i in range(self.num_layers)]
        self.hookpoints_attn_ln = [f"{self.layer_prefix}.{i}.{self.attn_layernorm_name}" for i in range(self.num_layers)]
        self.additional_ln = []
        if isinstance(model, Gemma2PreTrainedModel):
            self.additional_ln += [
                f"{self.layer_prefix}.{i}.{suffix}"
                for i in range(self.num_layers)
                for suffix in ["post_feedforward_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm"]
            ]
        self.name_to_module = {
            name: model.get_submodule(name) for name in self.hookpoints_layer + self.hookpoints_mlp + self.hookpoints_ln + self.hookpoints_attn_ln + self.additional_ln
        }
        self.name_to_index = {
            k: i
            for arr in [self.hookpoints_layer, self.hookpoints_mlp, self.hookpoints_ln, self.hookpoints_mlp_post]
            for i, k in enumerate(arr)
        }
        self.module_to_name = {v: k for k, v in self.model.named_modules()}
        self.pre_ln_hook = pre_ln_hook
        self.post_ln_hook = post_ln_hook

    def clear_hooks(self):
        for mod in self.model.modules():
            mod._forward_hooks = OrderedDict()

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def repeat_kv(self):
        if not hasattr(self.model.config, "num_key_value_heads"):
            return 1
        return self.model.config.num_attention_heads // self.model.config.num_key_value_heads

    def __call__(self, prompt: str | list[str] | torch.Tensor,
                 mask_features: dict[int, list[int]] = {},
                 steer_features: dict[int, list[(int, int, float)]] = {},
                 errors_from: TranscodedOutputs | None = None,
                 latents_from_errors: bool = False,
                 no_error: Literal[False, "none", "all", "after_first"] = "none"
                ) -> TranscodedOutputs:
        if isinstance(prompt, str):
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            logger.info(f"Tokenized prompt: {[self.decode_token(i) for i in tokenized_prompt.input_ids[0]]}")
        elif isinstance(prompt, list):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        elif isinstance(prompt, torch.Tensor):
            tokenized_prompt = SimpleNamespace(input_ids=prompt.to(self.device))
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        self.clear_hooks()

        def ln_record_hook(module, input, output):
            if "rmsnorm" in type(module).__name__.lower():
                mean = 0
            else:
                mean = input[0].mean(dim=-1, keepdim=True).detach()
            var = input[0].pow(2).mean(dim=-1, keepdim=True)
            scale = torch.rsqrt(var + getattr(module, "eps", 1e-5))
            scale = scale.detach()
            weight = module.weight
            if isinstance(self.model, Gemma2PreTrainedModel):
                weight = weight + 1
            multiplier = scale * weight
            return (input[0] - mean) * multiplier + getattr(module, "bias", 0)

        for hookpoint in self.hookpoints_layer:
            ln = getattr(self.name_to_module[hookpoint], self.attn_layernorm_name)
            ln.register_forward_hook(ln_record_hook)
            self.freeze_attention_pattern(hookpoint)

        for hookpoint in self.additional_ln:
            self.name_to_module[hookpoint].register_forward_hook(ln_record_hook)

        for hookpoint in self.hookpoints_ln:
            self.name_to_module[hookpoint].register_forward_hook(ln_record_hook)

        if (final_ln := self.final_ln) is not None:
            final_ln.register_forward_hook(ln_record_hook)

        if self.pre_ln_hook:
            resid_mid = {}
            def record_resid_mid(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                resid_mid[self.name_to_index[self.module_to_name[module]]] = input
            for hookpoint in self.hookpoints_ln:
                self.name_to_module[hookpoint].register_forward_hook(record_resid_mid)
        elif self.post_ln_hook:
            resid_mid = {}
            def record_resid_mid(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                resid_mid[self.name_to_index[self.module_to_name[module]]] = input
            for hookpoint in self.hookpoints_mlp:
                self.name_to_module[hookpoint].register_forward_hook(record_resid_mid)
        runner = CrossLayerRunner()
        target_transcoder_activations = {}
        source_transcoder_activations = {}
        l0s = {}
        transcoder_locations = {}
        errors = {}
        error_magnitudes = []
        l0s_per_layer = []

        def get_mlp_hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]

            # change module name in case we're reading from the post ff ln instead of the mlp
            module_name = self.module_to_name[module]
            layer_idx = self.name_to_index[module_name]
            module_name = self.hookpoints_mlp[layer_idx]
            if self.pre_ln_hook or self.post_ln_hook:
                input = resid_mid[layer_idx]
            output = output.detach()

            if self.offload:
                if module_name in self.transcoders:
                    del self.transcoders[module_name]
                self.transcoder_loaders[module_name]()
            transcoder = self.transcoders[module_name]
            # have to reshape input to lose the batch dimension
            batch_dims = input.shape[:-1]
            input = input.view(-1, input.shape[-1])
            with torch.set_grad_enabled(not self.offload):
                transcoder_acts = runner.encode(input, transcoder)
            if self.offload:
                pad_to = 256

                with torch.no_grad():
                    unique_ids = torch.unique(transcoder_acts.latent_indices.view(-1))
                    self.offloaded_encoder_indices[layer_idx] = unique_ids.tolist()
                    remapped_indices = (transcoder_acts.latent_indices[..., None] == unique_ids).int().argmax(dim=-1)
                    pad_amount = (pad_to - unique_ids.shape[0] % pad_to) % pad_to
                    unique_ids_padded = torch.nn.functional.pad(unique_ids, (0, pad_amount))
                    transcoder.encoder.weight.data = transcoder.encoder.weight.data[unique_ids_padded].clone()
                    new_bias = transcoder.encoder.bias.data[unique_ids_padded].clone()
                    inf_mask = torch.nn.functional.pad(torch.ones_like(unique_ids, dtype=torch.bool), (0, pad_amount))
                    new_bias = torch.where(inf_mask, new_bias, new_bias.min() - 1)
                    transcoder.encoder.bias.data = new_bias
                    gc.collect()
                    torch.cuda.empty_cache()
                second_acts = runner.encode(input, transcoder)
                with torch.no_grad():
                    matching_indices = remapped_indices[..., :, None] == second_acts.latent_indices[..., None, :]
                    source_index = matching_indices.half().argmax(dim=-1)
                    has_match = torch.any(matching_indices, dim=-1)
                gathered_acts = torch.gather(second_acts.latent_acts, dim=-1, index=source_index) * has_match.float()
                transcoder_acts.latent_acts = gathered_acts

            masked_features = mask_features.get(layer_idx, [])
            steered_features = steer_features.get(layer_idx, [])
            if latents_from_errors:
                act = errors_from.mlp_outputs[layer_idx].activation
                transcoder_acts.latent_acts = act.view(-1, act.shape[-1])
                loc = errors_from.mlp_outputs[layer_idx].location
                transcoder_acts.latent_indices = loc.view(-1, loc.shape[-1])
            if masked_features:
                acts = transcoder_acts.latent_acts
                indices = transcoder_acts.latent_indices
                # TODO: when patching, we can't use automatic attribution
                transcoder_acts.latent_acts = acts * (1 - torch.any(torch.stack([indices == i for i in masked_features], dim=0), dim=0).float())
            if steered_features:
                acts = transcoder_acts.latent_acts
                indices = transcoder_acts.latent_indices
                acts = acts.view(*batch_dims, -1)
                indices = indices.view(*batch_dims, -1)
                for seq_idx, feature, strength in steered_features:
                    prev_activations = acts[0, seq_idx].tolist()
                    if 0.0 not in prev_activations:
                        acts = torch.nn.functional.pad(acts, (0, 1))
                        indices = torch.nn.functional.pad(indices, (0, 1))
                        acts[:, seq_idx, -1] = strength
                        indices[:, seq_idx, -1] = feature
                    else:
                        zero_index = prev_activations.index(0.0)
                        acts[:, seq_idx, zero_index] = strength
                        indices[:, seq_idx, zero_index] = feature
                acts = acts.view(-1, acts.shape[-1])
                indices = indices.view(-1, indices.shape[-1])
                transcoder_acts.latent_acts = acts
                transcoder_acts.latent_indices = indices

            l0 = (transcoder_acts.latent_acts.unflatten(0, batch_dims)[:, 1:] != 0).float().sum(dim=-1).mean().item()
            l0s[module_name] = l0
            target_latent_acts = transcoder_acts.latent_acts.clone().unflatten(0, batch_dims)
            source_latent_acts = transcoder_acts.latent_acts
            source_latent_acts = source_latent_acts.unflatten(0, batch_dims).clone()
            source_latent_acts.detach_()
            source_latent_acts.requires_grad_(True)
            flat_source_acts = source_latent_acts.flatten(0, -2)
            transcoder_acts.latent_acts = flat_source_acts

            with torch.set_grad_enabled(not self.offload):
                outputs = runner.decode(transcoder_acts, None, module_name, advance=not self.offload)
            if self.offload:
                with torch.no_grad():
                    unique_decoder_indices = torch.unique(outputs.latent_indices.view(-1))
                    self.offloaded_decoder_indices[layer_idx] = unique_decoder_indices.tolist()
                    pad_amount = (pad_to - unique_decoder_indices.shape[0] % pad_to) % pad_to
                    unique_decoder_indices_padded = torch.nn.functional.pad(unique_decoder_indices, (0, pad_amount))
                    transcoder.W_dec.data = transcoder.W_dec.data[unique_decoder_indices_padded]
                    original_fn = transcoder.decode
                    def decode_remapped(top_acts, top_indices, *args, **kwargs):
                        remapped_indices = (top_indices[..., None] == unique_decoder_indices).int().argmax(dim=-1)
                        return original_fn(top_acts, remapped_indices, *args, **kwargs)
                    transcoder.decode = decode_remapped
                outputs = runner.decode(transcoder_acts, None, module_name)
                gc.collect()
                torch.cuda.empty_cache()
            sae_out = outputs.sae_out

            # have to reshape output to get the batch dimension back
            transcoder_out = sae_out.view(output.shape)
            diff = output - transcoder_out

            if no_error not in (False, "none"):
                if no_error == "after_first":
                    error = diff.clone()
                    error[:, 1:] = 0
                else:
                    error = torch.zeros_like(output)
            elif errors_from is None:
                error = diff
            else:
                error = errors_from.mlp_outputs[layer_idx].error

            error = error.clone()
            error.detach_()
            error.requires_grad_(True)
            fvu_approx = diff[:, 1:].pow(2).sum() / output[:, 1:].pow(2).sum()
            logger.info(f"Layer {module_name} error: {fvu_approx.item()} L0: {l0}")
            error_magnitudes.append(fvu_approx.item())
            l0s_per_layer.append(l0)

            latent_indices = transcoder_acts.latent_indices.unflatten(0, batch_dims)
            target_transcoder_activations[module_name] = target_latent_acts
            source_transcoder_activations[module_name] = source_latent_acts
            transcoder_locations[module_name] = latent_indices

            result = (transcoder_out + error).to(output)
            # if errors_from is None and not no_error:
                # result = output.detach() + (result - result.detach())
            errors[module_name] = error
            return result

        for hookpoint in self.hookpoints_mlp_post:
            self.name_to_module[hookpoint].register_forward_hook(get_mlp_hook)

        outputs = self.model(input_ids=tokenized_prompt.input_ids,
                            #  attention_mask=tokenized_prompt.attention_mask,
                             output_hidden_states=True)
        self.clear_hooks()

        logits = outputs.logits

        logger.info("Top last token logits:")
        for index in logits[0, -1].topk(10).indices:
            logger.info(f"{self.decode_token(index)}: {logits[0, -1][index].item()}")

        first_layer_activations = outputs.hidden_states[0]
        if first_layer_activations.requires_grad:
            first_layer_activations.retain_grad()

        last_layer_activations = outputs.hidden_states[-1]
        if last_layer_activations.requires_grad:
            last_layer_activations.retain_grad()

        runner.reset()

        mlp_outputs = {}
        for i in range(self.num_layers):
            mlp_outputs[i] = MLPOutputs(
                activation=target_transcoder_activations[self.hookpoints_mlp[i]],
                source_activation=source_transcoder_activations[self.hookpoints_mlp[i]],
                location=transcoder_locations[self.hookpoints_mlp[i]],
                error=errors[self.hookpoints_mlp[i]],
                source_error=errors[self.hookpoints_mlp[i]],
                l0=l0s[self.hookpoints_mlp[i]],
            )

        transcoded_outputs = TranscodedOutputs(
            input_ids=tokenized_prompt.input_ids,
            original_input_ids=tokenized_prompt.input_ids,
            mlp_outputs=mlp_outputs,
            first_layer_activations=first_layer_activations,
            last_layer_activations=last_layer_activations,
            logits=logits,
            error_magnitudes=error_magnitudes,
            l0_per_layer=l0s_per_layer,
        )

        return transcoded_outputs

    @property
    def layer_prefix(self):
        if isinstance(self.model, LlamaLike):
            return "model.layers"
        elif isinstance(self.model, GPT2Like):
            return "transformer.h"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def attn_layernorm_name(self):
        if isinstance(self.model, LlamaLike):
            return "input_layernorm"
        elif isinstance(self.model, GPT2Like):
            return "ln_1"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def mlp_layernorm_name(self):
        if isinstance(self.model, Gemma2PreTrainedModel):
            return "pre_feedforward_layernorm"
        elif isinstance(self.model, LlamaLike):
            return "post_attention_layernorm"
        elif isinstance(self.model, GPT2Like):
            return "ln_2"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def mlp_in_proj_name(self):
        if isinstance(self.model, GPT2Like):
            return "c_proj"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
    @property
    def embedding_module(self):
        if isinstance(self.model, LlamaLike):
            return self.model.model.embed_tokens
        elif isinstance(self.model, GPT2Like):
            return self.model.transformer.wte
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def embedding_weight(self):
        return self.embedding_module.weight

    @property
    def parallel_attn(self):
        return isinstance(self.model, GPTNeoPreTrainedModel)

    @property
    def final_ln(self):
        if isinstance(self.model, LlamaLike):
            return self.model.model.norm
        elif isinstance(self.model, GPT2Like):
            return self.model.transformer.ln_f
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def logit_weight(self):
        if isinstance(self.model, LlamaLike):
            return self.model.lm_head.weight * self.final_ln.weight
        elif isinstance(self.model, GPT2Like):
            return self.model.lm_head.weight * self.final_ln.weight
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def logit_bias(self):
        bias = self.model.lm_head.bias
        if bias is None:
            bias = 0
        if hasattr(self.final_ln, "bias"):
            bias = bias + self.logit_weight @ self.final_ln.bias
        return bias

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def num_attention_heads(self):
        return self.model.config.num_attention_heads

    @property
    def head_dim(self):
        return self.model.config.hidden_size // self.model.config.num_attention_heads

    @torch.no_grad()
    @torch.autocast("cuda")
    def w_dec(self, layer_idx: int, target_layer_idx: int | None = None, use_skip: bool = False) -> Float[Array, "features hidden_size"]:
        sparse_coder = self.transcoders[self.hookpoints_mlp[layer_idx]]
        try:
            if target_layer_idx is not None and target_layer_idx != layer_idx:
                raise AttributeError
            w_dec = sparse_coder.W_dec.clone()
            if use_skip:
                for skip_layer_idx in range(layer_idx + 1, self.num_layers):
                    w_dec += w_dec @ self.w_skip(skip_layer_idx, skip_layer_idx).T
            return w_dec
        except AttributeError:
            if target_layer_idx is None:
                logger.warning("Summing decoder weights because target_layer_idx is None")
                target_layer_idx = layer_idx
                weight_combined = torch.zeros((self.w_dec(layer_idx, layer_idx).shape[0], self.hidden_size,), device=self.device, dtype=torch.float32)
                for target_layer_idx in range(layer_idx, self.num_layers):
                    try:
                        weight_combined += self.w_dec(layer_idx, target_layer_idx)
                    except IndexError:
                        break
                # weights_at_layers = {}
                # while target_layer_idx < self.num_layers:
                #     weights_at_layers[target_layer_idx] = weight_combined
                #     for layer_from, weight_at in weights_at_layers.items():
                #         try:
                #             weight_combined += weight_at @ self.w_skip(layer_from, target_layer_idx).T
                #         except IndexError:
                #             pass
                #     try:
                #         weight_combined += self.w_dec(layer_idx, target_layer_idx)
                #     except IndexError:
                #         pass
                #     target_layer_idx += 1
                return weight_combined
            # assume the target layer is contiguous
            assert target_layer_idx >= layer_idx
            try:
                return self.transcoders[self.hookpoints_mlp[layer_idx]].W_decs[target_layer_idx - layer_idx]
            except AttributeError:
                raise IndexError

    @torch.no_grad()
    @torch.autocast("cuda")
    def w_dec_i(self, layer_idx: int, i: int, target_layer_idx: int | None = None, use_skip: bool = False) -> Float[Array, "hidden_size"]:
        sparse_coder = self.transcoders[self.hookpoints_mlp[layer_idx]]
        if self.offload:
            if use_skip:
                logger.warning("Skipping weights are not supported when offloading")
            if not hasattr(sparse_coder, "W_dec"):
                raise NotImplementedError("Multiple decoder weights are not supported when offloading")
            W_dec = list(sparse_coder.W_dec)
            offloaded_decoder_indices = self.offloaded_decoder_indices[layer_idx]
            return W_dec[offloaded_decoder_indices.index(i)]
        return self.w_dec(layer_idx, target_layer_idx, use_skip)[i]
    def w_skip(self, layer_idx: int, target_layer_idx: int | None = None) -> Float[Array, "hidden_size hidden_size"]:
        transcoder = self.transcoders[self.hookpoints_mlp[layer_idx]]
        try:
            transcoder.W_skip
        except AttributeError:
            assert target_layer_idx is not None, "target_layer_idx must be provided for multi-target transcoders"
            assert target_layer_idx >= layer_idx
            try:
                w_skip = transcoder.W_skips[target_layer_idx - layer_idx]
            except AttributeError:
                raise IndexError
            if w_skip is None:
                raise IndexError
            return w_skip
        else:
            if target_layer_idx != layer_idx:
                raise IndexError
            return transcoder.W_skip

    def w_enc(self, layer_idx: int) -> Float[Array, "features hidden_size"]:
        transcoder = self.transcoders[self.hookpoints_mlp[layer_idx]]
        W_enc = transcoder.encoder.weight
        return W_enc

    def w_enc_i(self, layer_idx: int, i: int) -> Float[Array, "features hidden_size"]:
        W_enc = self.w_enc(layer_idx)
        if self.offload:
            offloaded_encoder_indices = self.offloaded_encoder_indices[layer_idx]
            return W_enc[offloaded_encoder_indices.index(i)]
        return W_enc[i]

    def get_layer(self, layer_idx: int) -> torch.nn.Module:
        return self.model.get_submodule(self.layer_prefix)[layer_idx]

    def attn(self, layer_idx: int) -> torch.nn.Module:
        layer = self.get_layer(layer_idx)
        if isinstance(self.model, LlamaLike):
            return layer.self_attn
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return layer.attn.attention
        elif isinstance(self.model, GPT2PreTrainedModel):
            return layer.attn
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def attn_output(self, layer_idx: int) -> Float[Array, "hidden_size num_attention_heads head_dim"]:
        if isinstance(self.model, LlamaLike):
            w_o = self.attn(layer_idx).o_proj.weight
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            w_o = self.attn(layer_idx).out_proj.weight
        elif isinstance(self.model, GPT2PreTrainedModel):
            w_o = self.attn(layer_idx).c_proj.weight.T
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        return w_o.reshape(self.hidden_size, self.num_attention_heads, self.head_dim)

    def attn_value(self, layer_idx: int) -> Float[Array, "num_attention_heads head_dim hidden_size"]:
        if not isinstance(self.model, GPT2PreTrainedModel):
            w_v = self.attn(layer_idx).v_proj.weight
        else:
            w_q, w_k, w_v = torch.split(self.attn(layer_idx).c_attn.weight, self.hidden_size, dim=1)
            w_v = w_v.T
        w_v = torch.repeat_interleave(w_v, self.repeat_kv, dim=0)
        return w_v.reshape(self.num_attention_heads, self.head_dim, self.hidden_size)

    def project_v(self, layer_idx: int, layer_normed: Float[Array, "batch seq_len hidden_size"]) -> Float[Array, "batch seq_len num_attention_heads head_dim"]:
        if not isinstance(self.model, GPT2PreTrainedModel):
            return self.attn(layer_idx).v_proj(layer_normed)
        else:
            projected = self.attn(layer_idx).c_attn(layer_normed)
            q, k, v = torch.split(projected, self.hidden_size, dim=-1)
            return v

    def attn_q_slice(self, layer_idx: int):
        if isinstance(self.model, GPT2PreTrainedModel):
            return self.attn(layer_idx).c_attn, 0, self.hidden_size
        elif isinstance(self.model, (GPTNeoPreTrainedModel, LlamaLike)):
            return self.attn(layer_idx).q_proj, 0, self.hidden_size
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def attn_k_slice(self, layer_idx: int):
        if isinstance(self.model, GPT2PreTrainedModel):
            return self.attn(layer_idx).c_attn, self.hidden_size, self.hidden_size * 2
        elif isinstance(self.model, (GPTNeoPreTrainedModel, LlamaLike)):
            return self.attn(layer_idx).k_proj, 0, self.hidden_size
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def freeze_attention_pattern(self, hookpoint: str):
        index = self.name_to_index[hookpoint]
        outputs_saved = {}
        def freeze_slice(module, input, output, start, end):
            module_name = self.module_to_name[module]
            if module_name in outputs_saved:
                output = outputs_saved[module_name]
            if start == 0 and end == output.shape[-1]:
                result = output.detach()
            else:
                indices = torch.arange(output.shape[-1], device=output.device)
                mask = (indices >= start) & (indices < end)
                output = torch.where(mask, output.detach(), output)
                result = output.detach()
            outputs_saved[module_name] = result
            return result
        for module, start, end in (self.attn_q_slice(index), self.attn_k_slice(index)):
            module.register_forward_hook(partial(freeze_slice,
                                                 start=torch.tensor(start, device=self.device),
                                                 end=torch.tensor(end, device=self.device)))
