import json
import os
import random
import numpy as np
import torch
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from functools import partial
from scipy.sparse.linalg import LinearOperator
from loguru import logger
from tqdm.auto import tqdm, trange
from jaxtyping import Float, Int, Bool, Array
from simple_parsing import Serializable

from .caching import TranscodedModel, TranscodedOutputs
from .nodes import (Contribution, Edge, ErrorNode, InputNode, IntermediateNode,
                    Node, OutputNode)
from .utils import cantor, measure_time
from .saving import cache_features, make_latent_dataset, cache_contexts


@dataclass
class AttributionConfig:
    # name of the run
    name: str
    # ID for the model the features are from
    scan: str
    # seed for random number generator
    seed: int = 42

    # how many target nodes to compute contributions for
    flow_steps: int = 15000
    # batch size for MLP attribution
    batch_size: int = 32
    # whether to use the softmax gradient for the output node
    # instead of the logit
    softmax_grad_type: Literal["softmax", "mean", "straight"] = "mean"

    # remove features that are this dense
    filter_high_freq_early: float = 0.01

    # remove MLP edges below this threshold
    pre_filter_threshold: float = 1e-6
    # keep edges that make up this fraction of the total influence
    edge_cum_threshold: float = 0.98
    # keep top k edges for each node
    top_k_edges: int = 32
    # method for computing influence
    influence_method: Literal["gpu", "cpu", "iterative"] = "gpu"
    # keep nodes that make up this fraction of the total influence
    node_cum_threshold: float = 0.8
    # target node count for pruning
    pruning_target_counts: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    disable_prune_sweep: bool = True
    pruning_target_meaning: Literal["l0", "nodes"] = "l0"
    # keep per_layer_position nodes above this threshold for each layer/position pair
    secondary_threshold = 1e-5
    per_layer_position = 0
    # preserve all error/input nodes
    keep_all_error_nodes: bool = False
    keep_all_input_nodes: bool = True
    # debugging: use all nodes as targets
    use_all_targets: bool = False

    # correct for bias when saving top output logits
    use_logit_bias: bool = False

    use_self_explanation: bool = False
    selfe_min_strength: float = 2.0
    selfe_max_strength: float = 8.0
    self_sim_layer: int = 15
    selfe_n: int = 64
    selfe_pick: int = 5
    selfe_n_tokens: int = 8


@dataclass
class SaveResults(Serializable):
    completeness_score: float
    replacement_score: float
    completeness_score_unpruned: float
    replacement_score_unpruned: float
    sweep_pruning_results: dict[int, dict[str, float]]
    num_nodes: int
    node_threshold: float
    edge_threshold: float
    node_cum_threshold: float
    config: AttributionConfig
    l0s: list[int]
    errors: list[float]


class AttributionGraph:
    def __init__(
        self,
        model: TranscodedModel,
        cache: TranscodedOutputs,
        config: AttributionConfig
    ):
        self.config = config
        self.model = model
        self.cache = cache
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.nodes_by_layer_and_token: dict[int, dict[int, list[Node]]] = {}
        self.initialize_graph()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def num_layers(self):
        return self.model.num_layers

    @property
    def seq_len(self):
        return self.cache.seq_len

    @property
    def input_ids(self):
        return self.cache.input_ids[0]

    @property
    def logits(self):
        return self.cache.logits[0]

    def adjacency_matrix(self):
        logger.info("Deduplicating nodes")
        dedup_node_names = set()
        for edge in self.edges.values():
            dedup_node_names.add(edge.source.id)
            dedup_node_names.add(edge.target.id)
        dedup_node_names = list(dedup_node_names)
        dedup_node_indices = {name: i for i, name in enumerate(dedup_node_names)}

        n_initial = len(dedup_node_names)
        activation_sources = np.ones((n_initial,))
        for index, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, IntermediateNode):
                activation_sources[index] = node.activation

        logger.info("Finding adjacency matrix")
        logger.info(f"Number of nodes: {len(dedup_node_names)}")

        adj_matrix = np.zeros((n_initial, n_initial))
        for edge in self.edges.values():
            target_index = dedup_node_indices[edge.target.id]
            source_index = dedup_node_indices[edge.source.id]
            weight = edge.weight
            adj_matrix[target_index, source_index] = weight * activation_sources[source_index]

        return adj_matrix, dedup_node_names

    @torch.inference_mode()
    def find_influence(self, adj_matrix: np.ndarray):
        n_initial = adj_matrix.shape[0]
        adj_matrix = np.abs(adj_matrix)
        adj_matrix /= np.maximum(1e-10, np.nan_to_num(adj_matrix.sum(axis=1, keepdims=True)))
        if self.config.influence_method == "gpu":
            adj_gpu = torch.from_numpy(adj_matrix).to(self.model.device)
            identity = torch.eye(n_initial, device=self.model.device)
            influence = torch.inverse(identity - adj_gpu) - identity
            influence = influence.cpu().numpy()
        elif self.config.influence_method == "iterative":
            influence = LinearOperator(
                shape=(n_initial, n_initial),
                matvec=partial(compute_influence, A=adj_matrix.T),
                rmatvec=partial(compute_influence, A=adj_matrix),
                dtype=adj_matrix.dtype,
            )
        else:
            influence = np.linalg.inv(np.eye(n_initial) - adj_matrix) - np.eye(n_initial)
        return influence, adj_matrix

    def save_graph(self, save_dir: os.PathLike | None) -> SaveResults | None:
        adj_matrix, dedup_node_names = self.adjacency_matrix()
        dedup_node_indices = {name: i for i, name in enumerate(dedup_node_names)}

        n_initial = len(dedup_node_names)
        emb_mask = np.zeros((n_initial,), dtype=np.bool_)
        error_mask = np.zeros((n_initial,), dtype=np.bool_)
        for i, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, ErrorNode):
                error_mask[i] = 1
            if isinstance(node, InputNode):
                emb_mask[i] = 1

        logger.info("Finding influence sources")
        influence_sources = np.zeros((n_initial,))
        for index, node in enumerate(dedup_node_names):
            node = self.nodes[node]
            if isinstance(node, OutputNode):
                influence_sources[index] = node.probability

        logger.info("Finding influences")
        original_adj_matrix = adj_matrix.copy()
        influence, adj_matrix = self.find_influence(adj_matrix)
        usage = influence_sources @ influence
        logger.info(f"Top influences: {usage[np.argsort(usage)[-10:][::-1]].tolist()}")

        usage_and_sources = usage
        completeness_score_unpruned = (usage_and_sources * (1 - adj_matrix[:, error_mask].sum(axis=1))).sum() / usage_and_sources.sum()
        logger.info(f"Completeness score (unpruned): {completeness_score_unpruned:.3f}")

        influence_emb = usage[emb_mask].sum()
        influence_err = usage[error_mask].sum()
        replacement_score_unpruned = influence_emb / (influence_emb + influence_err)
        logger.info(f"Replacement score (unpruned): {replacement_score_unpruned:.3f}")

        logger.info("Selecting nodes and edges")

        usage_sort = np.argsort(usage)[::-1]
        usage_rev_sort = np.argsort(usage_sort)
        sorted_usage = usage[usage_sort]
        cumsum_usage = np.cumsum(sorted_usage)
        cumsum_usage = cumsum_usage / cumsum_usage[-1]
        node_threshold = sorted_usage[np.searchsorted(cumsum_usage, self.config.node_cum_threshold)]
        logger.info(f"Node threshold: {node_threshold}")


        def filter_nodes(node_threshold):
            selected_nodes = [node for node in dedup_node_names
                    if self.nodes[node].node_type in ("OutputNode",)
                    + (("InputNode",) if self.config.keep_all_input_nodes else ())
                    + (("ErrorNode",) if self.config.keep_all_error_nodes else ())]

            for seq_idx in range(self.cache.input_ids.shape[-1]):
                for layer_idx in range(self.num_layers):
                    matching_nodes = [
                        node for node in dedup_node_names
                        if self.nodes[node].layer_index == layer_idx
                        and self.nodes[node].token_position == seq_idx
                        and node not in selected_nodes
                    ]
                    matching_nodes.sort(key=lambda x: usage[dedup_node_indices[x]], reverse=True)
                    matching_nodes = matching_nodes[:self.config.per_layer_position] + [
                        node
                        for node in matching_nodes[self.config.per_layer_position:]
                        if usage[dedup_node_indices[node]] > node_threshold
                    ]
                    matching_nodes = [
                        node for node in matching_nodes
                        if usage[dedup_node_indices[node]] > self.config.secondary_threshold
                    ]
                    selected_nodes.extend(matching_nodes)
            filtered_mask = np.zeros((len(dedup_node_names),), dtype=bool)
            for node in selected_nodes:
                filtered_mask[dedup_node_indices[node]] = 1

            # pruned nodes can receive influence but don't contribute to the completeness score

            filtered_adj_matrix = original_adj_matrix.copy()
            # filtered_adj_matrix = np.abs(filtered_adj_matrix)
            # filtered_adj_matrix /= np.maximum(1e-10, np.nan_to_num(filtered_adj_matrix.sum(axis=1, keepdims=True)))
            for node_name in dedup_node_names:
                node = self.nodes[node_name]
                node_index = dedup_node_indices[node_name]
                contribution = original_adj_matrix[:, node_index]
                kept = filtered_mask[node_index]
                if not kept:
                    if not isinstance(node, ErrorNode):
                        filtered_adj_matrix[:, node_index] = 0
                        filtered_adj_matrix[node_index, :] = 0
                    error_node = self.nodes[f"error_{node.token_position}_{node.layer_index}"]
                    err_index = dedup_node_indices[error_node.id]
                    if filtered_mask[err_index] and not isinstance(node, ErrorNode):
                        filtered_adj_matrix[:, err_index] += contribution
            filtered_adj_matrix = filtered_adj_matrix * filtered_mask[:, None]

            filtered_influence, filtered_adj_matrix = self.find_influence(filtered_adj_matrix)
            filtered_influence = influence_sources + influence_sources @ filtered_influence

            err_mask = error_mask
            completeness_score = ((filtered_influence * filtered_mask) * (1 - (filtered_adj_matrix * err_mask[None, :]).sum(axis=1))).sum() / (filtered_influence * filtered_mask).sum()
            influence_emb = filtered_influence[emb_mask].sum()
            influence_err = filtered_influence[err_mask].sum()
            replacement_score = float(influence_emb / (influence_emb + influence_err))

            # prune back for later metrics with filtered indices
            filtered_adj_matrix = filtered_adj_matrix[filtered_mask][:, filtered_mask].copy()
            filtered_influence = filtered_influence[filtered_mask]

            return selected_nodes, filtered_mask, filtered_influence, filtered_adj_matrix, completeness_score, replacement_score

        selected_nodes, filtered_mask, filtered_influence, filtered_adj_matrix, completeness_score, replacement_score = filter_nodes(node_threshold)
        filtered_index = np.cumsum(filtered_mask) - 1
        filtered_influence_matrix = filtered_influence[:, None] * filtered_adj_matrix

        logger.info(f"Selected {len(selected_nodes)} nodes")
        logger.info(f"Completeness score: {completeness_score:.3f}")
        logger.info(f"Replacement score: {replacement_score:.3f}")

        sweep_pruning_results = {}
        if save_dir is None and self.config.disable_prune_sweep:
            logger.info("Not saving circuit, enabling prune sweep")
            self.config.disable_prune_sweep = True
        for n_nodes in (self.config.pruning_target_counts if not self.config.disable_prune_sweep else []):
            n_nodes_select = n_nodes
            if self.config.pruning_target_meaning == "l0":
                n_nodes_select = n_nodes_select * self.seq_len
            n_nodes_select = n_nodes_select + self.seq_len + len(self.output_nodes)
            threshold = sorted_usage[min(n_nodes_select, len(sorted_usage) - 1)]
            selected_nodes_, _filtered_mask, _filtered_influence, _filtered_adj_matrix, completeness_score, replacement_score = filter_nodes(threshold)
            logger.info(f"Pruned to {n_nodes_select} nodes, completeness={completeness_score:.3f}, replacement={replacement_score:.3f}")
            # logger.info(f"Fraction of error nodes: {np.sum(error_mask[filtered_mask]) / len(selected_nodes):.3f}")
            # logger.info(f"Fraction of input nodes: {np.sum(emb_mask[filtered_mask]) / len(selected_nodes):.3f}")
            sweep_pruning_results[n_nodes] = dict(
                n_selected=len(selected_nodes_),
                completeness_score=completeness_score,
                replacement_score=replacement_score,
            )

        selected_edge_matrix = np.sort(filtered_influence_matrix.flatten())[::-1]
        edge_cumsum = np.cumsum(selected_edge_matrix)
        edge_cumsum = edge_cumsum / edge_cumsum[-1]

        edge_threshold = selected_edge_matrix[np.searchsorted(edge_cumsum, self.config.edge_cum_threshold)]
        logger.info(f"Edge threshold: {edge_threshold}")

        if save_dir is None:
            return SaveResults(
                completeness_score=completeness_score,
                replacement_score=replacement_score,
                completeness_score_unpruned=completeness_score_unpruned,
                replacement_score_unpruned=replacement_score_unpruned,
                num_nodes=len(selected_nodes),
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                node_cum_threshold=self.config.node_cum_threshold,
                sweep_pruning_results=sweep_pruning_results,
                config=self.config,
                l0s=self.cache.l0_per_layer,
                errors=self.cache.error_magnitudes,
            )

        export_edges = []
        has_incoming = set()
        has_outgoing = set()
        for edge in self.edges.values():
            if not (edge.source.id in selected_nodes and edge.target.id in selected_nodes):
                continue
            if abs(
                filtered_adj_matrix[filtered_index[dedup_node_indices[edge.target.id]],
                                   filtered_index[dedup_node_indices[edge.source.id]]]
                ) < edge_threshold:
                continue
            has_outgoing.add(edge.source.id)
            has_incoming.add(edge.target.id)
            export_edges.append(edge)
        self.exported_edges = export_edges
        logger.info(f"Selected {len(export_edges)} edges")

        export_nodes = []
        for n in selected_nodes:
            if isinstance(self.nodes[n], IntermediateNode):
                if n not in has_incoming or n not in has_outgoing:
                    continue
            if isinstance(self.nodes[n], ErrorNode):
                if n not in has_outgoing:
                    continue
            export_nodes.append(self.nodes[n])
        self.exported_nodes = export_nodes
        logger.info(f"Final number of nodes: {len(export_nodes)}")

        logger.info("Exporting graph")
        export_offset = self.cache.original_input_ids.shape[-1] - self.cache.input_ids.shape[-1]
        input_ids = self.cache.original_input_ids[0].tolist()
        tokens = [self.model.decode_token(i) for i in input_ids]
        # prefix = ["<EOT>"]
        prefix = []
        metadata = dict(
            slug=self.config.name,
            # scan="jackl-circuits-runs-1-4-sofa-v3_0",
            scan=self.config.scan,
            prompt_tokens=prefix + tokens,
            prompt="".join(tokens),
            title_prefix="",
            n_layers=self.model.num_layers,
            node_sep_threshold=node_threshold,
            node_threshold=self.config.node_cum_threshold,
        )

        nodes_json = [
            dict(
                feature_type=dict(
                    IntermediateNode="cross layer transcoder",
                    OutputNode="logit",
                    InputNode="embedding",
                    ErrorNode="mlp reconstruction error",
                )[node.node_type],
                layer=(str(node.layer_index)
                if node.layer_index != self.model.num_layers else "-1")
                if node.layer_index != -1 else "E",
                node_id=node.id_js,
                jsNodeId=node.id_js + "-0",
                feature=int(node.feature_index) if hasattr(node, "feature_index") else None,
                ctx_idx=node.token_position + len(prefix) + export_offset,
                run_idx=0, reverse_ctx_idx=0,
                clerp="" if not isinstance(node, OutputNode) else f"output: \"{node.token_str}\" (p={node.probability:.3f})",
                token_prob=0.0 if not isinstance(node, OutputNode) else node.probability,
                is_target_logit=False,
                influence=float(cumsum_usage[usage_rev_sort[dedup_node_indices[node.id]]]),
            ) for node in export_nodes
        ]

        for node in nodes_json:
            if node["feature_type"] != "cross layer transcoder":
                continue
            layer, feature = int(node["layer"]), int(node["feature"])
            idx_cantor = cantor(layer, feature)
            node["feature"] = idx_cantor

        links_json = [
            e.to_dict() for e in export_edges
        ]

        result = dict(
            metadata=metadata,
            nodes=nodes_json,
            links=links_json,
            qParams=dict(
                linkType="both",
                # # node IDs
                # pinnedIds=[],
                # # clickedId=export_nodes[0].id,
                # clickedId=None,
                # supernodes=[],
                # # x/y position pairs for supernodes, separated by spaces
                # sg_pos="",
            )
        )

        save_dir = Path(save_dir)
        logger.info("Saving graph to", save_dir)

        graph_data_dir = save_dir / "graph_data"
        graph_data_dir.mkdir(parents=True, exist_ok=True)
        circuit_path = graph_data_dir / f"{self.config.name}.json"
        open(circuit_path, "w").write(json.dumps(result))

        logger.info("Collecting metadata")
        metadatas = []
        own_file_index = 0
        for i, save_file in enumerate(graph_data_dir.glob("*.json")):
            metadatas.append(json.loads(save_file.read_text())["metadata"])
            if save_file.stem == self.config.name:
                own_file_index = i
        metadatas[0], metadatas[own_file_index] = metadatas[own_file_index], metadatas[0]
        (save_dir / "data").mkdir(parents=True, exist_ok=True)
        open(save_dir / "data/graph-metadata.json", "w").write(json.dumps(dict(graphs=metadatas)))

        return circuit_path

    def get_dense_features(self, cache_path: os.PathLike):
        cache_path = Path(cache_path)
        dense_features = set()
        if not cache_path.exists() or not len(list(cache_path.glob("*"))) or not self.config.filter_high_freq_early:
            logger.warning("Skipping dense feature detection because cache does not exist or filter_high_freq_early is 0")
            self.dense_features = dense_features
            return
        dense_cache_path = cache_path.parent / "dense_features.json"
        if dense_cache_path.exists():
            dense = json.loads(dense_cache_path.read_text())
        else:
            dense = []
            ds = make_latent_dataset(self.model, cache_path, None)
            for buf in tqdm(ds.buffers, desc="Finding dense features"):
                module = buf.module_path
                all_features, _, all_tokens = buf.load()
                all_features = all_features[..., 2]

                # unique, counts = torch.unique(all_features, return_counts=True)
                # freqs = counts / all_tokens.numel()
                # too_high = freqs > self.config.filter_high_freq_early

                all_tokens, all_features = all_tokens.numpy(), all_features.numpy()
                unique, counts = np.unique(all_features, return_counts=True)
                freqs = counts / all_tokens.size
                too_high = freqs > self.config.filter_high_freq_early

                # dead_features[module].extend(unique[too_high].tolist())
                if self.model.pre_ln_hook:
                    module = module.replace(f".{self.model.mlp_layernorm_name}", ".mlp")
                layer_idx = self.model.temp_hookpoints_mlp.index(module)
                feature_names = [[layer_idx, feature] for feature in unique[too_high].tolist()]
                dense.extend(feature_names)
            open(dense_cache_path, "w").write(json.dumps(dense))

        self.dense_features = set((pos, *feature) for pos in range(self.seq_len) for feature in dense)

    def cache_features(self, save_dir: os.PathLike):
        nodes = []
        for node in self.exported_nodes:
            if node.node_type == "IntermediateNode":
                nodes.append((int(node.layer_index), int(node.feature_index)))
        cache_features(self.model, save_dir, nodes, scan=self.config.scan, use_logit_bias=self.config.use_logit_bias)

    def cache_self_explanations(self, save_dir: os.PathLike):
        if not self.config.use_self_explanation:
            return
        save_dir = Path(save_dir)
        for node in tqdm(self.exported_nodes, desc="Caching self-explanations"):
            if node.node_type == "IntermediateNode":
                explanation = self.self_explain_feature(node)
                feature_dir = save_dir / "features" / self.config.scan
                feature_dir.mkdir(parents=True, exist_ok=True)
                feature_path = feature_dir / f"{cantor(node.layer_index, node.feature_index)}.json"
                if feature_path.exists():
                    explanation = json.loads(feature_path.read_text()) | explanation
                feature_path.write_text(json.dumps(explanation))

    @torch.inference_mode()
    def self_explain_feature(self, node: IntermediateNode):
        layer, feature = node.layer_index, node.feature_index
        w_dec = self.model.w_dec_i(layer, feature, use_skip=True)
        w_enc = self.model.w_enc_i(layer, feature)
        strengths = torch.linspace(self.config.selfe_min_strength, self.config.selfe_max_strength, self.config.selfe_n, device=self.model.device)
        w_dec = w_dec[None, :] * strengths[:, None]
        w_enc = w_enc[None, :] * strengths[:, None]
        dec_explanations = self.self_explain_generate(w_dec, min_strength=self.config.selfe_min_strength, max_strength=self.config.selfe_max_strength, seq_len=self.config.selfe_n_tokens)
        enc_explanations = self.self_explain_generate(w_enc, min_strength=self.config.selfe_min_strength, max_strength=self.config.selfe_max_strength, seq_len=self.config.selfe_n_tokens)
        return dict(
            self_explanation_enc=enc_explanations,
            self_explanation_dec=dec_explanations,
        )

    def self_explain_generate(self, vectors: torch.Tensor,
                              *, min_strength: float,
                              max_strength: float,
                              seq_len: int):
        prompt = "ONLINE DICTIONARY\nThe meaning of the word ? is \""
        tokenized_prompt = self.model.tokenizer.encode(prompt)
        question_mark_token = self.model.tokenizer.encode("a ?")[-1]
        token_index = tokenized_prompt.index(question_mark_token)
        emb_vec = vectors / vectors.norm(dim=-1, keepdim=True)
        batch_size = vectors.shape[0]
        strengths = torch.linspace(min_strength, max_strength, batch_size, device=self.model.device)
        emb_vec = emb_vec[None, :] * strengths[:, None]
        tokens = self.model.tokenizer([prompt] * batch_size, return_tensors="pt").to(self.model.device).input_ids
        state = (tokens, tokens, None, None, None)
        def step(state):
            all_tokens, tokens, cache, entropy, self_similarity = state
            for m in self.model.model.modules():
                m._forward_hooks = OrderedDict()

            if tokens.shape[1] != 1:
                def patch(module, input, output):
                    output[:, token_index] = emb_vec
                    return output

                self.model.embedding_module.register_forward_hook(patch)

                def collect_self_similarity(module, input, output):
                    nonlocal self_similarity
                    if isinstance(output, tuple):
                        output = output[0]
                    self_similarity = torch.nn.functional.cosine_similarity(
                        output[:, -1],
                        emb_vec,
                        dim=-1
                    )[0]

                self.model.get_layer(self.config.self_sim_layer).register_forward_hook(collect_self_similarity)

            output = self.model.model(tokens, past_key_values=cache)
            logits = output.logits

            if entropy is None:
                probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)
                logprobs = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
                entropy = -torch.sum(probs * logprobs, dim=-1)

            probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            all_tokens = torch.cat([all_tokens, next_tokens], dim=1)
            return (all_tokens, next_tokens, output.past_key_values, entropy, self_similarity)

        try:
            for _ in range(seq_len):
                logger.disable("attribute")
                state = step(state)
                tokens = state[0]
        finally:
            logger.enable("attribute")
        decoded = [self.model.tokenizer.decode(seq[len(tokenized_prompt):]) for seq in tokens.tolist()]
        entropy, self_similarity = state[3:5]
        entropy = entropy - entropy.min()
        entropy = entropy / entropy.max()
        self_similarity = self_similarity - self_similarity.min()
        self_similarity = self_similarity / self_similarity.max()
        top_self_sims = torch.topk(self_similarity, k=self.config.selfe_pick).indices.tolist()
        decoded = [decoded[i] for i in top_self_sims]
        # decoded = [str((float(entropy), float(self_sim))) + decoded for entropy, self_sim, decoded in zip(entropy, self_similarity, decoded)]
        decoded = [decoded.partition('"')[0].partition("\n")[0] for decoded in decoded]
        return decoded

    def cache_contexts(self, cache_path: os.PathLike, save_dir: os.PathLike):
        nodes = []
        for node in self.exported_nodes:
            if node.node_type == "IntermediateNode":
                nodes.append((int(node.layer_index), int(node.feature_index)))
        cache_contexts(self.model, cache_path, save_dir, nodes, scan=self.config.scan)

    def initialize_graph(self):
        num_layers = self.num_layers
        logger.info(f"Initializing graph with {num_layers} layers")
        seq_len = self.seq_len
        self.nodes_by_layer_and_token = {
            layer: {token: [] for token in range(seq_len)}
            for layer in range(num_layers)
        }

        offset = self.cache.mlp_outputs[0].source_activation.shape[1] - seq_len

        # Start by creating the input nodes
        og_input_ids = self.cache.original_input_ids[0]
        for i in range(-offset, seq_len):
            embedding = self.model.embedding_weight[og_input_ids[i]]
            input_node = InputNode(
                id=f"input_{i}",
                token_position=i,
                token_idx=og_input_ids[i],
                token_str=self.tokenizer.decode([og_input_ids[i]]),
                output_vector=embedding.to(dtype=torch.bfloat16),
                layer_index=-1,
            )
            self.nodes[input_node.id] = input_node
            if i >= 0:
                self.nodes_by_layer_and_token[0][i].append(input_node)

        self.activation_indices_tensors = {}
        self.intermediate_nodes = {}
        self.intermediate_nodes_k = {}
        # Create the intermediate nodes
        for layer, activations in self.cache.mlp_outputs.items():
            activations_tensor, indices_tensor = (
                activations.activation[0],
                activations.location[0],
            )

            self.activation_indices_tensors[layer] = dict(
                activations=activations_tensor,
                indices=indices_tensor,
            )

            for token_position, (top_acts, top_indices) in enumerate(
                zip(activations_tensor.tolist(), indices_tensor.tolist())
            ):
                for k_index, (act, index) in enumerate(zip(top_acts, top_indices)):
                    intermediate_node = IntermediateNode(
                        id=f"intermediate_{token_position}_{layer}_{index}",
                        layer_index=layer,
                        feature_index=index,
                        token_position=token_position,
                        activation=float(act),
                        input_vector=None,
                    )
                    self.nodes[intermediate_node.id] = intermediate_node
                    self.nodes_by_layer_and_token[layer][token_position].append(
                        intermediate_node
                    )
                    self.intermediate_nodes_k[(token_position, layer, k_index)] = intermediate_node
                    self.intermediate_nodes[(token_position, layer, index)] = intermediate_node
                # Create the error and skip nodes
                error = activations.error[0, token_position]
                error_node = ErrorNode(
                    id=f"error_{token_position}_{layer}",
                    layer_index=layer,
                    token_position=token_position,
                    output_vector=error.to(dtype=torch.bfloat16),
                )
                self.nodes_by_layer_and_token[layer][token_position].append(
                    error_node
                )
                self.nodes[error_node.id] = error_node
        # Create the output node
        # Top 10 logits
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.logits[-1, :], dim=0)
        top_10_indices = torch.argsort(probabilities, descending=True)[:10]
        top_10_probabilities = probabilities[top_10_indices]
        total_probability = 0
        output_nodes = []
        for i in range(10):
            match self.config.softmax_grad_type:
                case "softmax":
                    # "logit_i - mean logit"; what does "mean logit" mean?
                    # i assume it is the mean according to the softmax distribution
                    # because that is the gradient of softmax ignoring the scaling factor
                    before_gradient = self.logits[-1, top_10_indices[i]] - \
                        torch.dot(self.logits[-1, :], self.logits[-1, :].softmax(dim=-1).detach())
                case "mean":
                    # https://transformer-circuits.pub/2025/attribution-graphs/methods.html
                    # #appendix-attribution-graph-computation
                    before_gradient = self.logits[-1, top_10_indices[i]] - torch.mean(self.logits[-1, :])
                case "straight":
                    before_gradient = self.logits[-1, top_10_indices[i]]
            before_gradient.backward(retain_graph=True)
            gradient = self.cache.last_layer_activations.grad
            assert gradient is not None
            self.cache.last_layer_activations.grad = None
            # gradient = self.logits[0,-1,top_10_indices[i]].expand(gradient.shape)
            # vector = self.model.lm_head.weight[top_10_indices[i]]#-torch.mean(self.model.lm_head.weight,dim=0)
            output_node = OutputNode(
                id=f"output_{seq_len-1}_{i}",
                token_position=seq_len - 1,
                token_idx=top_10_indices[i],
                token_str=self.tokenizer.decode([top_10_indices[i]]),
                probability=top_10_probabilities[i].item(),
                logit=self.logits[-1, top_10_indices[i]].item(),
                input_vector=gradient[0, -1].to(dtype=torch.bfloat16),
                layer_index=self.num_layers,
                logit_idx=i,
            )
            self.nodes[output_node.id] = output_node
            output_nodes.append(output_node)
            total_probability += top_10_probabilities[i]
            if total_probability > 0.95:
                break
        self.output_nodes = output_nodes

        # cleared each time we re-initialize the graph
        self.queue = AttributionQueue(self.cache, self)
        self.remaining_output_nodes = output_nodes.copy()

        self.dead_features = set()

    @torch.autocast("cuda")
    def flow_once(self):
        true_seq_len = self.cache.input_ids.shape[1]
        fake_seq_len = self.cache.mlp_outputs[0].source_activation.shape[1]
        offset = fake_seq_len - true_seq_len

        gradient = 0
        mlp_source = False
        # if the queue is empty, get the output node with the highest probability
        # TODO: handle the other output nodes
        if len(self.remaining_output_nodes) > 0:
            target_node = self.remaining_output_nodes.pop()
            influences, target_nodes = [target_node.probability], [target_node]
            logger.debug("Starting from output node")
        else:
            if len(self.queue) == 0:
                return False
            influences, max_layer, target_indices = self.queue.pop_n(self.config.batch_size)
            mlp_source = True
            list_target_indices = target_indices.tolist()
            target_nodes = [self.intermediate_nodes_k[i, max_layer, j] for i, j in list_target_indices]

        # compute all the contributions
        if isinstance(target_nodes[0], OutputNode):
            max_layer = target_nodes[0].layer_index
            gradient = target_nodes[0].input_vector
            target_graph_node = self.cache.last_layer_activations[0, target_nodes[0].token_position + offset]
            max_mlp_layer = self.model.num_layers
        elif mlp_source:
            target_graph_node = self.cache.mlp_outputs[max_layer].activation
            gradient = None
            batch_idx = torch.arange(len(target_indices))
            gradient = torch.zeros_like(target_graph_node)
            gradient[batch_idx, target_indices[:, 0], target_indices[:, 1]] += 1
            max_mlp_layer = max_layer
        else:
            raise ValueError

        backward_to = [self.cache.first_layer_activations] + [
            node
            for i in range(max_mlp_layer)
            for node in (
                self.cache.mlp_outputs[i].source_activation,
                self.cache.mlp_outputs[i].source_error,
            )
        ]
        gradients = torch.autograd.grad(
            [target_graph_node],
            backward_to,
            [gradient],
            retain_graph=True,
            allow_unused=False,
        )

        with measure_time("Summarizing contributions of node", disabled=True), torch.no_grad():
            for batch_idx, (target_node, influence) in enumerate(zip(target_nodes, influences)):
                all_contributions = []
                for seq_idx in range(-offset, target_node.token_position + 1):
                    contribution = gradients[0][batch_idx][seq_idx + offset] @ backward_to[0][batch_idx][seq_idx + offset]
                    input_node_name = f"input_{seq_idx}"
                    source = self.nodes[input_node_name]
                    all_contributions.append(Contribution(
                        source=source,
                        target=target_node,
                        contribution=contribution.to(device="cpu", non_blocking=True),
                    ))

                for layer_idx in range(max_mlp_layer):
                    error_index = 2 + layer_idx * 2
                    error_grad, error_val = gradients[error_index], backward_to[error_index]

                    for seq_idx in range(target_node.token_position + 1):
                        error_contribution = error_grad[batch_idx, -true_seq_len:][seq_idx] @ error_val[batch_idx, -true_seq_len:][seq_idx]
                        source = self.nodes[f"error_{seq_idx}_{layer_idx}"]
                        assert source.token_position <= target_node.token_position, f"{source.token_position} <= {target_node.token_position}"
                        all_contributions.append(Contribution(
                            source=source,
                            target=target_node,
                            contribution=error_contribution.to(device="cpu", non_blocking=True),
                        ))

                for layer_idx in range(max_mlp_layer):
                    mlp_index = 1 + layer_idx * 2
                    mlp_grad = gradients[mlp_index][batch_idx, -true_seq_len:]
                    edge = ((mlp_grad * (mlp_grad.abs() >= self.config.pre_filter_threshold)).abs() + (1e-8 if self.config.use_all_targets else 0)) * (influence if not self.config.use_all_targets else 1) * self.cache.mlp_outputs[layer_idx].activation[0]
                    self.queue.layers[layer_idx].contributions += edge
                    n_elem = min(self.config.top_k_edges, (edge > 0).sum().item())
                    if n_elem == 0:
                        continue
                    edge_indices = torch.topk(edge.flatten(), n_elem).indices
                    edge_seq, edge_k = edge_indices // edge.shape[-1], edge_indices % edge.shape[-1]
                    edge_feat = self.cache.mlp_outputs[layer_idx].location[batch_idx, -true_seq_len:].flatten()[edge_indices]
                    edge_val = mlp_grad[edge_seq, edge_k]
                    self.queue.layers[layer_idx].edge_source_seq.append(edge_seq.to("cpu", non_blocking=True))
                    self.queue.layers[layer_idx].edge_source_idx.append(edge_feat.to("cpu", non_blocking=True))
                    self.queue.layers[layer_idx].edge_target_layer.append(target_node.layer_index)
                    self.queue.layers[layer_idx].edge_target_seq.append(target_node.token_position)
                    self.queue.layers[layer_idx].edge_target_idx.append(target_node.feature_index if isinstance(target_node, IntermediateNode) else (-1 - target_node.logit_idx))
                    self.queue.layers[layer_idx].edge_weight.append(edge_val.to("cpu", non_blocking=True))

                # Make new paths using the last path
                for n_path in range(0, len(all_contributions)):
                    new_contribution = all_contributions[n_path]
                    new_source = new_contribution.source
                    assert new_source.token_position <= target_node.token_position, f"{new_source.token_position} <= {target_node.token_position}"
                    weight = float(new_contribution.contribution)
                    edge = Edge(
                        source=new_source,
                        target=target_node,
                        weight=float(weight),
                    )
                    self.edges[edge.id] = edge

        if mlp_source:
            for source_layer, source_seq, source_idx, target_layer, target_seq, target_idx, weight in self.queue.purge():
                try:
                    source = self.intermediate_nodes[(source_seq, source_layer, source_idx)]
                except KeyError:
                    continue
                if target_idx < 0:
                    target = self.output_nodes[-1 - target_idx]
                else:
                    target = self.intermediate_nodes[(target_seq, target_layer, target_idx)]
                edge = Edge(
                    source=source,
                    target=target,
                    weight=weight
                )
                self.edges[edge.id] = edge

        return len(self.queue) > 0 or not mlp_source

    def flow(self, num_iterations: Optional[int] = None):
        if num_iterations is None:
            num_iterations = self.config.flow_steps // self.config.batch_size
        for i in (bar := trange(num_iterations, desc="Flowing contributions")):
            with measure_time(
                f"Iteration {i}",
                disabled=True,
            ):
                if not self.flow_once():
                    logger.debug("Queue is empty")
                    break
                logger.debug(f"Queue has {len(self.queue)} paths")
                bar.set_postfix(queue_elements=len(self.queue))


@dataclass
class AttributionLayer:
    visited: Bool[Array, "seq_len k"]
    contributions: Float[Array, "seq_len k"]
    edge_source_seq: list[Int[Array, "..."]]
    edge_source_idx: list[Int[Array, "..."]]
    edge_target_layer: list[Int[Array, "..."]]
    edge_target_seq: list[Int[Array, "..."]]
    edge_target_idx: list[Int[Array, "..."]]
    edge_weight: list[Float[Array, "..."]]


class AttributionQueue:
    def __init__(self, cache: TranscodedOutputs, graph: AttributionGraph):
        self.graph = graph
        self.cache = cache
        self.layers = {
            layer: AttributionLayer(
                visited=torch.zeros(output.activation.shape[1:], device=cache.input_ids.device, dtype=torch.bool),
                contributions=torch.zeros(output.activation.shape[1:], device=cache.input_ids.device, dtype=torch.float32),
                edge_source_seq=[],
                edge_source_idx=[],
                edge_target_layer=[],
                edge_target_seq=[],
                edge_target_idx=[],
                edge_weight=[],
            )
            for layer, output in cache.mlp_outputs.items()
        }

    @torch.no_grad()
    def contribution_not_visited(self, layer: int, multiply_by_activation: bool = False):
        return (self.layers[layer].contributions * ~self.layers[layer].visited * (self.cache.mlp_outputs[layer].activation[0] if multiply_by_activation else 1)).flatten()

    @torch.no_grad()
    def pop_n(self, n: int):
        layer = random.choice(list(layer for layer in self.layers.keys() if self.contribution_not_visited(layer).sum().item() > 0))
        contribution_not_visited = self.contribution_not_visited(layer)
        n_to_visit = min(n, (contribution_not_visited > 0).sum().item())
        contribution, visited = torch.topk(contribution_not_visited, n_to_visit)
        k = self.layers[layer].contributions.shape[-1]
        visited_seq, visited_k = visited // k, visited % k
        self.layers[layer].visited[visited_seq, visited_k] = True
        return contribution.tolist(), layer, torch.stack([visited_seq, visited_k], dim=-1)

    def __len__(self):
        return sum((self.contribution_not_visited(layer) > 0).sum().item() for layer in self.layers.keys())

    def purge(self):
        for layer_idx, layer in self.layers.items():
            while len(layer.edge_source_seq) > 0:
                source_seqs, source_idces, weights = (
                    layer.edge_source_seq.pop().tolist(),
                    layer.edge_source_idx.pop().tolist(),
                    layer.edge_weight.pop().tolist(),
                )
                target_layer, target_seq, target_idx = (
                    int(layer.edge_target_layer.pop()),
                    int(layer.edge_target_seq.pop()),
                    int(layer.edge_target_idx.pop()),
                )
                for source_seq, source_idx, weight in zip(source_seqs, source_idces, weights):
                    if weight == 0:
                        continue
                    yield layer_idx, source_seq, source_idx, target_layer, target_seq, target_idx, weight


# taken from https://github.com/safety-research/open-source-circuit-finding
def compute_influence(vec: np.ndarray, A: np.ndarray, max_iter: int = 1000):
    # computes vec @ (A + A^2 + ...)
    vec_prod = vec @ A
    total = vec_prod
    iterations = 0
    while np.any(vec_prod):
        if iterations >= max_iter:
            raise RuntimeError(f"Influence computation failed to converge after {iterations} iterations")
        vec_prod = vec_prod @ A
        total += vec_prod
        iterations += 1
    return total
