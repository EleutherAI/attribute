# attribute

Reproduction of Anthropic's [Attribution Graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

## Setup

### Install UV

**Option 1: Use `mise`** 
Install mise: https://mise.jdx.dev/getting-started.html

Environment setup can be done with:
```sh
mise install
```
This will install `uv`.

**Option 2: Direct install**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Pull dependencies
Once you have `uv` installed:

```sh
uv venv --seed
uv sync
```

### Huggingface Token
To use some of the read restricted models, you'll need a registerd Huggingface Account and corresponding Read token. Use the following to pick it up:

```sh
export HF_TOKEN=<YOUR READ TOKEN>
```


## Cache activations
This step is not strictly necessary. Given the weights of a transcoder, we need to compute activations on a large dataset to be able to select maximum-activating examples.

```
# scripts/cache_activations
MODEL=HuggingFaceTB/SmolLM2-135M
TRANSCODER="nev/SmolLM2-CLT-135M-73k-k32"
DATASET="--dataset_repo EleutherAI/fineweb-edu-dedup-10b --dataset_split train --n_tokens 10_000_000"
NAME="smollm-v1"

uv run python cache.py $MODEL $TRANSCODER --num_gpus 1 $DATASET --hookpoints layers.0.mlp layers.1.mlp layers.2.mlp layers.3.mlp layers.4.mlp layers.5.mlp layers.6.mlp layers.7.mlp layers.8.mlp layers.9.mlp layers.10.mlp layers.11.mlp layers.12.mlp layers.13.mlp layers.14.mlp layers.15.mlp layers.16.mlp layers.17.mlp layers.18.mlp layers.19.mlp layers.20.mlp layers.21.mlp layers.22.mlp layers.23.mlp layers.24.mlp layers.25.mlp layers.26.mlp layers.27.mlp layers.28.mlp layers.29.mlp --name $NAME --batch_size 16
```

Alternatively, you can download the cache from Huggingface:

```sh
cd results && git clone https://huggingface.co/nev/SmolLM2-CLT-135M-73k-k32-cache --depth=1
```

## Web UI

We developed a gradio visualization for running attribution:

```sh
uv run gradio serve.py
```

## Colab (WIP)

The web UI can be run from [a Colab notebook](https://colab.research.google.com/github/EleutherAI/attribute/blob/main/serve.ipynb).

## Run in CLI

```sh
SCAN="smollm-v1"
PROMPT_NAME="smollm-v1"
PROMPT_TEXT="Michael Jordan plays the sport of"
uv run python -m attribute \
--cache_path=results/$SCAN/latents \
--name $PROMPT_NAME \
--scan "$SCAN" \
"$PROMPT_TEXT" \
--transcoder_path="nev/SmolLM2-CLT-135M-73k-k32" --model_name="HuggingFaceTB/SmolLM2-135M" \
```

```sh
uv run python -m attribute
cd attribution-graphs-frontend
uv run python serve.py 9999
ngrok http 9999
```

Note: Ngrok now requires authentication

## TODOs
- [ ] Cross-verify with Anthropic implementation
