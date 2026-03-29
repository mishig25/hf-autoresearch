# autoresearch on Hugging Face

Autonomous LLM pretraining research running entirely on [Hugging Face infrastructure](https://huggingface.co/docs/hub/jobs). An AI agent iterates on a training script — modifying architecture, optimizer, hyperparameters — while reading recent papers via [`hf papers`](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-papers) for ideas. No local GPU needed. You only need a Hugging Face account.

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted to run on HF Jobs with mounted datasets and storage buckets.

*See an [example 24-hour run on A100](https://huggingface.co/buckets/mishig/autoresearch-results) — experiment artifacts including results, best models, and the full agent chat transcript are all saved to the bucket.*

![progress](https://huggingface.co/buckets/mishig/autoresearch-results/resolve/progress.png)

## Two files. That's it.

The entire repo is just two files:

- **`train.py`** — self-contained training script. Model, optimizer, dataloader, evaluation, and all dependencies declared inline. The agent modifies this file.
- **`program.md`** — instructions for the AI agent. Point your agent here and let it go.

No configs, no setup scripts, no dependency files. Training runs for a fixed 5-minute budget. The metric is **val_bpb** (validation bits per byte) — lower is better.

## Quick start

```bash
# 1. Install the HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# 2. Login
hf auth login

# 3. Run a training experiment
hf jobs uv run \
    --flavor a100-large \
    --timeout 10m \
    --namespace <your-username> \
    -v hf://datasets/karpathy/climbmix-400b-shuffle:/data \  # training data
    -v hf://buckets/mishig/autoresearch-cache:/cache \  # cached tokenizer
    train.py
```

That's it. No downloads, no data prep, no wasted compute.

## Zero overhead with mounted volumes

The `-v` flags above use [HF volume mounts](https://github.com/huggingface/hf-mount) to make remote data appear as local files inside the job. The training script reads parquet shards from `/data` as if the entire [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) dataset (6,543 shards) was already on disk — no bulk download, no waiting. Files are fetched lazily on access.

The tokenizer and other reusable artifacts live in an [HF Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) ([`mishig/autoresearch-cache`](https://huggingface.co/buckets/mishig/autoresearch-cache)), mounted read-write at `/cache`. The original repo retrains a BPE tokenizer from scratch every run (~60s on CPU). By storing it in a bucket, we skip that entirely — zero compute wasted on repetitive setup work. Buckets are mutable, non-versioned storage ideal for intermediate artifacts like tokenizers, checkpoints, and logs.

## Running the agent

Point Claude Code (or any AI agent) at the repo and say:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The agent will read `train.py`, establish a baseline, then loop autonomously: search papers, implement ideas, submit jobs, evaluate results, keep or discard.

## What's on HF

| Resource | Purpose |
|---|---|
| [`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) | Training dataset (mounted read-only at `/data`) |
| [`mishig/autoresearch-cache`](https://huggingface.co/buckets/mishig/autoresearch-cache) | Tokenizer bucket (mounted at `/cache`) |
| [HF Jobs](https://huggingface.co/docs/hub/jobs) | Compute (A100, H200, etc.) |
| [`hf papers`](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-papers) | Research paper search and reading |
