# autoresearch

This is an experiment to have the LLM do its own research, running on Hugging Face infrastructure.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read `train.py`**: This is the only file that matters. It contains the full GPT model, optimizer, training loop, dataloader, and evaluation — all in one self-contained file with inline UV dependencies. Read it for full context.
4. **Ask for a results bucket**: Ask the user for an HF Storage Bucket where best models and results will be saved (e.g. `hf://buckets/<username>/autoresearch-results`). If it doesn't exist, create it:
   ```bash
   hf buckets create <username>/autoresearch-results
   ```
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

7. **Ensure `hf` CLI is installed**: Run `hf --help` to check. If not installed:
   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   ```
7. **Ensure logged in**: Run `hf auth whoami` to verify. If not logged in, tell the human to run `hf auth login`.

Once you get confirmation, kick off the experimentation.

## Running on HF Jobs

Each experiment runs on a single GPU via HF Jobs. Launch training with:

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 10m \
    --namespace mishig \
    -v hf://datasets/karpathy/climbmix-400b-shuffle:/data \  # training data
    -v hf://buckets/mishig/autoresearch-cache:/cache \  # cached tokenizer
    train.py 2>&1 | tee run.log
```

- The dataset is mounted read-only at `/data` (parquet shards)
- The tokenizer is mounted from a bucket at `/cache/tokenizer`
- `train.py` auto-detects these mount paths

## Experimentation

The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify the evaluation harness. The `evaluate_bpb` function is the ground truth metric.
- Modify the dataloader, tokenizer, or constants (`MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS`, etc.).
- Add dependencies beyond what's in the inline script metadata (you can only use what's there: torch, kernels, pyarrow, tiktoken, numpy).

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Research with `hf papers`

Before each experiment, use `hf papers` to find ideas from recent research. This is your primary source of inspiration for what to try next.

**Search for relevant techniques:**

```bash
# Search for papers on a specific topic
hf papers search "efficient transformer training"
hf papers search "learning rate schedule"
hf papers search "attention mechanism optimization"
hf papers search "small language model pretraining"
```

**Read promising papers:**

```bash
# Read a paper's full content as markdown
hf papers read <paper_id>
```

Papers are just one source of inspiration — you are equally encouraged to come up with your own novel ideas, combine techniques in new ways, or try things that no paper has explored. Use papers to bounce ideas off of, not as a script to follow.

**How to use papers in the loop:**

1. Before choosing your next experiment, search for papers related to your current bottleneck or area of interest (e.g. if loss is plateauing, search "loss plateau" or "learning rate warmup").
2. Read 1-2 promising papers. Skim for the key idea — you don't need to read the whole thing.
3. Extract one concrete, implementable change from the paper and try it.
4. If you're stuck or out of ideas, browse trending papers or search for broader topics like "transformer architecture" or "training efficiency".
5. Log which paper inspired each experiment in the description column of results.tsv.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_bpb	memory_gb	status	paper	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. paper ID that inspired the change (e.g. `2501.12345`) — use `-` if not paper-inspired
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	paper	description
a1b2c3d	0.997900	44.0	keep	-	baseline
b2c3d4e	0.993200	44.2	keep	2503.08234	differential attention from paper
c3d4e5f	1.005000	44.0	discard	2502.11091	replaced RMSNorm with CrmsNorm
d4e5f6g	0.000000	0.0	crash	-	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar27`).

LOOP FOREVER:

1. **Research**: Use `hf papers search` or `hf papers list` to find a promising technique or idea. Read the paper with `hf papers read <id>` if it looks relevant.
2. **Implement**: Modify `train.py` with the experimental idea.
3. **Commit**: `git commit` the change.
4. **Run**: Submit the HF Job (see "Running on HF Jobs" above). Wait for it to complete.
5. **Evaluate**: Check the job logs for results. Extract `val_bpb` and `peak_vram_mb`.
6. If the job produced no results, it crashed. Use `hf jobs logs <namespace>/<job_id>` to read the error and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. **Log**: Record the results in results.tsv (NOTE: do not commit results.tsv, leave it untracked by git).
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit. Then:
   - Save the current best `train.py` and `results.tsv` to the results bucket:
     ```bash
     hf buckets cp train.py hf://buckets/<username>/autoresearch-results/best_train.py
     hf buckets cp results.tsv hf://buckets/<username>/autoresearch-results/results.tsv
     ```
   - Update `README.md` with a results table showing all experiments so far, and for the new best: the val_bpb value and a 1-2 sentence description of what changed. Commit the README update.
9. If val_bpb is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, search for more papers — browse trending papers, try new search queries, look at different subfields. The loop runs until the human interrupts you, period.
