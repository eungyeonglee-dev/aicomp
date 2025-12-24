# DP dataset sharding & batch-size semantics (important)

### Key idea: DP does NOT create different datasets per-rank

In OptimusPrime (and also in common frameworks like DeepSpeed/Megatron), it is normal that **every rank holds the same Python `dataset` object/list in memory**. Data-parallel “sharding” is usually done by a **sampler**, i.e., by selecting **different indices** from the same dataset for each DP rank.

So, if you print `len(dataloader.dataset)` or the first few dataset strings on each rank, they can look identical and still be a correct DP setup.

### What matters: what each rank receives from the DataLoader iterator

What you should check is **the content of `batch` produced by**:

    for batch in dataloader:
        ...

If DP sharding is active, DP ranks within the same pipeline stage will typically see different `batch` content (different sample strings / different hashes).

### Global batch size (GBS) vs per-rank batch size

Most distributed training stacks treat DataLoader `batch_size` as **per-rank** batch size. Then, the “effective global batch” is typically:

\[
\text{GBS}_{\text{effective}} = \text{(per-rank batch)} \times \text{DP} \times \text{GAS}
\]

Where:
- **DP**: data parallel size
- **GAS**: gradient accumulation steps (number of micro-batches per optimizer step)

In this repo’s Llama example (`examples/pp_train_llama4.py`), `--batch-size` is intended to represent **GBS** in logs, but `Optimus_p.prepare_dataloader()` can be configured in two ways:

- **Mode A (GBS fixed, recommended)**: per-rank DataLoader batch = `GBS / DP`
  - DP increases → each rank processes fewer samples per step, keeping `GBS_effective` constant.
- **Mode B (GBS grows with DP)**: per-rank DataLoader batch = `GBS`
  - DP increases → `GBS_effective` becomes `GBS × DP` (more samples are consumed per step).
  - This is not “duplicate training”, but it changes the meaning of “step” (you consume DP× more data per step).

### PP note: DataLoader iteration happens on all PP stages

In the current pipeline code, **non-first PP stages also iterate the DataLoader**, even though only the first stage actually tokenizes/uses the raw text batch. As a result, you may observe the same raw batch appearing on stage 0 and stage 1 for the same DP-rank mapping; this is a pipeline implementation detail, not necessarily an error in DP sharding.

## Debug helpers for DP sharding (Llama example)

The file `examples/pp_train_llama4.py` includes debug switches to generate “proof logs” for DP sharding:

- **`--debug-dataset True`**: show that each rank holds the same dataset list in memory (expected in DP).
- **`--debug-batch True`**: show per-rank `len(batch)` and the “effective global batch used by stage 0”.
- **`--debug-batch-raw True`**: show raw batch contents and `sha1(batch)` per rank (strongest evidence of DP sharding).

Example:

    torchrun --standalone --nproc_per_node=8 --nnodes=1 --master_port=29500 \
      examples/pp_train_llama4.py --access-token $HF_ACCESS_TOKEN \
      --pp-degree 2 --dp-degree 4 --tp-degree 1 \
      --micro-batch-size 1 --batch-size 32 \
      --debug-dataset True --debug-batch True --debug-batch-raw True \
      --profile-cut True --profile-step 1

## Reference: how DeepSpeed and Megatron handle DP sampling (high-level)

- **DeepSpeed** (default path) uses `torch.utils.data.distributed.DistributedSampler(num_replicas=dp_world_size, rank=dp_rank)` and a per-rank DataLoader `batch_size`. Dataset objects can remain identical across ranks; the sampler sharding controls which samples each rank sees.
- **Megatron(-DeepSpeed)** often uses a custom batch sampler for pretraining: it conceptually builds a “global batch” and then slices it by `data_parallel_rank`, ensuring each DP rank gets a unique micro-batch slice.


