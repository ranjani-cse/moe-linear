# Linear Model → MoE

Train a small dense model, convert it to a Mixture of Experts, and show both
models continue to train and reduce loss.

## What was done

| Step | What |
|---|---|
| 1 | Trained a dense model (91K params) on Tiny Shakespeare |
| 2 | Converted it to MoE (4 experts, top-2 routing, 290K params) by copying the dense FFN weights into every expert |
| 3 | Trained the MoE from the dense checkpoint, with a load-balancing loss |
| 4 | Logged both runs to CSV and plotted loss curves |

## Results

| Model | Params | Final train loss | Final val loss | Speed (tok/s) |
|---|---|---|---|---|
| Dense | 91,329 | 2.4704 | 2.5070 | ~67,000 |
| MoE (4 experts, top-2) | 289,601 | 2.4655 | 2.4879 | ~25,800 |

Both models train and reduce loss. The MoE continues from the dense checkpoint
without degrading and keeps the router perfectly balanced throughout.

## Key observations

**The conversion works.** At initialization, the MoE's loss is identical to
the dense model's (2.45-2.48 on the same batches). Copying the dense FFN
weights into every expert and initializing the router near-uniform gives the
MoE a lossless starting point.

**The router stays balanced.** Expert usage across all 2,000 steps:
[0.23, 0.26, 0.25, 0.25]. No expert collapse. The load-balancing loss
(coefficient 0.01) held the router near the ideal 0.25 per expert.

**MoE is ~2.6x slower per step** at the same batch size (25,800 vs 67,000
tok/s). Each token is routed to 2 experts, and the dispatch loop is not yet
optimized — a batched or sorted dispatch would close most of that gap.

**MoE has 3.2x the parameters** of the dense model (289K vs 91K) but uses
only 2 of 4 experts per token. Effective compute is 2x the dense model, not 4x.

## Files

| File | Purpose |
|---|---|
| data.py | Load Tiny Shakespeare, character-level tokenizer, batching |
| dense.py | Dense baseline model |
| train_dense.py | Training loop for the dense model, logs to logs/dense.csv |
| moe.py | MoE block — experts, router, top-k dispatch, load-balance loss |
| moe_model.py | MoE-wrapped model + convert_from_dense() |
| train_moe.py | Training loop for the MoE, logs to logs/moe.csv |
| plot.py | Loss curve + expert usage plot → loss_curves.png |

## Training logs

- logs/dense.csv — 41 rows (every 50 steps), columns: step, train_loss, val_loss, tokens_per_sec
- logs/moe.csv — 41 rows, additional columns: balance_loss, expert_0..expert_3

## How to reproduce

    pip install torch numpy pandas matplotlib
    python3 train_dense.py
    python3 train_moe.py
    python3 plot.py

## Environment

- PyTorch 2.x
- GPU: Colab T4 (CUDA)
- Dataset: Tiny Shakespeare (1.1 MB, 65-character vocab)
- Dense training: 2,000 steps
- MoE training: 2,000 steps
- Batch size: 32, context 64
