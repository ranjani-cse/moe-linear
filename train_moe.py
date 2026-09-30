"""Train the MoE model, log loss and expert usage to CSV."""
import csv
import os
import time
import torch
import torch.nn.functional as F
from data import load_data, get_batch
from dense import DenseModel
from moe_model import MoEModel, convert_from_dense


def train(
    max_steps=2000,
    batch_size=32,
    block_size=64,
    lr=3e-4,
    balance_coef=0.01,
    log_path="logs/moe.csv",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    d = load_data()
    dense = DenseModel(vocab_size=d["vocab_size"], block_size=block_size).to(device)
    dense.load_state_dict(torch.load("dense_model.pt", map_location=device))

    model = MoEModel(vocab_size=d["vocab_size"], block_size=block_size,
                     n_experts=4, top_k=2).to(device)
    model = convert_from_dense(model, dense)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"device: {device} | params: {model.num_params():,}")

    f = open(log_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "train_loss", "val_loss", "balance_loss", "tokens_per_sec",
                     "expert_0", "expert_1", "expert_2", "expert_3"])

    t0 = time.time()
    tokens_seen = 0
    losses = []

    for step in range(1, max_steps + 1):
        model.train()
        x, y = get_batch(d["train"], block_size, batch_size, device)
        _, loss = model(x, y)
        bal = model.moe.load_balance_loss()
        total = loss + balance_coef * bal

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        tokens_seen += x.numel()
        losses.append(loss.item())

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(d["val"], block_size, batch_size, device)
                _, vloss = model(vx, vy)

            usage = F.one_hot(model.moe.last_topk_idx, 4).float().sum(dim=(0, 1))
            usage = (usage / usage.sum()).tolist()

            train_avg = sum(losses[-50:]) / len(losses[-50:])
            writer.writerow([step, round(train_avg, 4), round(vloss.item(), 4),
                             round(bal.item(), 4), round(tps)]
                            + [round(u, 3) for u in usage])
            f.flush()
            print(f"step {step:5d} | train {train_avg:.4f} | val {vloss.item():.4f} | "
                  f"bal {bal.item():.3f} | {tps:.0f} tok/s | usage {[round(u,2) for u in usage]}")

    f.close()
    final = sum(losses[-50:]) / 50
    print(f"\ndone. final train loss: {final:.4f}")
    return model


if __name__ == "__main__":
    train()
