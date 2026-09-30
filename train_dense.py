"""Train the dense model, log loss to CSV, save the weights."""
import csv
import os
import time
import torch
from data import load_data, get_batch
from dense import DenseModel


def train(
    max_steps=2000,
    batch_size=32,
    block_size=64,
    lr=3e-4,
    log_path="logs/dense.csv",
    save_path="dense_model.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    d = load_data()
    model = DenseModel(vocab_size=d["vocab_size"], block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"device: {device} | params: {model.num_params():,}")

    f = open(log_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "train_loss", "val_loss", "tokens_per_sec"])

    t0 = time.time()
    tokens_seen = 0
    losses = []

    for step in range(1, max_steps + 1):
        model.train()
        x, y = get_batch(d["train"], block_size, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
            train_avg = sum(losses[-50:]) / len(losses[-50:])
            writer.writerow([step, round(train_avg, 4), round(vloss.item(), 4), round(tps)])
            f.flush()
            print(f"step {step:5d} | train {train_avg:.4f} | val {vloss.item():.4f} | {tps:.0f} tok/s")

    f.close()
    torch.save(model.state_dict(), save_path)
    final = sum(losses[-50:]) / 50
    print(f"\ndone. final train loss: {final:.4f}")
    print(f"saved to {save_path}")
    return model


if __name__ == "__main__":
    train()
