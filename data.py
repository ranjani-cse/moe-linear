"""Load Tiny Shakespeare and serve batches."""
import torch


def load_data(path="data/input.txt", val_frac=0.1):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(len(data) * (1 - val_frac))

    return {
        "train": data[:n],
        "val": data[n:],
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": itos,
        "encode": encode,
        "decode": decode,
    }


def get_batch(data, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
