"""Plot dense vs MoE loss curves."""
import pandas as pd
import matplotlib.pyplot as plt


def plot():
    dense = pd.read_csv("logs/dense.csv")
    moe = pd.read_csv("logs/moe.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(dense["step"], dense["train_loss"], label="Dense (train)", alpha=0.7)
    axes[0].plot(dense["step"], dense["val_loss"], label="Dense (val)", alpha=0.7)
    axes[0].plot(moe["step"], moe["train_loss"], label="MoE (train)", alpha=0.7)
    axes[0].plot(moe["step"], moe["val_loss"], label="MoE (val)", alpha=0.7)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Dense vs MoE — loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for i in range(4):
        axes[1].plot(moe["step"], moe[f"expert_{i}"], label=f"expert {i}")
    axes[1].axhline(0.25, color="black", linestyle="--", alpha=0.5, label="ideal (0.25)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("fraction of tokens")
    axes[1].set_title("MoE — expert usage")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=120)
    print("saved loss_curves.png")


if __name__ == "__main__":
    plot()
