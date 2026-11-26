"""Plot validation accuracy across training sizes and seeds."""

import matplotlib.pyplot as plt


def plot_learning_curve(summary, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, rows in summary.groupby("method", sort=False):
        rows = rows.sort_values("train_size")
        sizes = rows["train_size"].to_numpy()
        means = rows["mean_accuracy"].to_numpy()

        (line,) = ax.plot(sizes, means, marker="o", label=method)
        ax.fill_between(
            sizes,
            rows["min_accuracy"].to_numpy(),
            rows["max_accuracy"].to_numpy(),
            color=line.get_color(),
            alpha=0.15,
        )

    ax.set(
        xlabel="Training examples",
        ylabel="Validation accuracy",
        title="Ticket classification: mean and range across three seeds",
        ylim=(0, 1),
    )
    ax.set_xticks(sorted(summary["train_size"].unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
