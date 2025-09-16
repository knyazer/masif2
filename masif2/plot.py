import os
import sys
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from .main import load_model, Config
from jax import numpy as jnp
import numpy as np
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings


def _dataset_labels(name):
    """Return tick-label list (or None) for a known hyper-parameter dataset."""
    name = str(name).lower() if name is not None else None

    taskset_labels = [
        r"$\alpha$ (lr)",
        r"$\beta_1$",
        r"$\beta_2$",
        r"$\epsilon$",
        "L2 reg",
        "L1 reg",
        r"$\lambda_{\mathrm{exp}}$",
        r"$\lambda_{\mathrm{lin}}$",
    ]

    lcbench_labels = [
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_decay",
        "n_layers",
        "max_units",
        "dropout",
    ]

    if name == "lcbench":
        return lcbench_labels
    if name == "taskset":
        return taskset_labels
    print(f"No identifiable labels for {ds}")
    return None


def plot_cov(model, out_dir="plots/png", title="", ds=None):
    # loads and plots a covariance as a heatmap
    tri = jnp.tril(model.inv_cov_prm)
    inv_cov = (tri @ tri.T) + jnp.eye(len(model.inv_cov_prm)) * 1e-7
    cov = np.asarray(jnp.linalg.pinv(inv_cov).block_until_ready())

    labels = _dataset_labels(ds)
    n = len(labels) if labels is not None else len(cov)
    cov = cov[:n, :n]

    vmax = np.max(np.abs(cov))
    vmin = -vmax

    # nice diverging palette with white centre
    cmap = sns.color_palette("vlag", as_cmap=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cov,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws=dict(label="covariance"),
        ax=ax,
    )

    if labels and len(labels) == cov.shape[0]:
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
        ax.set_yticklabels(labels, fontsize=9, ha="right", rotation=0)
    else:
        # fallback to numeric ticks
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax.set_title(title, pad=12, fonsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_{title}.png"), dpi=300)


def make_agg(exp_name="exp01"):
    version_files = Path().rglob(f"models/{exp_name}/**/.version")
    agg = []
    for version_file in tqdm(version_files):
        folder_path = Path(version_file).parent
        tqdm.write(f"Aggregating: {folder_path}")
        config = Config.load(folder_path)
        results = config.load_results()
        if results is None:
            tqdm.write(f"Skipping - results cannot be loaded for {folder_path}")
            continue
        results["model_kind"] = config.model_kind
        results["ft_kind"] = config.ft_kind
        results["ft_dataset"] = config.ft_dataset
        results["ft_trained_for"] = config.ft_trained_for
        results["ft_trained_on"] = config.ft_trained_on
        agg.append(results)
    return pd.concat(agg, axis=0, ignore_index=True)


def plot_predictive_by_dataset_ctx_and_metric(agg):
    # non finetuned
    agg_no_ft = agg[agg["ft_kind"].isna()]
    assert len(agg_no_ft) != 0 and len(agg) != len(agg_no_ft)
    agg = agg_no_ft

    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # Get unique datasets and metrics
    datasets = sorted(agg["dataset"].unique())
    metrics = ["ll", "mmedll"]

    # Create figure with gridspec for better control
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(
        len(metrics), len(datasets), figure=fig, hspace=0.3, wspace=0.3, bottom=0.15, top=0.9
    )

    # Create subplots
    axes = {}
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[i, j])
            axes[(metric, dataset)] = ax

    # Plot data for each metric and dataset
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = axes[(metric, dataset)]

            # Filter data for this dataset
            data = agg[agg["dataset"] == dataset]

            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Group by context size and plot
            ctx_sizes = sorted(data["ctx_size"].unique())
            values = [data[data["ctx_size"] == ctx][metric].values for ctx in ctx_sizes]

            # Plot by method using different colors
            if len(data) > 0:
                methods = sorted(data["model_kind"].unique())
                colors = ["blue", "red"]  # Two distinct colors for the two methods

                # Map model_kind to display labels
                method_labels = {"cov": "Covariance", "learned": "Learned"}

                for method_idx, method in enumerate(methods):
                    method_data = data[data["model_kind"] == method]
                    if len(method_data) == 0:
                        continue

                    ctx_sizes_method = sorted(method_data["ctx_size"].unique())

                    # Calculate mean values for line plot
                    mean_values = []
                    for ctx in ctx_sizes_method:
                        ctx_data = method_data[method_data["ctx_size"] == ctx][metric].values
                        if len(ctx_data) > 0:
                            mean_values.append(np.mean(ctx_data))
                        else:
                            mean_values.append(np.nan)

                    # Plot line
                    label = method_labels.get(method, method)
                    ax.plot(
                        ctx_sizes_method,
                        mean_values,
                        color=colors[method_idx],
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=label,
                    )

                ax.set_xscale("log")
                ax.set_xticks(ctx_sizes)
                ax.set_xticklabels(ctx_sizes)

            # Set labels and title
            if i == 0:  # Top row
                ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
            if j == 0:  # Left column
                ax.set_ylabel(metric.upper(), fontsize=12)
            if i == len(metrics) - 1:  # Bottom row
                ax.set_xlabel("Context Size", fontsize=12)

    # Add legend for methods
    handles, labels = axes[(metrics[0], datasets[0])].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(labels)
        )

    # Use tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("perf_vs_context_size_per_dataset.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    agg = make_agg()
    plot_predictive_by_dataset_ctx_and_metric(agg)
