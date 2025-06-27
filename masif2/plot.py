import os
import sys
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from .main import load_model
from jax import numpy as jnp
import numpy as np
import seaborn as sns


def make_variant(kind, method):
    return kind.capitalize() + " (" + method.lower() + ")"


def make_df(
    csv_path,
    kind: Literal["covariance", "learned"],
    method: Literal["comb", "full"],
):
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    # Normalise empty ft_variant cells
    df["ft_variant"] = df["ft_variant"].fillna("")

    # Construct a human-readable variant label
    df["variant"] = make_variant(df["kind"].str, df["ft_variant"].str)

    zero_mask = (df["kind"] == kind) & (df["ft_tuning_size"] == 0)
    df.loc[zero_mask, "variant"] = make_variant(kind, method)
    return df


def plot_single(
    kind: Literal["covariance", "learned"],
    method: Literal["comb", "full"],
    csv_path: str = "summary.csv",
    out_dir: str = "plots/png",
) -> None:
    df = make_df(csv_path, kind, method)
    variant = make_variant(kind, method)
    # ----------------------------- load & tidy --------------------------------
    ds = "taskset"
    metric = "med"

    mean_metric = f"mean_{metric}"
    std_metric = f"std_{metric}"

    # ----------------------------- main curves --------------------------------
    curves_df = df.query("eval_ds == @ds and variant == @variant").copy()

    # Derive IFBO baseline (dashed black)
    baseline = (
        df.query('kind == "ifbo" and eval_ds == @ds and ft_tuning_ds.isna()')
        .groupby("eval_ctx", as_index=False)[mean_metric]
        .mean()
    )

    # ----------------------------- output dir ---------------------------------
    os.makedirs(out_dir, exist_ok=True)

    # ========== FIGURE 1: MeanMed ± SD vs FT-size (curves = contexts) =========
    plt.figure()
    cmap = plt.get_cmap("tab10")

    for i, (ctx_len, g) in enumerate(curves_df.groupby("eval_ctx", sort=True)):
        agg = (
            g.groupby("ft_tuning_size", as_index=False)
            .agg(mean_med=(mean_metric, "mean"), std_med=(std_metric, "mean"))
            .sort_values("ft_tuning_size")
        )
        print(agg[mean_metric])
        plt.errorbar(
            agg["ft_tuning_size"],
            agg[mean_metric],
            yerr=agg[std_metric],
            label=f"CTX {ctx_len}",
            marker="o",
            linestyle="-",
            capsize=3,
            color=cmap(i % 10),
        )

        # Dashed IFBO baseline (horizontal) for this context
        y_base = baseline.loc[baseline["eval_ctx"] == ctx_len, mean_metric]
        if not y_base.empty:
            plt.axhline(
                y=y_base.values[0],
                color=cmap(i % 10),
                linestyle="--",
                alpha=0.6,
            )

    plt.xlabel("Fine-tuning set size")
    plt.ylabel(f"Mean {metric}")
    plt.title(f"{variant} - {metric} +- SD vs Fine-tuning Size")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ds}_mean_med_vs_ft_ctx_{variant}.png"), dpi=300)
    plt.close()

    # ========== FIGURE 2: MeanMed ± SD vs Context (curves = FT-sizes) =========
    plt.figure()
    colour_cycle = itertools.cycle(plt.cm.tab10.colors)

    for ft_size in reversed(sorted(curves_df["ft_tuning_size"].unique())):
        g = curves_df[curves_df["ft_tuning_size"] == ft_size]
        eval_ctx = sorted(g["eval_ctx"].unique())
        y, std = [], []
        for ctx in eval_ctx:
            indexed = g[g["eval_ctx"] == ctx]
            y.append(indexed[mean_metric].mean())
            std.append(indexed[std_metric].mean())

        clr = next(colour_cycle)
        plt.errorbar(
            eval_ctx,
            y,
            yerr=std,
            label=f"FT {ft_size}",
            marker="s",
            linestyle="-",
            capsize=3,
            color=clr,
            alpha=0.7,
        )

    # Single dashed IFBO curve (varies with context)
    plt.plot(
        baseline["eval_ctx"],
        baseline[mean_metric],
        color="black",
        linestyle="--",
        label="IFBO baseline",
    )

    plt.xlabel("Context length (tokens)")
    plt.ylabel(f"Mean {mean_metric}")
    plt.title(f"{variant} - {metric} ± SD vs Context Length")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ds}_mean_med_vs_ctx_ft_{variant}.png"), dpi=300)
    plt.close()


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
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9, ha="right", rotation=0)
    else:
        # fallback to numeric ticks
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_{title}.png"), dpi=300)


if __name__ == "__main__":
    for ds in ["taskset"]:
        for method in ["covariance"]:
            for kind in ["comb"]:
                for n_data in [100, 400, 1600]:
                    name = f"finetuned/{ds}/{method}/{kind}/{n_data}"
                    model = load_model(f"models/{name}/model", kind=method)
                    plot_cov(model, ds=ds, title=f"{ds}_{method}_{kind}_ft{n_data}")

    model = load_model("models/masif_covariance.eqx", kind="covariance")
    plot_cov(model, title="covariance_comb_ft0")

    for method in ["covariance", "learned"]:
        for kind in ["comb"]:
            plot_single(method, kind, csv_path="summary_taskset.csv")
