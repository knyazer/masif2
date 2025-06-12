import os
import sys
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal


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
    metric = "med"

    mean_metric = f"mean_{metric}"
    std_metric = f"std_{metric}"

    # ----------------------------- main curves --------------------------------
    curves_df = df.query("eval_ds == 'lcbench' and variant == @variant").copy()

    # Derive IFBO baseline (dashed black)
    baseline = (
        df.query('kind == "ifbo" and eval_ds == "lcbench" and ft_tuning_ds.isna()')
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
    plt.savefig(os.path.join(out_dir, f"mean_med_vs_ft_ctx_{variant}.png"), dpi=300)
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
    plt.savefig(os.path.join(out_dir, f"mean_med_vs_ctx_ft_{variant}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    for kind in ["covariance", "learned"]:
        for method in ["comb", "full"]:
            plot_single(kind, method)
