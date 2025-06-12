import os
import sys
import itertools
import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path: str = "summary.csv", out_dir: str = "plots/png") -> None:
    # ----------------------------- load & tidy --------------------------------
    df = pd.read_csv(csv_path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    # Normalise empty ft_variant cells
    df["ft_variant"] = df["ft_variant"].fillna("")

    # Construct a human-readable variant label
    df["variant"] = df["kind"].str.capitalize() + " (" + df["ft_variant"].str.lower() + ")"

    # Treat every covariance row with FT-size == 0 as "Covariance (comb)"
    zero_mask = (df["kind"] == "covariance") & (df["ft_tuning_size"] == 0)
    target_variant = "Covariance (comb)"
    df.loc[zero_mask, "variant"] = target_variant

    metric = "med"

    mean_metric = f"mean_{metric}"
    std_metric = f"std_{metric}"

    # ----------------------------- main curves --------------------------------
    curves_df = df.query("eval_ds == 'lcbench' and variant == @target_variant").copy()

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
    plt.title(f"Covariance (comb) - {metric} ± SD vs Fine-tuning Size")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_med_vs_ft_ctx.png"), dpi=300)
    plt.close()

    # ========== FIGURE 2: MeanMed ± SD vs Context (curves = FT-sizes) =========
    plt.figure()
    colour_cycle = itertools.cycle(plt.cm.tab10.colors)

    for ft_size, g in curves_df.groupby("ft_tuning_size", sort=True):
        agg = (
            g.groupby("eval_ctx", as_index=False)
            .agg(mean_med=(mean_metric, "mean"), std_med=(std_metric, "mean"))
            .sort_values("eval_ctx")
        )
        clr = next(colour_cycle)
        plt.errorbar(
            agg["eval_ctx"],
            agg[mean_metric],
            yerr=agg[std_metric],
            label=f"FT {ft_size}",
            marker="s",
            linestyle="-",
            capsize=3,
            color=clr,
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
    plt.title(f"Covariance (comb) - {metric} ± SD vs Context Length")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_med_vs_ctx_ft.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "summary.csv"
    main(csv_arg)
