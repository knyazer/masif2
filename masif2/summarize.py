from collections import defaultdict
import os
from pathlib import Path

from tqdm import tqdm
from itertools import product
import equinox as eqx
import functools
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt

np.random.seed(0)


method_label = {
    "learned": "Learned",
    "covariance": "Covariance",
    "identity": "Identity",
    "ifbo": "IFBO",
}
methods = ["learned", "covariance", "identity", "ifbo"]
datasets = ["taskset", "lcbench", "pd1"]

ft_variants = ["comb", "full"]
ft_methods = ["learned", "covariance"]
ft_on = ["learned"]
ft_benched = ["lcbench", "taskset", "pd1"]
ft_tuning_size = [100, 400, 1600]
ft_context_size = [200, 400, 800, 1600, 3200]
CTX = [200, 400, 800, 1600, 3200]


def get_stuff_from_file(npz_path, *, borders, total, count, sigs):
    for s in sigs:
        if s not in total or s not in count:
            total[s] = 0
            count[s] = 0
    if not npz_path.exists():
        return None
    f = np.load(npz_path)
    all_probs = f["arr_0"].astype(np.float32)
    all_targets = f["arr_1"].astype(np.float32)
    print(npz_path, all_targets.shape)

    for probs, tgt in zip(all_probs, all_targets):
        probs, tgt = probs.squeeze(), tgt.squeeze()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_p = np.cumsum(sorted_probs)

        for sig in sigs:
            cutoff = np.searchsorted(cumsum_p, sig) + 1
            selected_indices = sorted_indices[:cutoff]
            bin_starts = borders[selected_indices]
            bin_ends = borders[selected_indices + 1]
            total[sig] += 1
            for st, en in zip(bin_starts, bin_ends):
                if st <= tgt < en:
                    count[sig] += 1
                    break

    # computing log likelihood related stuff
    lls = []
    for probs, tgt in zip(all_probs, all_targets):
        probs, tgt = probs.squeeze(), tgt.squeeze()
        idx = np.searchsorted(borders, tgt, side="right") - 1
        if 0 <= idx < len(probs):
            p = float(probs[idx])
            ll = np.log(max(p, 1e-12)) - np.log(borders[idx + 1] - borders[idx])
            lls.append(ll)

    if not lls:
        return None

    mean_ll = np.mean(lls) if lls else np.nan
    med_ll = np.median(lls) if lls else np.nan
    std_ll = np.std(lls) / np.sqrt(len(lls) - 1) if lls else np.nan

    n_boot = 1000
    meds = [np.median(np.random.choice(lls, size=len(lls), replace=True)) for _ in range(n_boot)]
    std_med = np.std(meds, ddof=1)

    return {
        "allocations": len(all_probs),
        "mean_ll": float(mean_ll),
        "median_ll": float(med_ll),
        "std_ll": float(std_ll),
        "med_std": float(std_med),
    }


def summarize_results_at(
    path="results/learned", dataset="lcbench", context_points=400, precision=3, spec={}
):
    path = Path(path)
    f = np.load(path / Path("borders.npz"))
    borders = f["arr_0"]

    res = {}

    sigs = np.arange(20) / 20 + 1.0 / 40
    total = {}
    count = {}
    benchmark_rows = []
    bench_dir = path / Path(dataset)
    if not bench_dir.exists():
        print(f"[x] {bench_dir} not found - skipping.")
        return None

    folders = os.listdir(bench_dir)
    folders.sort()
    for i, subbench_dir in enumerate(folders):
        npz_path = Path(bench_dir) / Path(subbench_dir) / f"ctx_{context_points}.npz"
        stuff = get_stuff_from_file(npz_path, borders=borders, total=total, count=count, sigs=sigs)
        if stuff is not None:
            benchmark_rows.append(stuff)

    if benchmark_rows == []:
        print(f"[x] {bench_dir} contains no useful folders - skipping")
        return None

    bdf = pd.DataFrame(benchmark_rows)
    try:
        mean_med = bdf["median_ll"].mean()
        mean_ll = bdf["mean_ll"].mean()
        std_ll = bdf["std_ll"].mean() / np.sqrt(len(bdf) - 1)
        std_med = bdf["med_std"].mean() / np.sqrt(len(bdf) - 1)

        print(f"{path}: LL: {mean_ll:.2f}+-{std_ll:.2f}; MMedLL {mean_med:.2f}+-{std_med:.2f}")
    except Exception as e:
        breakpoint()
        raise e

    arr = np.array(list(total.keys()))
    arr.sort()

    reliability = []
    for sig in arr:
        reliability.append(count[sig] / total[sig])
    # i would love here to just return a bunch of rows, with the spec:
    return {
        **spec,
        "std_ll": std_ll,
        "mean_ll": mean_ll,
        "std_med": std_med,
        "mean_med": mean_med,
        "reliability": np.array(reliability),
    }


def _format(val: float, err: float, bold: bool) -> str:
    """Format a value ± error with 2 decimals and optional boldface."""
    txt = f"{val:.2f}\\pm{err:.2f}"
    if bold:
        txt = rf"\mathbf{{{txt}}}"
    return f"${txt}$"


def make_table(summary) -> str:
    """Return a LaTeX table summarising *summary* in the format used in the paper.

    Parameters
    ----------
    summary : dict
        Nested dictionary *exactly* as described at the top of the file.
    """
    budgets = sorted(summary.keys())

    # ---------------------------------------------------------------------
    # Canonical orderings & user-friendly labels
    # ---------------------------------------------------------------------
    canonical_methods = ["learned", "covariance", "identity", "ifbo"]
    methods = [m for m in canonical_methods if any(m in summary[b] for b in budgets)]
    # append any extra methods not listed above, in alphabetical order
    extra_methods = {m for b in budgets for m in summary[b].keys()} - set(methods)
    methods.extend(sorted(extra_methods))

    method_label = {
        "learned": "Learned",
        "covariance": "Covariance",
        "identity": "Identity",
        "ifbo": "IFBO",
    }
    # fallback – capitalise first letter
    for m in methods:
        method_label.setdefault(m, m.capitalize())

    dataset_order = ["taskset", "lcbench", "pd1"]
    datasets = [
        d for d in dataset_order if any(d in summary[b][m] for b in budgets for m in methods)
    ]
    # same trick for extras
    extra_dsets = {d for b in budgets for m in methods for d in summary[b][m].keys()} - set(
        datasets
    )
    datasets.extend(sorted(extra_dsets))

    dataset_label = {
        "taskset": "Taskset",
        "lcbench": "LCBench",
        "pd1": "PD1",
    }
    for d in datasets:
        dataset_label.setdefault(d, d.capitalize())

    # ---------------------------------------------------------------------
    # Decide which cells to boldface
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Begin LaTeX generation – preamble
    # ---------------------------------------------------------------------
    lines = []
    push = lines.append

    push(r"\begin{table}[t]")
    push(r"\centering")
    push(r"\scriptsize")
    push(
        r"\caption{Grand mean log-likelihood (LL) and median dataset mean log-likelihood (MMedLL) across five context-size budgets.\\"
    )
    push(r"Values are mean $\,\pm\,$ bootstrapped standard error.\\")
    push(r"Boldface marks a method whose mean exceeds the runner-up by $\ge 2$ standard errors.}")
    push(r"\setlength{\tabcolsep}{4pt}")
    push(r"\begin{tabular}{lll" + "r" * len(budgets) + "}")
    push(r"\toprule")
    push(
        r"\multicolumn{3}{c}{} & \multicolumn{"
        + str(len(budgets))
        + r"}{c}{\textbf{Context size}}\\"
    )
    push(r"\cmidrule(lr){4-" + str(3 + len(budgets)) + r"}")
    header_cells = ["%d" % b for b in budgets]
    push(
        r"\textbf{Type} & \textbf{Dataset} & \textbf{Metric} & "
        + " & ".join(rf"\textbf{{{c}}}" for c in header_cells)
        + r"\\"
    )
    push(r"\midrule")

    # ---------------------------------------------------------------------
    # Body – iterate over methods → datasets → metrics
    # ---------------------------------------------------------------------
    metrics = (
        ("MMedLL", "median_ll", "std_med"),
        ("LL", "mean_ll", "std_ll"),
    )

    for mi, method in enumerate(methods):
        for di, dset in enumerate(datasets):
            # row prefix: Type on first dataset row, else blank
            type_cell = method_label[method] if di == 0 else ""
            # optional multirow if first dataset in method block
            if di == 0:
                # total rows in this method block = 2 * len(datasets)
                push(r"\multirow{" + str(2 * len(datasets)) + r"}{*}{" + type_cell + r"}")
            else:
                push("  ")  # indent for clarity, will be ignored by TeX

            for mi2, (metric_label, mean_key, err_key) in enumerate(metrics):
                if di > 0 or mi2 > 0:
                    # For subsequent rows we need the correct ampersand alignment
                    row = [" " * 2] * 3  # placeholders which we will overwrite
                else:
                    row = []
                # ------ 1) dataset name (only on first metric row) ------
                dataset_cell = rf"\multirow{{2}}{{*}}{{{dataset_label[dset]}}}" if mi2 == 0 else ""
                # ------ 2) metric label ------
                metric_cell = metric_label

                # ------ 3) values for each budget ------
                value_cells = []
                for b in budgets:
                    try:
                        cell = summary[b][method][dset]
                        mean, err = 0, 0
                        for val in cell:
                            mean += float(val[mean_key])
                            err += float(val[err_key])
                        mean /= len(cell)
                        err /= len(cell)
                    except KeyError:
                        mean, err = float("nan"), float("nan")
                    value_cells.append(_format(mean, err, False))

                # assemble the row
                if mi2 == 0:
                    prefix = " & ".join([dataset_cell, metric_cell])
                else:
                    prefix = " & " + metric_cell
                push("  & " + prefix + " & " + " & ".join(value_cells) + r"\\")

            # add small vertical space between dataset blocks
            if di < len(datasets) - 1:
                push(r"  [2pt]")
        # horizontal line between methods except last
        if mi < len(methods) - 1:
            push(r"\midrule")

    # ---------------------------------------------------------------------
    # Finish the table
    # ---------------------------------------------------------------------
    push(r"\bottomrule")
    push(r"\end{tabular}")
    push(r"\end{table}")

    return "\n".join(lines)


CB_PALETTE = [
    "#377eb8",
    "#e41a1c",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]


def make_perf_context_size_plot(summary):
    budgets = np.array(sorted(summary.keys()))

    # Prepare containers
    avg_mmedll = {m: [] for m in methods}
    avg_std = {m: [] for m in methods}

    # Aggregate per-method statistics
    for b in budgets:
        for m in methods:
            data = summary[b].get(m, None)
            if not data:
                avg_mmedll[m].append(np.nan)
                avg_std[m].append(np.nan)
                continue

            # Per-dataset mean of median_ll and std
            median_vals = [
                np.mean([rec["median_ll"] for rec in recs]) for recs in data.values() if recs
            ]
            std_vals = [
                np.mean([rec["std_med"] for rec in recs]) / np.sqrt(len(recs))
                for recs in data.values()
                if recs
            ]

            avg_mmedll[m].append(np.nanmean(median_vals) if median_vals else np.nan)
            avg_std[m].append(np.nanmean(std_vals) if std_vals else np.nan)

    # Convert to arrays for plotting
    for m in methods:
        avg_mmedll[m] = np.array(avg_mmedll[m])
        avg_std[m] = np.array(avg_std[m])

    # --- Plot styling for scientific-paper readiness ---
    plt.figure(figsize=(6, 4))
    plt.rc("font", family="serif", size=10)
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)
    plt.rc("legend", fontsize=9)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Plot each method with error shading
    for idx, m in enumerate(methods):
        color = CB_PALETTE[idx % len(CB_PALETTE)]
        y = avg_mmedll[m]
        yerr = avg_std[m]

        ax.plot(budgets, y, marker="o", linestyle="-", label=method_label.get(m, m), color=color)
        ax.fill_between(budgets, y - yerr, y + yerr, alpha=0.2, color=color)

    # Use logarithmic scale on x-axis with explicit ticks
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    # Format tick labels as integers if budgets are integral
    ax.set_xticklabels([str(int(b)) for b in budgets])

    # Labels and legend
    ax.set_xlabel("Context size")
    ax.set_ylabel("Average MMedLL")
    ax.set_title("Average MMedLL vs. Context Size")
    ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig("plots/perf_vs_context_size.svg", format="svg", dpi=300)
    plt.savefig("plots/png/perf_vs_context_size.png", format="png", dpi=300)
    plt.close()


def make_reliability_plots(rel):
    """
    Plot all models' averaged reliability curves (with ±1 std shading)
    on the same NeurIPS-style, colorblind-friendly figure.
    """
    # Compute predicted probability bins
    num_bins = len(next(iter(rel.values()))["learned"]["taskset"])
    pred_probs = np.linspace(0.025, 0.975, num_bins)

    # Models
    models = sorted(next(iter(rel.values())).keys())

    # Compute average and std curves for each model
    avg_curves = {}
    std_curves = {}
    for model in models:
        curves = np.array([[rel[cs][model][ds] for cs in rel] for ds in datasets])
        avg_curves[model] = curves.mean(axis=(0, 1))
        std_curves[model] = curves.std(axis=(0, 1)) / np.sqrt(curves.shape[0] * curves.shape[1])

    # Figure & style
    plt.figure(figsize=(6, 4))
    plt.rc("font", family="serif", size=10)
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)
    plt.rc("legend", fontsize=9)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Perfect calibration
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, alpha=0.7)

    # Plot each model
    for idx, model in enumerate(models):
        color = CB_PALETTE[idx % len(CB_PALETTE)]
        y = avg_curves[model]
        yerr = std_curves[model]
        ax.plot(pred_probs, y, linestyle="-", label=model.capitalize(), color=color)
        ax.fill_between(pred_probs, y - yerr, y + yerr, alpha=0.2, color=color)

    # Labels, legend
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0.025, 0.975)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig("plots/reliability.svg", format="svg", dpi=300)
    plt.savefig("plots/png/reliability.png", format="png", dpi=300)
    plt.close()


def make_reliability_per_context(rel, contexts=CTX):
    # Predicted probability bins
    num_bins = len(next(iter(rel.values()))["covariance"]["taskset"])
    pred_probs = np.linspace(0.025, 0.975, num_bins)

    models = ["covariance", "ifbo"]

    # Figure and style
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.rc("font", family="serif", size=10)
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)
    plt.rc("legend", fontsize=9)

    for ax, model in zip(axes, models):
        # Spines and grid
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=1)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, alpha=0.7)

        # Plot each selected context size
        for idx, cs in enumerate(contexts):
            color = CB_PALETTE[idx % len(CB_PALETTE)]
            y_agg = np.array([rel[cs][model][ds] for ds in datasets])
            y = np.mean(y_agg, axis=0)
            yerr = np.std(y_agg) / (len(y_agg) * 4)  # 4 is number of curves
            ax.plot(pred_probs, y, linestyle="-", label=f"{cs}", color=color)
            ax.fill_between(pred_probs, y - yerr, y + yerr, alpha=0.2, color=color)

        ax.set_title(model.capitalize())
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_xlim(0.025, 0.975)
        ax.set_ylim(0, 1)
        ax.legend(title="Context size", frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig("plots/reliability_per_context.svg", format="svg", dpi=300)
    plt.savefig("plots/png/reliability_per_context.png", format="png", dpi=300)
    plt.close()


def make_reliability_per_dataset(rel):
    num_bins = len(next(iter(rel.values()))["covariance"]["taskset"])
    pred_probs = np.linspace(0.025, 0.975, num_bins)

    models = ["learned", "covariance", "identity", "ifbo"]

    # Figure and style
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    plt.rc("font", family="serif", size=10)
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)
    plt.rc("legend", fontsize=9)

    for ax, ds in zip(axes, datasets):
        # Spines and grid
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=1)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, alpha=0.7)

        for idx, model in enumerate(models):
            # Plot each selected context size
            color = CB_PALETTE[idx % len(CB_PALETTE)]
            y_agg = np.array([rel[cs][model][ds] for cs in CTX])
            y = np.mean(y_agg, axis=0)
            yerr = np.std(y_agg) / (len(y_agg) * 4)  # 4 is number of curves
            ax.plot(pred_probs, y, linestyle="-", label=f"{model}", color=color)
            ax.fill_between(pred_probs, y - yerr, y + yerr, alpha=0.2, color=color)

        ax.set_title(ds.capitalize())
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_xlim(0.025, 0.975)
        ax.set_ylim(0, 1)
        ax.legend(title="Context size", frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig("plots/reliability_per_dataset.svg", format="svg", dpi=300)
    plt.savefig("plots/png/reliability_per_dataset.png", format="png", dpi=300)
    plt.close()


def make_perf_context_size_subplots(summary):
    # Extract sorted context sizes (budgets)
    budgets = np.array(sorted(summary.keys()))

    # Determine the set of all datasets
    datasets = sorted({ds for b in budgets for m in methods for ds in summary[b].get(m, {}).keys()})

    # Set up subplots: one per dataset
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    # Styling for scientific-paper readiness
    plt.rc("font", family="serif", size=10)
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)
    plt.rc("legend", fontsize=9)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ax.set_title(dataset)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=1)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Accumulate per-method stats for this dataset
        for m_idx, m in enumerate(methods):
            y_vals = []
            y_errs = []
            for b in budgets:
                data = summary[b].get(m, {})
                recs = data.get(dataset, None)
                if not recs:
                    y_vals.append(np.nan)
                    y_errs.append(np.nan)
                else:
                    # Compute median of median_ll and stderr of std_med
                    median_ll = np.mean([r["median_ll"] for r in recs])
                    stderr = np.mean([r["std_med"] for r in recs]) / np.sqrt(len(recs))
                    y_vals.append(median_ll)
                    y_errs.append(stderr)

            y = np.array(y_vals)
            yerr = np.array(y_errs)
            color = CB_PALETTE[m_idx % len(CB_PALETTE)]

            ax.plot(budgets, y, linestyle="-", label=method_label.get(m, m), color=color)
            ax.fill_between(budgets, y - yerr, y + yerr, alpha=0.2, color=color)

        # Log scale on x-axis
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(int(b)) for b in budgets])
        ax.set_xlabel("Context size")
        if idx == 0:
            ax.set_ylabel("Average MMedLL")

    # Shared legend below subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False)

    plt.tight_layout()
    fig.savefig("plots/perf_vs_context_size_per_dataset.svg", format="svg", dpi=300)
    fig.savefig("plots/png/perf_vs_context_size_per_dataset.png", format="png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # start with finetuning stuff
    ft_variants = ["comb"]
    ft_methods = ["learned", "covariance"]
    ft_on = ["taskset"]
    ft_benched = ["taskset"]
    ft_tuning_sizes = [100, 800, 1600]
    CTX = [200, 400, 800, 1600, 3200]
    all_lst_prod = list(product(CTX, ft_benched, ft_on, ft_tuning_sizes, ft_methods, ft_variants))

    rows = []
    for eval_ctx, eval_ds, ft_tuning_ds, ft_tuning_size, kind, ft_variant in tqdm(all_lst_prod):
        p = f"results/finetuned/{ft_tuning_ds}/{kind}/{ft_variant}/{ft_tuning_size}/{eval_ctx}"
        try:
            row = summarize_results_at(
                p,
                dataset=eval_ds,
                context_points=eval_ctx,
                spec={
                    "eval_ctx": eval_ctx,
                    "eval_ds": eval_ds,
                    "ft_tuning_ds": ft_tuning_ds,
                    "ft_tuning_size": ft_tuning_size,
                    "kind": kind,
                    "ft_variant": ft_variant,
                },
            )
            if row is None:
                breakpoint()
        except Exception as e:
            print(p, e)
            breakpoint()
        rows.append(row)

    all_lst_prod = list(product(CTX, ["taskset"], ["learned", "covariance", "ifbo"]))
    for ctx, ds, kind in tqdm(all_lst_prod):
        p = f"results/{kind}"
        try:
            row = summarize_results_at(
                p,
                dataset=ds,
                context_points=ctx,
                spec={
                    "eval_ctx": ctx,
                    "eval_ds": ds,
                    "ft_tuning_ds": None,
                    "ft_tuning_size": 0,
                    "kind": kind,
                    "ft_variant": None,
                },
            )
            if row is None:
                breakpoint()
        except Exception as e:
            print(e)
            continue
        rows.append(row)

    res_df = pd.DataFrame(rows)
    res_df.to_csv("summary_taskset.csv")

    print(make_table(summary))
    make_perf_context_size_plot(summary)
    make_perf_context_size_subplots(summary)

    make_reliability_plots(rel)
    make_reliability_per_context(rel)
    make_reliability_per_dataset(rel)
