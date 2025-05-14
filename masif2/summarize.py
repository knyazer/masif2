import os
from pathlib import Path

import equinox as eqx
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


def summarize_results(
    name="default", context_points=400, results_root="results", benchmarks=None, precision=3
):
    if benchmarks is None:
        root_dir = Path(results_root) / name
        if not root_dir.exists():
            print(f"[x] {root_dir} not found - nothing to summarise.")
            return
        benchmarks = [p.name for p in root_dir.iterdir() if p.is_dir()]

    import ast

    f = np.load(Path(results_root) / Path(name) / Path("borders.npz"))
    borders = f["arr_0"]

    res = {}

    for benchmark in benchmarks:
        benchmark_rows = []
        bench_dir = Path(results_root) / name / benchmark
        if not bench_dir.exists():
            print(f"[x] {bench_dir} not found - skipping.")
            continue

        for i, subbench_dir in enumerate(bench_dir.iterdir()):
            if i > 3:
                break
            if not subbench_dir.is_dir():
                continue
            npz_path = subbench_dir / f"ctx_{context_points}.npz"
            if not npz_path.exists():
                continue
            f = np.load(npz_path)
            all_probs = f["arr_0"].astype(np.float32)
            all_targets = f["arr_1"].astype(np.float32)

            lls = []
            for probs, tgt in zip(all_probs, all_targets):
                probs, tgt = probs.squeeze(), tgt.squeeze()
                idx = np.searchsorted(borders, tgt, side="right") - 1
                if 0 <= idx < len(probs):
                    p = float(probs[idx])
                    ll = np.log(max(p, 1e-12)) - np.log(borders[idx + 1] - borders[idx])
                    lls.append(ll)

            if not lls:
                continue

            mean_ll = np.mean(lls) if lls else np.nan
            med_ll = np.median(lls) if lls else np.nan
            std_ll = np.std(lls) / np.sqrt(len(lls) - 1) if lls else np.nan

            n_boot = 1000
            meds = [
                np.median(np.random.choice(lls, size=len(lls), replace=True)) for _ in range(n_boot)
            ]
            std_med = np.std(meds, ddof=1)
            benchmark_rows.append(
                {
                    "benchmark": benchmark,
                    "dataset": subbench_dir.name,
                    "allocations": len(all_probs),
                    "mean_ll": round(mean_ll, precision) if not np.isnan(mean_ll) else "-",
                    "median_ll": round(med_ll, precision) if not np.isnan(med_ll) else "-",
                    "std_ll": std_ll if not np.isnan(std_ll) else "-",
                    "std_med": std_med,
                    "lls": np.array(lls, dtype=np.float16),
                }
            )

        if benchmark_rows == []:
            continue
        bdf = pd.DataFrame(benchmark_rows)
        res[benchmark] = benchmark_rows
        try:
            meanmed = bdf["median_ll"].mean()
            mean = bdf["mean_ll"].mean()
            epi_std = bdf["mean_ll"].std() / np.sqrt(len(bdf))
            mean_std = bdf["std_ll"].mean() / np.sqrt(len(bdf) - 1)
            med_std = bdf["std_med"].mean() / np.sqrt(len(bdf) - 1)

            print(
                f"{name}/{benchmark}: LL: {mean:.2f}+-{mean_std:.2f}; MMedLL {meanmed:.2f}+-{med_std:.2f}"
            )
        except Exception as e:
            breakpoint()
            pass

    return res


def _format(val: float, err: float, bold: bool) -> str:
    """Format a value ± error with 2 decimals and optional boldface."""
    txt = f"{val:.2f}\\pm{err:.2f}"
    if bold:
        txt = rf"\mathbf{{{txt}}}"
    return f"${txt}$"


def _find_bold_cells(summary, budgets, methods, datasets):
    """Return mapping (dataset, metric, budget) → method that should be bold."""
    return {}
    bold = {}

    for dset in datasets:
        for metric, mean_key, err_key in (
            ("LL", "mean_ll", "std_ll"),
            ("MMedLL", "median_ll", "std_med"),
        ):
            for b in budgets:
                # Collect all (method, mean, stderr)
                rows = []
                for m in methods:
                    mean, std = 0, 0
                    try:
                        cell = summary[b][m][dset]
                        for v in cell:
                            mean += v[mean_key]
                            std += v[err_key]
                        mean /= len(cell)
                        std /= len(cell)
                    except KeyError:
                        continue  # allow for sparse entries
                    rows.append((m, mean, std))
                if len(rows) < 2:
                    continue  # cannot compute runner-up
                # sort by mean desc
                rows.sort(key=lambda t: t[1], reverse=True)
                best_m, best_val, best_err = rows[0]
                runner_val = rows[1][1]
                if best_val - runner_val >= 2 * best_err:
                    bold[(dset, metric, b)] = best_m
    return bold


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
    bold_cells = _find_bold_cells(summary, budgets, methods, datasets)

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
                    bold = bold_cells.get((dset, metric_label, b)) == method
                    value_cells.append(_format(mean, err, bold))

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
    plt.close()


if __name__ == "__main__":
    summary = {}
    for ctx in [200, 400, 800, 1600, 3200]:
        summary[ctx] = {}
        print(f"Context size: {ctx}")
        summary[ctx]["learned"] = summarize_results("learned", ctx)
        summary[ctx]["covariance"] = summarize_results("covariance", ctx)
        summary[ctx]["identity"] = summarize_results("identity", ctx)
        summary[ctx]["ifbo"] = summarize_results("ifbo", ctx)
    print(make_table(summary))
    make_perf_context_size_plot(summary)
