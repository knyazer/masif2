import os
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from tabulate import tabulate


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

    benchmark_rows = []

    for benchmark in benchmarks:
        bench_dir = Path(results_root) / name / benchmark
        if not bench_dir.exists():
            print(f"[x] {bench_dir} not found - skipping.")
            continue

        for subbench_dir in bench_dir.iterdir():
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

            benchmark_rows.append(
                {
                    "benchmark": benchmark,
                    "dataset": subbench_dir.name,
                    "allocations": len(all_probs),
                    "mean_ll": round(mean_ll, precision) if not np.isnan(mean_ll) else "-",
                    "median_ll": round(med_ll, precision) if not np.isnan(med_ll) else "-",
                    "std_ll": std_ll if not np.isnan(std_ll) else "-",
                }
            )

        if benchmark_rows == []:
            continue
        bdf = pd.DataFrame(benchmark_rows)
        try:
            meanmed = bdf["median_ll"].mean()
            mean = bdf["mean_ll"].mean()
            epi_std = bdf["mean_ll"].std() / np.sqrt(len(bdf))
            datasetwise_std = bdf["std_ll"].mean()
            print(f"{name}/{benchmark}: {mean:.2f}/{meanmed:.2f}+-{datasetwise_std:.2f}")
        except Exception as e:
            breakpoint()
            pass

    if not benchmark_rows:
        print("Something went wrong :(")
        return


if __name__ == "__main__":
    for ctx in [200, 400, 800, 1600]:
        print(f"Context size: {ctx}")
        summarize_results("learned", ctx)
        summarize_results("covariance", ctx)
        summarize_results("identity", ctx)
        summarize_results("ifbo", ctx)
