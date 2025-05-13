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

    benchmark_rows = []

    for benchmark in benchmarks:
        bench_dir = Path(results_root) / name / benchmark
        if not bench_dir.exists():
            print(f"[x] {bench_dir} not found - skipping.")
            continue

        for subbench_dir in bench_dir.iterdir():
            if not subbench_dir.is_dir():
                continue
            csv_path = subbench_dir / f"ctx_{context_points}.csv"
            if not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
            except:
                print(f"skiiping {csv_path}")
                continue
            lls = []
            for _, row in df.iterrows():
                raw = row["raw"]
                # ensure we have a dict
                if isinstance(raw, str):
                    try:
                        raw = ast.literal_eval(raw)
                    except Exception:
                        raw = None
                if not isinstance(raw, dict):
                    continue

                assert "probs" in raw.keys()
                assert "borders" in raw.keys()

                probs = np.array(raw["probs"]).squeeze()
                borders = np.array(raw["borders"]).squeeze()
                tgt = row["target"]
                idx = np.searchsorted(borders, tgt, side="right") - 1
                if 0 <= idx < len(probs):
                    p = float(probs[idx])
                    ll = np.log(max(p, 1e-12)) - np.log(borders[idx + 1] - borders[idx])
                    lls.append(ll)

            mean_ll = np.mean(lls) if lls else np.nan
            med_ll = np.median(lls) if lls else np.nan
            std_ll = np.std(lls) / np.sqrt(len(lls) - 1) if lls else np.nan

            benchmark_rows.append(
                {
                    "benchmark": benchmark,
                    "dataset": subbench_dir.name,
                    "allocations": len(df),
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
        except:
            breakpoint()

    if not benchmark_rows:
        print("No CSV files found - nothing to summarise.")
        return


if __name__ == "__main__":
    for ctx in [200, 400, 800, 1600]:
        print(f"Context size: {ctx}")
        summarize_results("learned", ctx)
        summarize_results("covariance", ctx)
        summarize_results("identity", ctx)
        summarize_results("ifbo", ctx)
