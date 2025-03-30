#!/usr/bin/env python3
import json
import pandas as pd
import argparse


def flatten_dict(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary.
    For example, {"lr_hparams": {"initial_value": 0.001}} becomes {"lr_hparams.initial_value": 0.001}.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def parse_jsonl(file_path):
    """
    Parses a JSONL file and returns a dictionary mapping dataset names to lists of records.
    Each record is a flat dictionary of hyperparameters and a "data" field containing the test error rates.
    """
    dataset_groups = {}
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            # Determine which dataset this record belongs to.
            dataset = record.get("dataset", "unknown")

            # Get the hyperparameters from the nested "hparams" dictionary and flatten it.
            hparams = record.get("hparams", {})
            all_flat_params = flatten_dict(hparams)

            flat_hparams = {}

            for key in all_flat_params.keys():
                if (
                    key == "lr_decay_factor"
                    or key == "lr_hparams.initial_value"
                    or key == "lr_hparams.power"
                    or key == "opt_hparams.momentum"
                ):
                    flat_hparams[key] = all_flat_params[key]
            # You could also merge in other keys starting with "hps." if desired:
            # for key, value in record.items():
            #     if key.startswith("hps.") and key not in flat_hparams:
            #         flat_hparams[key] = value

            # Add a column "data" containing the test error rate (a list in this case).
            flat_hparams["data"] = record.get("test/error_rate")
            if flat_hparams["data"] is None:
                flat_hparams["data"] = record.get("valid/error_rate")

            # Group records by dataset.
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(flat_hparams)

    # Convert each list of records to a DataFrame.
    dataframes = []
    for ds, records in dataset_groups.items():
        df = pd.DataFrame(records)
        # Optionally, add the dataset as a column in the DataFrame.
        df["dataset"] = ds
        dataframes.append(df)
    return dataframes


def main():
    dfs = parse_jsonl("pd1_raw/pd1_unmatched_phase1_results.jsonl")

    # For demonstration, print out a summary for each dataset.
    for df in dfs:
        df = df.dropna(subset="data")
        if len(df) != 0:
            dataset = df["dataset"].iloc[0]
            print(f"Dataset done: {dataset}")
            df.to_csv(f"pd1/{dataset}.csv", compression="gzip")
            print(df.head())


if __name__ == "__main__":
    main()
