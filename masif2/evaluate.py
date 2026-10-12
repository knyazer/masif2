import os
import traceback as tb
from pathlib import Path

import pandas as pd
from jax import random as jr
from tqdm import tqdm

from .main import Config, get_eval_fn

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


def evaluate_all_models(exp_name="exp01", eval_contexts=None):
    if eval_contexts is None:
        eval_contexts = [100, 200, 400, 800, 1600]

    # Fixed random key for consistent evaluation
    eval_key = jr.PRNGKey(42)

    # Find all folders with .version files
    version_files = Path().rglob(f"models/{exp_name}/**/.version")

    results = []

    for version_file in tqdm(version_files, "Evaluating all the models..."):
        folder_path = Path(version_file).parent
        tqdm.write(f"Evaluating model in: {folder_path}")
        print(folder_path)

        try:
            # Load model and config using the same method as in main.py
            config = Config.load(folder_path)

            # Check if results already exist
            cached_results = config.load_results()
            if cached_results is not None:
                tqdm.write("Cache hit, thus skipping...")
                results.append(cached_results)
                continue
            model = config.load_model()

            eval_results = []
            for ctx in eval_contexts:
                # Create evaluation function with the same random key
                eval_fn = get_eval_fn(
                    eval_key,
                    config,
                    kind="test",
                    ctx_variants=[ctx],
                    has_aux=True,
                    subset_size=3200,
                )

                # Evaluate the model
                _test_loss, aux_results = eval_fn(model)
                tqdm.write(f"For {ctx} the test loss is: {_test_loss:.4f}")

                # Save results using config
                for ds in aux_results:
                    eval_results.append(
                        {
                            "ctx_size": ctx,
                            "dataset": ds,
                            "ll": aux_results[ds]["ll"],
                            "mmedll": aux_results[ds]["mmedll"],
                        }
                    )
            eval_results = pd.DataFrame(eval_results)
            config.save_results(eval_results)
            results.append(eval_results)

        except Exception as e:
            print(f"  Error evaluating model in {folder_path}: {tb.format_exception(e)}")
            continue

    return results


if __name__ == "__main__":
    exp_name = "exp01"
    results = evaluate_all_models(exp_name)
