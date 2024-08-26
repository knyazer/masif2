#!/usr/bin/env python

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import openml
import optax
import pandas as pd
from openml.tasks.task import TaskType
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

openml.config.set_root_cache_directory(Path.expanduser(Path(".openml_cache")))

jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def download_regression_problems():
    # Filter tasks to exclude datasets with missing values
    regression_tasks = openml.tasks.list_tasks(
        task_type=TaskType.SUPERVISED_REGRESSION,
        output_format="dataframe",
        tag="OpenML-Reg19",  # This tag includes datasets without missing values
    )
    assert (
        "tid" in regression_tasks
    ), "No regression tasks found. You might have been rate-limited by OpenML."
    regression_datasets = []
    processed_dataset_ids = set()
    for task_id in (
        pbar := tqdm(
            regression_tasks["tid"],
            desc="Processing regression problems",
        )
    ):
        pbar.set_description(
            f"Done with {len(regression_datasets)} datasets. Processing task {task_id}",
        )

        # Get the task and dataset
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        n_instances = int(dataset._qualities["NumberOfInstances"])  # type: ignore # noqa
        if n_instances > 100_000:
            continue

        # Check if the dataset has already been processed
        if dataset.dataset_id in processed_dataset_ids:
            continue

        # Get the data from the dataset
        x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        try:
            # Convert categorical features to numeric
            x = pd.get_dummies(x, drop_first=True)

            # Convert target variable to numeric if it is categorical
            if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype):  # type: ignore
                y = pd.Categorical(y).codes

            # Normalize the inputs
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

            regression_datasets.append((x, y))

            # Add the dataset ID to the processed set
            processed_dataset_ids.add(dataset.dataset_id)
        except Exception as e:
            print(f"Failed to process dataset {dataset.dataset_id}: {e}")  # noqa

    return regression_datasets


def normalize_regression_problems(regression_problems):
    normalized_problems = []
    for _x, _y in regression_problems:
        x = np.array(_x)
        y = np.array(_y)
        # Normalize inputs (x)
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_normalized = (x - x_mean) / (
            x_std + 1e-8
        )  # Add small constant to avoid division by zero

        # Normalize outputs (y)
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / (y_std + 1e-8)

        normalized_problems.append((x_normalized, y_normalized))
    return normalized_problems


class MLP(eqx.Module):
    layers: list

    def __init__(self, in_size, hidden_sizes, key):
        out_size = 1
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size, *hidden_sizes, out_size]
        self.layers = [
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        ]

    @eqx.filter_jit
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x), approximate=True)
        return self.layers[-1](x)


@eqx.filter_jit
def predvmap(model, x):
    return jax.vmap(model)(x)


@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    pred = predvmap(model, x)
    return jnp.mean((pred.ravel() - y.ravel()) ** 2)


@eqx.filter_jit
def eval_loss_fn(model, x, y, eps, c1, c2):
    # accuracy: number of predictions that are within eps of the true value
    # plus slowly decreasing value if it is far away, e.g. if it is 1 away from the
    # true value 'accuracy' is 0.1
    pred = predvmap(model, x)
    accuracy = jnp.mean(jnp.abs(pred.ravel() - y.ravel()) < eps) * c1
    penalty = jnp.mean(jnp.maximum(0, 1.0 - jnp.abs(pred.ravel() - y.ravel())) ** 2)
    return jnp.minimum(accuracy + jnp.minimum(penalty, c2), 1)


@eqx.filter_jit
def train_step(model, opt_state, lr, optimizer, x, y):
    loss, grads = loss_fn(model, x, y)
    grads = jax.tree.map(
        lambda x: jnp.nan_to_num(x) * lr, grads, is_leaf=eqx.is_inexact_array
    )  # just in case, anti-nan
    update, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, update)
    return model, opt_state, loss


@eqx.filter_jit
def scan_fn(carry, _, *, batch_size, partitions, eps, c1, c2):
    model, opt_state, lr, i, key = carry
    x_train, y_train = partitions["train_x"][i], partitions["train_y"][i]

    # Shuffle and batch the training data
    num_samples = x_train.shape[0]
    key, subkey = jr.split(key)
    perm = jr.permutation(subkey, jnp.arange(num_samples))
    x_train, y_train = x_train[perm], y_train[perm]

    @eqx.filter_jit
    def batch_scan_fn(carry, start):
        model, opt_state = carry
        x_batch = jax.lax.dynamic_slice_in_dim(x_train, start, batch_size, axis=0)
        y_batch = jax.lax.dynamic_slice_in_dim(y_train, start, batch_size, axis=0)
        model, opt_state, train_loss = train_step(
            model,
            opt_state,
            lr,
            optimizer,
            x_batch,
            y_batch,
        )
        return (model, opt_state), train_loss

    (model, opt_state), train_losses = jax.lax.scan(
        batch_scan_fn,
        (model, opt_state),
        jnp.arange(0, num_samples, batch_size),
    )

    x_test, y_test = partitions["test_x"][i], partitions["test_y"][i]
    test_loss = eval_loss_fn(model, x_test, y_test, eps[i], c1[i], c2[i])
    return (model, opt_state, lr, i, key), (train_losses[-1], test_loss)


@eqx.filter_jit
def single_epoch(i, key, d_model, opt_state, lr, epoch_range, s_model, eps, c1, c2):
    # key = jr.split(jr.wrap_key_data(jr.key_data(key) + i.astype(jnp.uint32) + 1))[ ##
    #     0
    # ]  # fuck me

    model = eqx.combine(d_model, s_model)
    carry, losses = jax.lax.scan(
        f=lambda *args: eqx.Partial(
            scan_fn, partitions=partitions, batch_size=batch_size, eps=eps, c1=c1, c2=c2
        )(*args),
        init=(
            model,
            opt_state,
            lr,
            i,
            key,
        ),
        xs=epoch_range,
        length=epoch_range.shape[0],
    )
    return carry, losses


def train_mlp(
    make_mlp,
    learning_rates,
    optimizer,
    num_epochs,
    partitions,
    batch_size,
    key,
    epses,
    c1s,
    c2s,
):
    train_losses = []
    test_losses = []
    log_interval = max(1, num_epochs // 10)
    num_sequential = 8  # tradeoff for compilation speed
    n_splits = len(learning_rates) // num_sequential

    key, subkey = jr.split(key)
    opt_states = []
    models = []
    lrs = []
    for rate in learning_rates:
        model = make_mlp(key)
        key, subkey = jr.split(subkey)
        opt_states.append(optimizer.init(eqx.filter(model, eqx.is_inexact_array)))
        models.append(model)
        lrs.append(rate / 1e-4)
    all_lrs = jnp.array(lrs)
    all_opt_states = jax.tree.map(
        lambda *args: jnp.stack(args),
        *opt_states,
        is_leaf=eqx.is_inexact_array,
    )
    all_models = jax.tree.map(
        lambda *args: jnp.stack(args),
        *models,
        is_leaf=eqx.is_inexact_array,
    )
    del opt_states, models, lrs

    for seqi in range(num_sequential):
        starti = seqi * n_splits
        endi = starti + n_splits
        with tqdm(
            total=num_epochs,
        ) as pbar:
            models = jax.tree.map(
                lambda x: x[starti:endi], all_models, is_leaf=eqx.is_inexact_array
            )
            opt_states = jax.tree.map(
                lambda x: x[starti:endi], all_opt_states, is_leaf=eqx.is_array
            )
            lrs = all_lrs[starti:endi]
            eps = epses[starti:endi]
            c1 = c1s[starti:endi]
            c2 = c2s[starti:endi]
            for epoch_start in range(0, num_epochs, log_interval):
                epoch_end = min(epoch_start + log_interval, num_epochs)
                epoch_range = jnp.arange(epoch_start, epoch_end)
                key, subkey = jr.split(key)

                d_models, s_models = eqx.partition(models, eqx.is_inexact_array)
                (models, opt_states, *_), losses = eqx.filter_vmap(
                    eqx.Partial(
                        single_epoch,
                        epoch_range=epoch_range,
                        s_model=s_models,
                        eps=eps,
                        c1=c1,
                        c2=c2,
                    ),
                )(
                    jnp.arange(n_splits),
                    jr.split(key, n_splits),
                    d_models,
                    opt_states,
                    lrs,
                )
                # select last one from each stacked carry as the model and opt state
                # first dim is vmap, second dim is stacking, i guess? idfk but it works

                epoch_train_losses, epoch_test_losses = losses
                train_losses.append(epoch_train_losses)
                test_losses.append(epoch_test_losses)

                pbar.update(len(epoch_range))
                pbar.set_description_str(
                    f"trloss1={epoch_train_losses[0][-1]:.4f} | "
                    f"trloss2={epoch_train_losses[1][-1]:.4f} | "
                    f"evloss1={epoch_test_losses[0][-1]:.4f} | "
                    f"evloss2={epoch_test_losses[1][-1]:.4f}"
                )
    return None, train_losses, test_losses


def create_train_test_partitions(x, y, num_partitions, key, test_frac):
    num_samples = x.shape[0]
    indices = jnp.arange(num_samples)
    train_x, train_y, test_x, test_y = [], [], [], []

    for _ in range(num_partitions):
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, indices)
        split = int((1 - test_frac) * num_samples + 1)
        train_idx, test_idx = perm[:split], perm[split:]
        train_x.append(x[train_idx])
        train_y.append(y[train_idx])
        test_x.append(x[test_idx])
        test_y.append(y[test_idx])

    return {
        "train_x": jnp.stack(train_x),
        "train_y": jnp.stack(train_y),
        "test_x": jnp.stack(test_x),
        "test_y": jnp.stack(test_y),
    }


if __name__ == "__main__":
    regression_problems = download_regression_problems()
    normalized_problems_cpu = normalize_regression_problems(regression_problems)

    # Hyperparameter ranges
    max_layers = 6
    max_width = 128
    min_width = 16
    min_lr, max_lr = 1e-5, 5e-1
    min_batch_size, max_batch_size = 32, 1024
    min_test_frac, max_test_frac = 0.3, 0.7
    num_epochs = 500
    num_models = 1_000_000
    different_lr_partitions = 80
    num_datasets = 10
    optimizers = [optax.adam, optax.sgd, optax.rmsprop]

    for j in range(num_models):
        normalized_problems = jax.tree.map(jnp.array, normalized_problems_cpu)
        key = jr.key(j)
        key, subkey = jr.split(key)

        while True:
            subkey, layer_key = jr.split(subkey)
            num_layers = int(jr.poisson(layer_key, lam=1.5) + 1)
            if num_layers <= max_layers:
                break

        subkey, size_key = jr.split(subkey)
        hidden_sizes = (
            jnp.round(
                jnp.exp(
                    jr.uniform(
                        size_key,
                        (num_layers,),
                        minval=jnp.log(min_width),
                        maxval=jnp.log(max_width),
                    )
                )
            )
            .astype(int)
            .tolist()
        )

        subkey, lr_key = jr.split(subkey)
        learning_rates = jnp.exp(
            jr.uniform(
                lr_key,
                (different_lr_partitions,),
                minval=jnp.log(min_lr),
                maxval=jnp.log(max_lr),
            )
        )

        subkey, batch_key = jr.split(subkey)
        batch_size = int(
            jnp.exp(
                jr.uniform(
                    batch_key,
                    minval=jnp.log(min_batch_size),
                    maxval=jnp.log(max_batch_size),
                )
            )
        )

        subkey, test_frac_key = jr.split(subkey)
        test_frac = jr.uniform(
            test_frac_key, minval=min_test_frac, maxval=max_test_frac
        )
        subkey, eps_key = jr.split(subkey)
        epses = jnp.exp(
            jr.uniform(
                eps_key,
                (different_lr_partitions,),
                minval=jnp.log(2e-2),
                maxval=jnp.log(2e-1),
            )
        )

        subkey, c1_key = jr.split(subkey)
        c1s = jr.uniform(
            c1_key,
            (different_lr_partitions,),
            minval=0.6,
            maxval=1.0,
        )

        subkey, c2_key = jr.split(subkey)
        c2s = jr.uniform(
            c2_key,
            (different_lr_partitions,),
            minval=0.6,
            maxval=1.0,
        )

        print(  # noqa
            f"Model {j+1}/{num_models}:\n"
            f"  layers={num_layers},\n"
            f"  sizes={hidden_sizes},\n"
            f"  lrs={learning_rates},\n"
            f"  batch_size={batch_size},\n"
            f"  test_frac={test_frac:.2f},\n"
            f"  epses={epses},\n"
            f"  c1s={c1s},\n"
            f"  c2s={c2s}",
        )

        # Train the model on all datasets sequentially
        final_losses = []
        original_batch_size = batch_size
        train_losses, test_losses = [], []
        # shuffle and choose a subset of the datasets
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, len(normalized_problems))
        selected_datasets = [normalized_problems[i] for i in perm[:num_datasets]]
        final_losses = []
        opt_index = jr.choice(subkey, len(optimizers))

        hyps = {
            "num_layers": num_layers,
            "hidden_sizes": hidden_sizes,
            "learning_rates": learning_rates,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "optimizer": opt_index,
            "test_frac": test_frac,
            "key": key,
            "epses": epses,
            "c1s": c1s,
            "c2s": c2s,
        }

        Path("data").mkdir(parents=True, exist_ok=True)

        hyperparameters_path = Path(f"data/hyperparameters_model_{j+1}.npz")
        losses_path = Path(f"data/losses_model_{j+1}.npz")
        # Check if the hyperparameters file and the data file both exist
        if hyperparameters_path.exists() and losses_path.exists():
            # Load the existing hyperparameters
            existing_hyps = np.load(hyperparameters_path)

            try:
                # Check if the existing hyperparameters coincide with the current ones
                if (
                    existing_hyps["num_layers"] == num_layers
                    and np.array_equal(existing_hyps["hidden_sizes"], hidden_sizes)
                    and np.array_equal(existing_hyps["learning_rates"], learning_rates)
                    and existing_hyps["batch_size"] == batch_size
                    and existing_hyps["optimizer_type"] == opt_index
                    and np.array_equal(existing_hyps["datasets"], perm[:num_datasets])
                    and existing_hyps["test_frac"] == test_frac
                    and np.array_equal(existing_hyps["epses"], epses)
                    and np.array_equal(existing_hyps["c1s"], c1s)
                    and np.array_equal(existing_hyps["c2s"], c2s)
                ):
                    print(f"Skipping model {j+1}/{num_models} as it already exists.")  # noqa
                    continue
            except KeyError:
                print(f"KeyError encountered for model {j+1}/{num_models}, skipping.")  # noqa
                continue
            except Exception as e:
                print(  # noqa
                    f"Error loading hyperparameters for model {j+1}/{num_models}: {e}"
                )
                continue

        # Store the hyperparameters otherwise
        with hyperparameters_path.open("wb") as f:
            np.savez(
                f,
                num_layers=num_layers,
                hidden_sizes=hidden_sizes,
                learning_rates=learning_rates,
                batch_size=batch_size,
                optimizer_type=opt_index,
                datasets=perm[:num_datasets],
                test_frac=test_frac,
                key=key,
                epses=epses,
                c1s=c1s,
                c2s=c2s,
            )

        if losses_path.exists():
            losses_path.unlink()  # err, traditional unix name lol
        for i, dataset in enumerate(selected_datasets):
            x, y = dataset

            key, subkey = jr.split(key)
            partitions = create_train_test_partitions(
                x, y, different_lr_partitions, subkey, test_frac
            )
            batch_size = min(original_batch_size, partitions["train_x"].shape[1])
            print(  # noqa
                f"Training on dataset {perm[i]}: num_features={x.shape[1]}, "
                f"num_samples={x.shape[0]}, batch_size={batch_size}",
            )
            num_features = x.shape[1]

            subkey, model_key = jr.split(subkey)
            _, train_losses, test_losses = train_mlp(  # never jit!
                lambda key: MLP(num_features, hidden_sizes, key),
                learning_rates,
                optimizer := optimizers[opt_index](learning_rate=1e-4),
                num_epochs,
                partitions,
                batch_size,
                subkey,
                epses,
                c1s,
                c2s,
            )

            final_losses.append((train_losses, test_losses))
            # store the final losses in a file

            with losses_path.open("wb") as f:
                np.savez(f, final_losses)

        # clean all the jax arrays from the memory: resolves any leak issues
        # also gives a hard bound on memory consumption, that I am too lazy to compute
        for buf in jax.live_arrays():
            if buf is not key:
                buf.delete()
