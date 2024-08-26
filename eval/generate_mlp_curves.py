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

        n_instances = int(dataset._qualities["NumberOfInstances"])
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
    for x, y in regression_problems:
        x = np.array(x)
        y = np.array(y)
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
def mse_loss(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred.ravel() - y.ravel()) ** 2)


@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    return mse_loss(model, x, y)


def train_step(model, opt_state, lr, optimizer, x, y):
    loss, grads = loss_fn(model, x, y)
    grads = jax.tree.map(
        lambda x: jnp.nan_to_num(x) * lr, grads, is_leaf=eqx.is_inexact_array
    )  # just in case, anti-nan
    update, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, update)
    return model, opt_state, loss


def train_mlp(make_mlp, learning_rates, num_epochs, partitions, batch_size, key):
    train_losses = []
    test_losses = []
    log_interval = max(1, num_epochs // 5)

    @eqx.filter_jit
    def scan_fn(carry, _):
        models, opt_states, lr, i, key = carry
        model = jax.tree.map(lambda m: m[i], models, is_leaf=eqx.is_inexact_array)
        opt_state = jax.tree.map(
            lambda s: s[i], opt_states, is_leaf=eqx.is_inexact_array
        )
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
        test_loss = mse_loss(model, x_test, y_test)
        new_models = jax.tree.map(
            lambda leaf, new_leaf: leaf.at[i].set(new_leaf),
            models,
            model,
            is_leaf=eqx.is_inexact_array,
        )
        new_opt_states = jax.tree.map(
            lambda leaf, new_leaf: leaf.at[i].set(new_leaf),
            opt_states,
            opt_state,
            is_leaf=eqx.is_inexact_array,
        )
        return (new_models, new_opt_states, lr, i, key), (train_losses[-1], test_loss)

    with tqdm(
        total=num_epochs,
    ) as pbar:
        key, subkey = jr.split(key)
        opt_states = []
        models = []
        lrs = []
        opt_index = jr.choice(subkey, len(optimizers))
        optimizer = optimizers[opt_index](learning_rate=1e-4)
        for rate in learning_rates:
            model = make_mlp(key)
            key, subkey = jr.split(subkey)
            opt_states.append(optimizer.init(eqx.filter(model, eqx.is_inexact_array)))
            models.append(model)
            lrs.append(rate / 1e-4)
        lrs = jnp.array(lrs)
        opt_states = jax.tree.map(
            lambda *args: jnp.stack(args),
            *opt_states,
            is_leaf=eqx.is_inexact_array,
        )
        models = jax.tree.map(
            lambda *args: jnp.stack(args),
            *models,
            is_leaf=eqx.is_inexact_array,
        )

        for epoch_start in range(0, num_epochs, log_interval):
            epoch_end = min(epoch_start + log_interval, num_epochs)
            epoch_range = jnp.arange(epoch_start, epoch_end)
            key, subkey = jr.split(key)

            n_splits = len(learning_rates)
            (models, opt_states, *_), losses = eqx.filter_vmap(
                lambda i, key: jax.lax.scan(
                    scan_fn,
                    (models, opt_states, lrs[i], i, key),
                    epoch_range,
                )
            )(
                jnp.arange(n_splits),
                jr.split(key, n_splits),
            )
            # select last one from each stacked carry as the model and opt state
            # first dim is vmap, second dim is stacking, i guess? idfk but it works
            models = jax.tree.map(
                lambda leaf: leaf[:, -1, ...], models, is_leaf=eqx.is_inexact_array
            )
            opt_states = jax.tree.map(
                lambda leaf: leaf[:, -1, ...],
                opt_states,
                is_leaf=eqx.is_inexact_array,
            )

            epoch_train_losses, epoch_test_losses = losses
            train_losses.append(epoch_train_losses)
            test_losses.append(epoch_test_losses)

            pbar.update(len(epoch_range))
            pbar.set_description_str(
                f"tloss1={epoch_train_losses[0][-1]:.4f}|tloss2={epoch_train_losses[1][-1]:.4f}"
            )
    del models, opt_states
    return None, train_losses, test_losses


def create_train_test_partitions(x, y, num_partitions, key):
    num_samples = x.shape[0]
    indices = jnp.arange(num_samples)
    train_x, train_y, test_x, test_y = [], [], [], []

    for _ in range(num_partitions):
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, indices)
        split = int(0.8 * num_samples + 1)
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
    min_batch_size, max_batch_size = 32, 512
    num_epochs = 500
    num_models = 1_000
    different_lr_partitions = 20
    num_datasets = 4
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

        print(
            f"Model {j+1}/{num_models}: layers={num_layers}, sizes={hidden_sizes}, lrs={learning_rates}, batch_size={batch_size}",
        )

        # Train the model on all datasets sequentially
        final_losses = []
        original_batch_size = batch_size
        train_losses, test_losses = [], []
        # shuffle and choose a subset of the datasets
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, len(normalized_problems))
        selected_datasets = [normalized_problems[i] for i in perm[:num_datasets]]
        for i, dataset in enumerate(selected_datasets):
            x, y = dataset

            key, subkey = jr.split(key)
            partitions = create_train_test_partitions(
                x, y, different_lr_partitions, subkey
            )
            batch_size = min(original_batch_size, partitions["train_x"].shape[1])
            print(
                f"Training on dataset {perm[i]}: num_features={x.shape[1]}, num_samples={x.shape[0]}, batch_size={batch_size}",
            )
            num_features = x.shape[1]

            subkey, model_key = jr.split(subkey)
            _, train_losses, test_losses = train_mlp(  # never jit!
                lambda key: MLP(num_features, hidden_sizes, key),
                learning_rates,
                num_epochs,
                partitions,
                batch_size,
                subkey,
            )
        for buf in jax.live_arrays():
            if buf is not key:
                buf.delete()
