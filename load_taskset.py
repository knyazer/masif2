import json
import os
import urllib
from concurrent import futures

import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf  # for gfile.
import tqdm

gfile = tf.io.gfile


def load_joint_cache(task, opt_set_name):
    """Loads the learning curves for the given task and opt_set_name."""
    base_dir = "gs://gresearch/task_set_data/"
    p = os.path.join(base_dir, task, "%s_10000_replica5.npz" % (opt_set_name))
    cc = np.load(gfile.GFile(p, "rb"))
    return cc["optimizers"], cc["xs"], cc["ys"]


def threaded_tqdm_map(threads, func, data):
    """Helper that does a map on multiple threads."""
    future_list = []
    with futures.ThreadPoolExecutor(threads) as executor:
        for l in tqdm.tqdm(data, position=0):
            future_list.append(executor.submit(func, l))
        return [x.result() for x in tqdm.tqdm(future_list, position=0)]


def load_tasks(tasks):
    """Multi threaded loading of all data for each task.
    Args:
      tasks: list of task names
    Returns:
      A dictionary mapping taks name to tuples of:
      (optimizer names, x data points, and y data points)
    """

    def load_one(t):
        adam8p = load_joint_cache(t, "adam8p_wide_grid_1k")
        adam4p = load_joint_cache(t, "adam4p_wide_grid_1k")
        return {
            "adam8p": adam8p,
            "adam4p": adam4p,
        }

    results = threaded_tqdm_map(100, load_one, tasks)

    for k, v in zip(tasks, results):
        if v is None:
            print("No data found for task: %s" % k)

    return {k: v for k, v in zip(tasks, results) if v is not None}


def get_task_names():
    content = gfile.GFile("gs://gresearch/task_set_data/task_names.txt").read()
    return sorted(content.strip().split("\n"))


tasks = get_task_names()

print("Tasks total: %s" % len(tasks))


tasks = list(filter(lambda x: "FixedTextRNN" in x, tasks))

results = load_tasks(tasks)

opt_names, x, y = results[list(results.keys())[0]]["adam4p"]
y[1, 0, :, 2]

path8 = "https://raw.githubusercontent.com/google-research/google-research/master/task_set/optimizers/configs/adam8p_wide_grid.json"
path4 = "https://raw.githubusercontent.com/google-research/google-research/master/task_set/optimizers/configs/adam4p_wide_grid.json"
adam_hp = {
    "adam8p": json.loads(urllib.request.urlopen(path8).read()),
    "adam4p": json.loads(urllib.request.urlopen(path4).read()),
}

for task in tasks:
    result = results[task]
    for opt in ["adam4p", "adam8p"]:
        df_list = []
        opt_names, x, y = result[opt]
        y = np.nan_to_num(y, 1e8)
        y = y - y.min()
        y = 1.0 - np.clip(y / np.median(y[:, :, :, 0]), 0, 1)
        hparams = [adam_hp[opt][optname.decode("utf8")][0] for optname in opt_names]

        for i in range(len(y)):
            for _seed in range(5):
                df_list.append({"data": y[i, _seed, :, 2], **hparams[i]})
        df = pd.DataFrame(df_list)
        df.to_csv(f"taskset/{opt}_{task}.csv", compression="gzip")
        print(f"Task {task}-{opt} with {df.head()} is done")
