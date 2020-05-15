# Run from root directory, not util_scripts
import sys
sys.path.append('.')

import argparse
import numpy as np

import utils

def format_time(seconds):
    if seconds < 60:
        return "{:.2}s".format(seconds)
    else:
        whole_min = int(seconds // 60)
        whole_sec = int(round(seconds % 60))
        return "{}m{}s".format(whole_min, whole_sec)

# Util script for computing runtimes. Works on one wandb "sweep" (set of training runs
# with different hyperparameters).
def main():
    parser = argparse.ArgumentParser(description='Runtime calculator')
    parser.add_argument("--id", type=str, help="sweep id")
    args = parser.parse_args()

    assert args.id, "No sweep id given"

    runs = utils.get_sweep_runs(args.id)


    # best runs for each model and dataset
    model_runs = {}

    # Runtimes for each kind of model
    model_runtimes = {}

    for run in runs:
        ds = run.config["dataset"]
        model = run.config["model"]
        ll = run.summary["log_likelihood"]

        if not (model in model_runtimes):
            model_runtimes[model] = []

        model_runtimes[model].append(run.summary["_runtime"])

        if not (model in model_runs):
            model_runs[model] = {}

        # Check for NaN or -inf
        if not (type(ll) == str):
            if not (ds in model_runs[model]) or\
                ll > model_runs[model][ds].summary["log_likelihood"]:

                model_runs[model][ds] = run

    # Compute statistics
    n_epochs = runs[0].config["epochs"]
    mean_rt = {model: np.mean(times) for (model, times) in model_runtimes.items()}
    print("Mean training time statistics over {} epochs:".format(n_epochs))
    for model, time in mean_rt.items():
        print("{} | total: {}\ttime/epoch: {}".format(
            model, format_time(time), format_time(time/n_epochs)))
    print()

    print("Time to early stopping:")
    # Time to early stopping
    for model, datasets in model_runs.items():
        time_to_best = [(run.summary["best_epoch"]/n_epochs)*run.summary["_runtime"]
                for _, run in datasets.items()]
        mean_ttb = np.mean(time_to_best)
        std_ttb = np.std(time_to_best)

        best_epochs = [run.summary["best_epoch"] for _, run in datasets.items()]
        mean_be = np.mean(best_epochs)
        std_be = np.std(best_epochs)

        print("{} | {}±{} to mean epoch {:.5}±{:.5}".format(model, format_time(mean_ttb),
            format_time(std_ttb), mean_be, std_be))


if __name__ == "__main__":
    main()

