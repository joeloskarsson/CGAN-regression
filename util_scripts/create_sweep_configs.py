# Run from root directory, not util_scripts
import sys
sys.path.append('.')

import wandb
import math
import argparse
import json
import os

import utils
import constants

# Util script for creating multiple test config files from a wandb sweep.
# A grouping is specified, meaning only the best performing model in each group will be
# used for testing. Performance is measured as log-likelihood on the validation data.
#
# For example, consider a sweep containing two datasets, two models and different sets of
# hyperparameters. If grouping is done on "dataset,model", 4 test configs will be
# created. Each model will have a test config for each dataset. The best hyperparameters
# will be chosen because of the grouping.
def main():
    parser = argparse.ArgumentParser(description='Sweep test config generator')
    parser.add_argument("--id", type=str, help="id of wandb sweep")
    parser.add_argument("--base_config", type=str,
            help="Base config file for test setup")
    parser.add_argument("--fields", type=str, default="",
            help="Fields to carry over from training config to test config")
    parser.add_argument("--grouping", type=str, default="dataset,cgan_type",
        help="Parameters to group by, only the best (in validation) model is tested")
    parser.add_argument("--out_dir", type=str, help="Config file output directory")
    args = parser.parse_args()

    assert args.id, "Must specify id"
    assert args.out_dir, "Must specify output dir"
    assert args.base_config, "Must specify base test config"

    base_config = utils.json_to_dict(args.base_config)
    if args.fields:
        extra_fields = args.fields.split(",")
    else:
        extra_fields = []

    groupings = args.grouping.split(",")

    runs = utils.get_sweep_runs(args.id)

    # Groups are mapped using a string: grouping1_grouping2_grouping3 etc.
    best_in_groups = {} # Dict of dataset-group pairs to (ll, run)

    for run in runs:
        ds = run.config["dataset"]
        group_values = [str(run.config[g]) for g in groupings]
        key = "_".join(group_values)

        # Exclude crashed runs
        if "log_likelihood" in run.summary:
            ll = run.summary["log_likelihood"]

            # Check for NaN or -inf
            if not (type(ll) == str):
                if (not (key in best_in_groups)) or (ll > best_in_groups[key][0]):
                    best_in_groups[key] = (ll, run)

    for grouping, (ll, run) in best_in_groups.items():
        # Create configs
        config = base_config

        # Always carry over dataset and model
        config["dataset"] = run.config["dataset"]
        config["model"] = run.config["model"]

        file_name = grouping.replace("/","_") # Need to clean options with a /

        config["restore"] = run.id
        config["plot_prefix"] = "{}_".format(file_name)

        for field in extra_fields + groupings:
            config[field] = run.config[field]

        config_path = os.path.join(args.out_dir, "{}.json".format(file_name))
        with open(config_path, 'w') as fp:
            json.dump(config, fp, indent=0)

        print("Created config for {}".format(grouping))

    print("done")


if __name__ == "__main__":
    main()

