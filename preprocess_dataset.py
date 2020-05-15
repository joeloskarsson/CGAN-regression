import argparse
import numpy as np
import os

import constants
import dataset_list

def save_splits(dataset_name, splits):
    # Create directory
    dataset_dir = os.path.join(constants.DATASET_PATH, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for name, data in splits.items():
        split_path = os.path.join(dataset_dir, "{}.csv".format(name))
        np.savetxt(split_path, data, delimiter=",")


def generate(name, path=None):
    # Set seed!
    np.random.seed(42)

    dataset = dataset_list.get_dataset_spec(name)()

    if dataset.synthetic:
        # Generate and save splits
        splits = {split_name: dataset.sample(dataset.n_samples[split_name]) for
                split_name in ["train", "val", "test"]}
        save_splits(name, splits)

    else:
        if path or (not dataset.requires_path):
            # Preprocess real world dataset
            all_data = dataset.preprocess(path)
            n = all_data.shape[0]

            # Shuffle data
            np.random.shuffle(all_data)

            # Create splits
            test_index = int(n*dataset.test_percent)
            val_index = test_index + int(n*dataset.val_percent)
            splits = {}
            splits["test"], splits["val"], splits["train"] = np.split(all_data,
                    [test_index, val_index], axis=0)

            save_splits(name, splits)

        else:
            print("No dataset path given for dataset {}, skipping preprocessing.".format(
                name))

# Pre-process a specific dataset, or all possible of "all" is given as dataset name.
#
# For synthetic dataset pre-processing simply means generating the data.
# For most real datasets an additional file-path is required, to where to read the data # from. Pre-processing then includes things like standardization of the loaded data.
#
# The datasets are split into train,val and test and saved in the datasets directory.
def main():
    parser = argparse.ArgumentParser(description='Pre-process or generate datasets')
    parser.add_argument("--dataset", type=str, help="Which dataset to pre-process")
    parser.add_argument("--file", type=str,
        help="Path to file to use for dataset creation (only for some (real) datasets)")
    args = parser.parse_args()

    assert args.dataset, "Must specify dataset"

    if args.dataset == "all":
        # Generate all synthetic datasets
        for ds_name in dataset_list.sets:
            generate(ds_name)
    else:
        assert args.dataset in dataset_list.sets, (
                "Unknown dataset: {}".format(args.dataset))
        generate(args.dataset, args.file)


if __name__ == "__main__":
    main()
