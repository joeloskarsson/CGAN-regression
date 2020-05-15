WANDB_PROJECT = "cgan_regression"

DATASET_PATH = "datasets"
CGAN_SPEC_DIR = "cgan_specs"
NN_SPEC_DIR = "nn_specs"
BEST_PARAMS_FILE = "epoch_best.pt"

LOAD_WORKERS = 4

# Longer names for evaluation metrics
EVAL_NAME_MAP = {
    "ll": "log_likelihood",
    "mae": "mae",
    "ds_specific": "ds_specific",
}

MODEL_NICE_NAMES = {
    "cgan": "CGAN",
    "gp": "GP",
    "nn_reg": "NN Reg.",
    "nn_het": "Het. NN. Reg.",
    "mdn": "MDN",
    "cgmmn": "CGMMN",
    "gmmn": "GMMN",
    "dctd": "DCTD",
}

PLOT_POINTS = 100 # Amount of points to use for x-axis when plotting curves

# Named matplotlib colors to use for plotting
COLORS = [
"hotpink",
"darkgreen",
"orange",
"dodgerblue",
]

LINE_STYLES = [
"-",
"--",
"-.",
":",
]

SCATTER_MARKERS = [
".",
"v",
"P",
"x",
"*",
"s",
"D",
"h",
"2",
"^",
]*10 # Repeat to 100 after first 10

TEST_SEEDS = [
4541,
7202,
3628,
6442,
987,
6795,
8360,
2824,
8978,
1408,
8064,
5373,
329,
772,
7657,
]
