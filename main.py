import os
import argparse
import numpy as np
import torch
import time
import wandb
import ast

import constants
import utils
import models.cgan_versions as cgans
import models.cgmmn as cgmmn
import models.gmmn as gmmn
import models.gp as gp
import models.nn_regressor as nn_reg
import models.nn_heteroskedastic as nn_het
import models.mdn as mdn
import models.dctd as dctd
import dataset_list
import evaluation as ev

# Available models
models = {
    "cgan": cgans.build_cgan,
    "gp": gp.build_gp,
    "nn_reg": nn_reg.NNRegressor,
    "nn_het": nn_het.NNHeteroskedastic,
    "mdn": mdn.MDN,
    "cgmmn": cgmmn.CGMMN,
    "gmmn": gmmn.GMMN,
    "dctd": dctd.DCTD,
}

def get_config():
    parser = argparse.ArgumentParser(description='Train model')
    # If config file should be used
    parser.add_argument("--config", type=str, help="Config file to read run config from")

    # General
    parser.add_argument("--dataset", type=str, help="Which dataset to use")
    parser.add_argument("--model", type=str,
            help="Which type of model to use")
    parser.add_argument("--test", type=int, default=0,
            help="If model should be tested (at the end of possible training)")
    parser.add_argument("--train", type=int, default=1,
            help="If model should be trained")
    parser.add_argument("--name", type=str, help="Name of the run for WandB")
    parser.add_argument("--seed", type=int, default=42,
            help="Seed for random number generator")
    parser.add_argument("--cpu", type=int, default=0,
            help="Force to run on CPU")

    # Evaluation
    parser.add_argument("--test_runs", type=int, default=10,
            help="Testing runs to average score for")
    parser.add_argument("--restore", type=str,
            help="WandB run_id to restore parameters from (requires wandb logging)")
    parser.add_argument("--restore_file", type=str,
            help="Path to file to restore parameters from")
    parser.add_argument("--eval_div", type=str,
            help="Evaluate model by estimating a divergence")
    parser.add_argument("--eval_cgan", type=str,
            help="CGAN (network architecture) to use for evaluation")

    # Plotting
    parser.add_argument("--scatter", type=int, default=0,
            help="If scatter-plots should be created during validation/testing")
    parser.add_argument("--cond_scatter", type=str,
            help="Create scatter plot for conditional distribution at given x:s")
    parser.add_argument("--plot_pdf", type=str,
            help="List of x-values to plot pdf at during validation/testing")
    parser.add_argument("--plot_pdf_index", type=str,
            help="List of test/validation set indexes to plot pdf for")
    parser.add_argument("--plot_functions", type=int, default=0,
            help="Plot some sampled functions by varying x and keeping noise constant")
    parser.add_argument("--plot_gt", type=int, default=0,
            help="Plot ground truth only, instead of model")
    parser.add_argument("--plot_prefix", type=str,
            help="Prefix to be prepended to plot file names")
    parser.add_argument("--cond_plot_trajectories", type=str,
            help="""(For trajectories datasets) Plot 2D trajectory samples.
            If an index is given, plots trajectories for corresponding test sample.
            If a tuple is given, trajectories are conditioned on the tuple as x-value.
            """)
    parser.add_argument("--plot_trajectories", type=int, default=20,
            help="Amount of trajectories to plot.")

    # Batched training models (i.e. neural network based)
    parser.add_argument("--epochs", type=int,
            help="How many epochs to train for", default=10)
    parser.add_argument("--val_interval", type=int, default=10,
            help="Evaluate model every eval_interval:th epoch")
    parser.add_argument("--batch_size", type=int,
            help="Batch size for training", default=128)
    parser.add_argument("--eval_batch_size", type=int,
            help="Batch size to use outside training, in validation etc.",
            default=1000)
    parser.add_argument("--lr", type=float,
            help="Learning rate", default=1e-3)
    parser.add_argument("--lr_decay", type=float,
            help="Multiplicative learning rate decay", default=1.0)
    parser.add_argument("--optimizer", type=str,
            help="Optimizer to use for training", default="rmsprop")

    # KDE
    parser.add_argument("--kernel_scales", type=int, default=50,
            help="Amount of kernel scale parameters in KDE to try for validation")
    parser.add_argument("--kernel_scale_min", type=float, default=0.001,
            help="Lower bound of allowed kernel scale range for KDE")
    parser.add_argument("--kernel_scale_max", type=float, default=0.5,
            help="Upper bound of allowed kernel scale range for KDE")
    parser.add_argument("--eval_samples", type=int, default=200,
            help="How many samples to draw for estimating KDE in evaluation")
    parser.add_argument("--kde_val", type=int, default=0,
            help="Get KDE estimate also in validation.")
    parser.add_argument("--kde_batch_size", type=int, default=10,
            help="How many kernels scales to compute KDE for at the same time")

    # CGAN
    parser.add_argument("--cgan_nets", type=str,
            help="""Name of CGAN network specification, available specs can be
                found in cgan_specs directory.""")
    parser.add_argument("--cgan_type", type=str, default="standard",
            help="""Version of CGAN training objective to use,
                see models/cgan_versions for a list""")
    parser.add_argument("--noise_dim", type=int, default=1,
            help="Dimensionality of noise vector fed to generator")
    parser.add_argument("--noise_dist", type=str, default="gaussian",
            help="Distribution to sample noise vector from")
    parser.add_argument("--gen_lr", type=float,
            help="Generator learning rate")
    parser.add_argument("--disc_lr", type=float,
            help="Discriminator learning rate")
    parser.add_argument("--gen_lr_decay", type=float,
            help="Multiplicative learning rate decay for generator)")
    parser.add_argument("--disc_lr_decay", type=float,
            help="Multiplicative learning rate decay for discriminator)")
    parser.add_argument("--gen_optimizer", type=str,
            help="Optimizer to use for generator training")
    parser.add_argument("--disc_optimizer", type=str,
            help="Optimizer to use for discriminator training")
    parser.add_argument("--clip_grad", type=float, default=0.,
            help="Value to clip gradients at (clipping by norm). 0 is no clipping.")
    parser.add_argument("--gen_samples", type=int, default=1,
            help="How many generator samples to draw for each x in generator training")

    # GMMN (and CGMMN)
    parser.add_argument("--mmd_scales", type=str, default="1,5,10,20",
            help="""Scale parameter to use in MMD-based loss
                (if specific values for x and y are not set)""")
    parser.add_argument("--mmd_scales_x", type=str,
            help="MMD scale parameter for kernel applied on x")
    parser.add_argument("--mmd_scales_y", type=str,
            help="MMD scale parameter for kernel applied on y")
    parser.add_argument("--kernel_lr", type=float, default=0.01,
            help="(only GMMN) Learning rate for kernel parameter tuning")
    parser.add_argument("--mmd_lambda", type=float, default=1.0,
            help="(only CGMMN) Regularizer lambda to stabilize matrix inversions in MMD")
    parser.add_argument("--sqrt_loss", type=int, default=1,
            help="""(only CGMMN) Use square root of the loss,
                can yield better results, see Li et al.""")

    # NN-based models (mdn, nn_reg, nn_het, dctd, cgmmn, gmmn)
    parser.add_argument("--network", type=str,
            help="""Name of network specification to use, available specs can be
            found in nn_specs directory.""")
    parser.add_argument("--l2_reg", type=float, default=0.0,
            help="L2-regularization added to cost function (aka weight decay)")

    # MDN
    parser.add_argument("--mixture_comp", type=int, default=5,
            help="Amount of mixture components in MDN")
    parser.add_argument("--log_coefficients", type=int, default=0,
            help="If mixture coefficients should be logged to wandb")

    # GP
    parser.add_argument("--gp_kernel", type=str, default="rbf",
            help="Which kernel type to use in GP")
    parser.add_argument("--opt_restarts", type=int, default=0,
            help="Restarts in kernel hyperparameter optimization process")

    # DCTD
    parser.add_argument("--imp_samples", type=int, default=500,
            help="Amount of importance samples used to estimate normalization Z")
    parser.add_argument("--proposal_scales", type=str, default="0.5,1,5",
            help="Scales of gaussians in mixture proposal distribution")
    parser.add_argument("--mode_find_steps", type=int, default=100, help=(
        "Amount of optimization steps in mode finding for DCTD proposal distribution"))
    parser.add_argument("--mode_find_lr", type=float, default=1e-2,
            help="Learning rate in mode finding for DCTD proposal distribution")
    parser.add_argument("--plot_dctd_modes", type=int, default=0,
            help="Create additional scatter plot with modes of DCTD model")

    args = parser.parse_args()
    config = vars(args)

    # Read additional config from file
    if args.config:
        assert os.path.exists(args.config), "No config file: {}".format(args.config)
        config_from_file = utils.json_to_dict(args.config)

        # Make sure all options in config file also exist in argparse config.
        # Avoids choosing wrong parameters because of typos etc.
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(["Unknown option in config file: {}".format(opt)
            for opt in unknown_options])
        assert (not unknown_options), unknown_error

        config.update(config_from_file)

    assert config["dataset"], "No dataset specified"
    assert config["dataset"] in dataset_list.sets, (
            "Unknown dataset: {}".format(config["dataset"]))

    assert config["model"], "No model specified"
    assert config["model"] in models, "Unknown model '{}'".format(config["model"])

    for split_option in [
            "plot_pdf",
            "plot_pdf_index",
            "cond_scatter",
            "mmd_scales",
            "mmd_scales_x",
            "mmd_scales_y",
            "proposal_scales",
            "cond_plot_trajectories",
            ]:
        opt_value = config[split_option]
        if opt_value:
            if "(" in opt_value:
                # entries are tuples (e.g. multi-dimensional x)
                # extra "," to always get a tuple of tuples
                parsed = ast.literal_eval(opt_value + ",")

                # Make into list of floats
                config[split_option] = [[float(e) for e in v] for v in parsed]
            else:
                # entries are single floats
                config[split_option] = [float(s) for s in
                        opt_value.split(",")]

    return config

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    config = get_config()

    # Set all random seeds
    seed_all(config["seed"])

    # Figure out what device to use, (GP needs data in cpu-memory)
    if (not config["cpu"]) and (not config["model"]=="gp") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Read data
    dataset = dataset_list.get_dataset_spec(config["dataset"])()
    train_data, val_data, test_data = dataset.load(device)
    config["x_dim"] = dataset.x_dim
    config["y_dim"] = dataset.y_dim

    # Init wandb
    if config["name"] == None:
        config["name"] = "{}-{}-{}".format(config["model"], config["dataset"],
                time.strftime("%H-%M"))
    tags = []
    if config["test"]:
        tags.append("test")
    # Also setting config
    wandb.init(project=constants.WANDB_PROJECT, config=config,
            name=config["name"], tags=tags)

    # Set computational device (should not be synched to wandb)
    config["device"] = device

    # Load model
    model = models[config["model"]](config)

    # Train model
    if config["train"]:
        # Create evaluation function to feed to model for use during training
        def val_model(model, epoch_i):
            if config["kde_val"]:
                # Eval KDE
                ev.evaluate_model(model, data=val_data, config=config,
                        epoch_i=epoch_i, kde=True, make_plots=False)
            # Eval true
            return ev.evaluate_model(model, data=val_data,
                config=config, epoch_i=epoch_i)

        model.train(train_data, config, val_func=val_model)

    # Test model
    if config["test"]:
        # determine kernel scaling from evaluation set
        _, best_ks = utils.kde_eval(model, val_data, config)
        print("Kernel Scale: {}".format(best_ks))
        model.kernel_scale = best_ks # Store best kernel scale in model

        # Get true (according to model) log-likelihood
        true_evaluation_vals = ev.evaluate_model(model, test_data, config=config,
                    make_plots=True) # Make plots only this run

        # Average testing using KDE over multiple random seeds
        eval_list = [] # list of dicts mapping eval_metric -> value
        for i in range(config["test_runs"]):
            seed_all(constants.TEST_SEEDS[i])
            evaluation_vals = ev.evaluate_model(model, test_data, config=config,
                    make_plots=False, kde=True)
            eval_list.append(evaluation_vals)

        wandb_test_dict = {}
        for key in eval_list[0].keys():
            eval_values = [val_dict[key] for val_dict in eval_list]
            wandb_test_dict["test_{}_mean".format(constants.EVAL_NAME_MAP[key])] =\
                np.mean(eval_values)
            wandb_test_dict["test_{}_std_dev".format(constants.EVAL_NAME_MAP[key])] =\
                np.std(eval_values)

            if key in true_evaluation_vals:
                wandb_test_dict["test_{}_true".format(constants.EVAL_NAME_MAP[key])] =\
                    true_evaluation_vals[key]

        wandb.log(wandb_test_dict)
        test_print_string = "\t".join([])

        print("Test metrics over {} seeds".format(config["test_runs"]))
        for key in eval_list[0].keys():
            wandb_key = "test_{}_".format(constants.EVAL_NAME_MAP[key])
            print("{}: {:.5}±{:.5}".format(
                key,
                wandb_test_dict[wandb_key+"mean"],
                wandb_test_dict[wandb_key+"std_dev"],
                ))
            if key in true_evaluation_vals:
                print("true {}: {:.5}".format(key, wandb_test_dict[wandb_key+"true"]))

        if config["eval_div"]:
            divergence = ev.estimate_divergences(model, config, {
                    "train": train_data,
                    "val": val_data,
                    "test": test_data,
                })

if __name__ == "__main__":
    main()

