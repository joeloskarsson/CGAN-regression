import wandb
import torch
import numpy as np

import dataset_specifications.dataset as ds
import models.cgan_versions as cgans
import utils
import constants
import visualization as vis

# Function that handles evaluation both at validation and test-time
# Also checks if plotting should be done, prepares the data that should be plotted and
# calls correct plotting functions.
def evaluate_model(model, data, config, epoch_i=None,
        make_plots=True, kde=False):
    testing = (epoch_i == None) # Need None comparison since epoch_i can be 0

    if kde:
        ks = None
        if testing:
            assert model.kernel_scale, "No kernel scale stored for model"
            ks = model.kernel_scale

        evaluation_vals, _ = utils.kde_eval(model, data, config, kernel_scale=ks)
        val_method = "kde"
    else:
        evaluation_vals = model.eval(data, config, use_best_kernel_scale=testing)
        val_method = "true"

    metric_string = "\t".join(["{}: {:.5}".format(key, val) for
        (key,val) in evaluation_vals.items()])
    if testing:
        print("Test, {}\t{}".format(val_method, metric_string))
    else:
        wandb.log({
            "{}_{}".format(val_method, constants.EVAL_NAME_MAP[key]): val
            for (key, val) in evaluation_vals.items()
        })
        print("Epoch {}, {}\t{}".format(epoch_i, val_method, metric_string))

    if make_plots:
        if config["scatter"]:
            model_samples = model.sample(data.x, batch_size=config["eval_batch_size"])
            if type(model_samples) == torch.Tensor:
                # Move to cpu memory
                model_samples = model_samples.to("cpu")
            model_data = ds.LabelledData(x=data.x, y=model_samples)

            if config["y_dim"] == 1:
                if testing:
                    plot_title = None
                    wandb_name = "test_scatter"
                    file_name = "test_scatter"

                    if config["plot_prefix"]:
                        file_name = config["plot_prefix"] + file_name

                    if config["plot_gt"]:
                        sample_sets = [data]
                    else:
                        sample_sets = [model_data]

                    labels = None
                else:
                    plot_title = "Validation epoch {}".format(epoch_i)
                    file_name = None
                    wandb_name = "scatter"
                    sample_sets = [data, model_data]
                    labels = ["Ground Truth", config["model"]]

                    # Mode plotting for DCTD model
                    # (mainly for debug / hyperparamater tweaking)
                    if config["plot_dctd_modes"]:
                        assert config["model"] == "dctd", (
                                "Can not plot modes for other model than DCTD")

                        modes = model.find_modes(data.x.to(config["device"])).to("cpu")
                        sample_sets.append(ds.LabelledData(x=data.x, y=modes))
                        labels.append("dctd modes")

                vis.plot_samples(sample_sets, file_name=file_name, wandb_name=wandb_name,
                        labels=labels, title=plot_title, range_dataset=data)
            elif config["y_dim"] == 2:
                model_cloud = torch.cat((data.x, model_samples), dim=1)
                true_cloud = torch.cat((data.x, data.y), dim=1)

                if config["plot_gt"]:
                    point_cloud = true_cloud.numpy()
                else:
                    model_w_class = torch.cat(
                            (model_cloud, 1*torch.ones_like(data.x)), dim=1)
                    true_w_class = torch.cat(
                            (true_cloud, 2*torch.ones_like(data.x)), dim=1)
                    point_cloud = torch.cat((model_w_class, true_w_class), dim=0).numpy()

                wandb.log({"3d_scatter": wandb.Object3D(point_cloud)})

        if config["cond_scatter"]:
            for x in config["cond_scatter"]:
                xs = torch.tensor(x, device=config["device"]).repeat(
                        config["eval_samples"], 1)

                model_samples = model.sample(xs)
                if type(model_samples) == torch.Tensor:
                    # Move to cpu memory
                    model_samples = model_samples.to("cpu")

                true_samples = data.spec.sample_ys(xs.to("cpu").numpy())
                dim_y = true_samples.shape[1]

                if dim_y > 2:
                    # Higher dimensional y
                    assert (dim_y % 2) == 0, (
                            "Can not create scatter for y with odd dimensionality")
                    model_set = model_samples
                    true_set = true_samples
                else:
                    model_set = ds.LabelledData(
                            x=model_samples[:,0], y=model_samples[:,1])
                    true_set = ds.LabelledData(
                            x=true_samples[:,0], y=true_samples[:,1])

                if type(x) == float:
                    x_name = "{:.4}".format(x)
                else:
                    # x is vector (python list)
                    x_name = "[{:.4}, ...]".format(x[0])

                if testing:
                    plot_title = None
                    wandb_name = "test_scatter_x_{}".format(x)
                    file_name = "test_scatter_x_{}".format(x)
                    if config["plot_prefix"]:
                        file_name = config["plot_prefix"] + file_name

                    if config["plot_gt"]:
                        sample_sets = [true_set]
                    else:
                        sample_sets = [model_set]

                    labels = None

                else:
                    plot_title = "Validation epoch {}".format(epoch_i)
                    file_name = None
                    wandb_name = "val_scatter_x_{}".format(x)

                    sample_sets = [true_set, model_set]
                    labels = ["Ground Truth", config["model"]]

                if dim_y > 2:
                    vis.plot_positions(sample_sets, file_name=file_name,
                            wandb_name=wandb_name, title=plot_title)
                else:
                    vis.plot_samples(sample_sets, file_name=file_name,
                            wandb_name=wandb_name, labels=labels, title=plot_title,
                            both_y=True)

        if config["plot_pdf"] or config["plot_pdf_index"]:
            # Build up list of x:s to plot pdf for
            pdf_xs = []
            x_names = []
            if config["plot_pdf"]:
                pdf_xs = pdf_xs + config["plot_pdf"]

                for x in config["plot_pdf"]:
                    if type(x) == float:
                        x_name = "{:.4}".format(x)
                    else:
                        # x is vector (python list)
                        x_name = str(x[:3])
                    x_names.append(x_name)

            if config["plot_pdf_index"]:
                pdf_xs = pdf_xs + [data.x[int(i)] for i in config["plot_pdf_index"]]
                x_names = x_names + [str(int(i)) for i in config["plot_pdf_index"]]

            for x, x_name in zip(pdf_xs, x_names):
                model_pdf = model.get_pdf(x, n_samples=config["eval_samples"])

                if type(x) == float:
                    short_x_name = x_name
                else:
                    # x is vector (python list)
                    short_x_name = "[{:.4}, ...]".format(x[0])

                if testing:
                    plot_title = None
                    wandb_name = "test_pdf_x_{}".format(x_name)
                    file_name = "test_pdf_x_{}".format(x_name)
                    if config["plot_prefix"]:
                        file_name = config["plot_prefix"] + file_name
                else:
                    plot_title = "x = {}".format(short_x_name)
                    file_name = None
                    wandb_name = "val_pdf_x_{}".format(x_name)

                pdfs = [model_pdf]
                labels = [constants.MODEL_NICE_NAMES[config["model"]]]

                if data.spec.synthetic:
                    true_pdf = data.spec.get_pdf(x)
                    pdfs.append(true_pdf)
                    labels.append("True")

                vis.plot_pdfs(pdfs, support=data.spec.get_support(x),
                        labels=labels, title=plot_title,
                        file_name=file_name, wandb_name=wandb_name)

        if config["plot_functions"]:
            x_range = (data.x.min().item(), data.x.max().item())
            xs = torch.linspace(x_range[0], x_range[1], steps=constants.PLOT_POINTS)

            lines = [model.sample(torch.unsqueeze(xs, dim=1), fixed_noise=True)
                        for _ in range(config["plot_functions"])]

            if testing:
                plot_title = None
                wandb_name = "test_functions"
                file_name = "test_functions"
                if config["plot_prefix"]:
                    file_name = config["plot_prefix"] + file_name
            else:
                plot_title = "Validation epoch {}".format(epoch_i)
                file_name = None
                wandb_name = "functions"

            vis.plot_functions(lines, xs, file_name, wandb_name, title=plot_title)

        if config["cond_plot_trajectories"]:
            assert data.spec.y_dim % 2 == 0, (
                    "Can not plot trajectories for dataset with odd y-dimensionality")

            for plot_spec in config["cond_plot_trajectories"]:
                if type(plot_spec) == float:
                    # Use sample at given index as conditioning value
                    sample_i = int(plot_spec)
                    cond_value = data.x[sample_i]
                else:
                    # Use given value as conditioning value
                    cond_value = torch.tensor(plot_spec)

                xs = cond_value.to(config["device"]
                        ).unsqueeze(0).repeat(config["plot_trajectories"],1)
                true_samples = torch.tensor(data.spec.sample_ys(xs.to("cpu").numpy()))
                if config["plot_gt"]:
                    model_samples = true_samples
                else:
                    model_samples = model.sample(xs).to("cpu")

                if testing:
                    plot_title = None
                    wandb_name = "test_trajectories_{}".format(plot_spec)
                    file_name = "test_trajectories_{}".format(plot_spec)
                    if config["plot_prefix"]:
                        file_name = config["plot_prefix"] + file_name

                    labels = None
                    trajs = [model_samples]
                else:
                    plot_title = "Validation epoch {}".format(epoch_i)
                    file_name = None
                    wandb_name = "trajectories_{}".format(plot_spec)

                    labels = ["Ground Truth", config["model"]]
                    trajs = [true_samples, model_samples]

                vis.plot_trajectories(trajs, file_name, wandb_name, labels=labels,
                        title=plot_title, range_trajs=true_samples)

    # Return evaluation metrics
    return evaluation_vals

# Estimate divergence between two distributions only based on two datasets
# The dataset from the model distribution is generated from the given model
# A CGAN-discriminator is used for the estimation
def estimate_divergences(model, config, datasets):
    assert config["eval_cgan"], "No CGAN specified for use in evaluation"

    # Split divergences
    divs = config["eval_div"].split(",")
    # Check so all divergences are defined
    for div in divs:
        assert (div in cgans.cgans), "Unknown evaluation divergence (CGAN): {}".format(
            div)

    for div in divs:
        # Create datasets drawn from model distribution
        sample_sets = {
                name: ds.LabelledData(
                    x=data.x,
                    y=model.sample(
                        data.x, batch_size=config["eval_batch_size"]).to("cpu")
                )
            for name, data in datasets.items()}

        # Create CGAN to get a classifier for correct divergence
        div_estimator = cgans.build_cgan(config, div_estimator=div)
        div_estimator.train_discriminator(config, datasets["train"],
                sample_sets["train"], datasets["val"], sample_sets["val"])
        test_div = div_estimator.compute_divergence(datasets["test"],
                sample_sets["test"], config["eval_batch_size"])

        wandb.log({"test_est_{}_div".format(div): test_div})
        print("Estimated {}-divergence: {}".format(div, test_div))

