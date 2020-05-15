import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
import torch

import constants

# Set default settings on import
plt.rcParams.update({'font.size': 13})
plt.rc("axes", labelsize=18)
plt.rc("axes", titlesize=21)
plt.rc("legend", fontsize=18)

def save_plot(file_name, wandb_name, fig):
    # Style
    fig.tight_layout()

    if file_name:
        # Save .pdf-file
        save_path = os.path.join(wandb.run.dir, "{}.pdf".format(file_name))
        plt.savefig(save_path)

    # Save image to wandb
    wandb.log({wandb_name: wandb.Image(plt)})
    plt.close(fig)


def plot_samples(sample_sets, file_name, wandb_name, labels=None, range_dataset=None,
        title=None, both_y=False):
    n_sets = len(sample_sets)
    fig, axes = plt.subplots(nrows=1, ncols=n_sets, squeeze=False,
            figsize=(7.*n_sets, 4.5))

    # If not given, determine ranges from sample_set[0]
    if not range_dataset:
        range_dataset = sample_sets[0]

    x_range = (min(range_dataset.x), max(range_dataset.x))
    y_range = (min(range_dataset.y), max(range_dataset.y))

    for set_i, (samples, ax) in enumerate(zip(sample_sets, axes[0])):
        ax.scatter(samples.x, samples.y, s=3, color="green")

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        if labels:
            ax.set(title=labels[set_i])
        if both_y:
            ax.set(xlabel="$y_1$", ylabel="$y_2$")
        else:
            ax.set(xlabel="$x$", ylabel="$y$")

    if title:
        fig.suptitle(title)

    save_plot(file_name, wandb_name, fig)

def plot_positions(sample_sets, file_name, wandb_name, title=None):
    n_sets = len(sample_sets)
    n_batch = sample_sets[0].shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=n_sets, squeeze=False,
            figsize=(7.*n_sets, 4.5))

    reshaped_sets = [sample_set.reshape(n_batch,-1,2) for sample_set in sample_sets]

    range_dataset = reshaped_sets[0]
    x_range = (range_dataset[:,:,0].min(), range_dataset[:,:,0].max())
    y_range = (range_dataset[:,:,1].min(), range_dataset[:,:,1].max())

    for set_i, (samples, ax) in enumerate(zip(reshaped_sets, axes[0])):
        for pos_i in range(samples.shape[1]):
            ax.scatter(samples[:,pos_i,0], samples[:,pos_i,1], s=5,
                    marker=constants.SCATTER_MARKERS[pos_i])

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        ax.set(xlabel="$y_1,y_3,...$", ylabel="$y_2,y_4,...$")

    if title:
        fig.suptitle(title)

    save_plot(file_name, wandb_name, fig)

def plot_pdfs(pdfs, file_name, wandb_name, support=(-1,1), labels=None,
        n_points=constants.PLOT_POINTS, title=None):
    if labels:
        assert len(pdfs) == len(labels), "Amount of pdfs and labels given must agree"

    xs = np.linspace(support[0], support[1], num=n_points)
    pdf_ys = [np.array([pdf(x) for x in xs]) for pdf in pdfs]

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111)

    for pdf_i, ys in enumerate(pdf_ys):
        line, = ax.plot(xs, ys, color=constants.COLORS[pdf_i],
                ls=constants.LINE_STYLES[pdf_i], linewidth=3)

        if labels:
            line.set_label(labels[pdf_i])

    if labels:
        ax.legend(prop={"size": 12})

    if title:
        ax.set(title=title)
    ax.set(xlabel="$y$", ylabel=("$p(y|x)$"))

    save_plot(file_name, wandb_name, fig)

def plot_functions(lines, xs, file_name, wandb_name, title=None):
    fig = plt.figure(figsize=(7., 4.5))
    ax = fig.add_subplot(111)

    for ys in lines:
        line, = ax.plot(xs, ys, color="red")

    if title:
        ax.set(title=title)
    ax.set(xlabel="$x$", ylabel="$y$")

    save_plot(file_name, wandb_name, fig)

def plot_trajectories(trajs, file_name, wandb_name, labels=None, title=None,
        range_trajs=None):
    n_trajs = len(trajs)
    n_samples = trajs[0].shape[0] # Size of batch dimension

    fig, axes = plt.subplots(nrows=1, ncols=n_trajs, squeeze=False,
            figsize=(5*n_trajs, 4.5))

    # Determine axis ranges
    if not (type(range_trajs) == torch.Tensor):
        # If not given, use first in list
        range_trajs = trajs[0]
    reshaped_range_traj = range_trajs.reshape(n_samples, -1, 2)
    range_xs = reshaped_range_traj[:,:,0]
    range_ys = reshaped_range_traj[:,:,1]
    x_range = (range_xs.min(), range_xs.max())
    y_range = (range_ys.min(), range_ys.max())

    # Make axis scales the same, as to not distort distances
    axis_range = max((y_range[1] - y_range[0]), (x_range[1] - x_range[0]))
    x_center = x_range[0] + (x_range[1] - x_range[0])/2
    y_center = y_range[0] + (y_range[1] - y_range[0])/2
    x_range = ((x_center - axis_range/2), (x_center + axis_range/2))
    y_range = ((y_center - axis_range/2), (y_center + axis_range/2))

    # Reshape data
    reshaped_trajs = [traj.reshape(n_samples, -1, 2) for traj in trajs]
    # Shape is (batch_size, n_steps, 2) (2 for 2D position (x,y))

    for traj_i, (t, ax) in enumerate(zip(reshaped_trajs, axes[0])):
        ax.plot(t[:,:,0].T, t[:,:,1].T, linewidth=3)

        ax.set(xlabel="$y_1,y_3,...$", ylabel="$y_2,y_4,...$")
        if labels:
            ax.set(title=labels[traj_i])
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

    if title:
        fig.suptitle(title)
    save_plot(file_name, wandb_name, fig)

