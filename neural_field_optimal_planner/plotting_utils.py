import numpy as np
import torch.nn.functional
from matplotlib import pyplot as plt
from torch import nn


def plot_planner_data(plotted_trajectory, collision_model, boundaries, obstacle_points, device="cpu"):
    plot_model_heatmap(collision_model, boundaries, device)
    plot_obstacle_points(obstacle_points)
    plt.scatter(plotted_trajectory[:, 0], plotted_trajectory[:, 1], color="yellow", s=1)
    plt.tight_layout()


def prepare_figure(boundaries, number=None):
    fig = plt.figure(number, dpi=200)
    plt.gca().set_aspect("equal")
    plt.xlim(boundaries[0], boundaries[1])
    plt.ylim(boundaries[2], boundaries[3])
    return fig


def plot_model_heatmap(model, boundaries, device):
    grid_x, grid_y = np.meshgrid(np.linspace(boundaries[0], boundaries[1], 100),
                                 np.linspace(boundaries[2], boundaries[3], 100))
    grid = np.stack([grid_x, grid_y], axis=2).reshape(-1, 2)
    obstacle_probabilities = nn.functional.softplus(model(torch.tensor(grid.astype(np.float32), device=device)))
    obstacle_probabilities = obstacle_probabilities.cpu().detach().numpy().reshape(100, 100)
    grid = grid.reshape(100, 100, 2)
    plt.gca().pcolormesh(grid[:, :, 0], grid[:, :, 1], obstacle_probabilities, cmap='RdBu', shading='auto',
                         vmin=0, vmax=10)


def plot_obstacle_points(obstacle_points):
    plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], color="black")
