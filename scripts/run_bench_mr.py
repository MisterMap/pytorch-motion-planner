#! /usr/bin/python3

import numpy as np
import torch
from matplotlib import pyplot as plt
import argparse
import os
from pytorch_lightning.utilities import AttributeDict
import json

from neural_field_optimal_planner.planner_factory import PlannerFactory
from neural_field_optimal_planner.plotting_utils import prepare_figure, plot_planner_data, plot_nerf_opt_planner, \
    plot_collision_positions
from neural_field_optimal_planner.benchmark_adapter import BenchmarkAdapter
from neural_field_optimal_planner.benchmark_adapter.benchmark_collision_checker import BenchmarkCollisionChecker

start_point = np.array(list(map(int, os.environ.get('START').split())))
goal_point = np.array(list(map(int, os.environ.get('END').split())))
start_point = np.append(start_point, 0)
goal_point = np.append(goal_point, 0)
torch.random.manual_seed(100)
np.random.seed(400)

planner_parameters = AttributeDict(
    device="cpu",
    trajectory_length=100,
    collision_model=AttributeDict(
        mean=0,
        sigma=3,
        use_cos=True,
        bias=True,
        use_normal_init=True,
        angle_encoding=True,
        name="ONF"
    ),
    collision_optimizer=AttributeDict(
        lr=1e-2,
        betas=(0.9, 0.9)
    ),
    trajectory_optimizer=AttributeDict(
        lr=5e-2,
        betas=(0.9, 0.9)
    ),
    planner=AttributeDict(
        name="ConstrainedNERFOptPlanner",
        trajectory_random_offset=0.02,
        collision_weight=20,
        velocity_hessian_weight=0.5,
        random_field_points=10,
        init_collision_iteration=0,
        constraint_deltas_weight=200,
        multipliers_lr=0.1,
        init_collision_points=100,
        reparametrize_trajectory_freq=10,
        optimize_collision_model_freq=1,
        angle_weight=5,
        angle_offset=0.3,
        boundary_weight=1,
        collision_multipliers_lr=1e-3
    )
)

parser = argparse.ArgumentParser()
parser.add_argument("settings")
parser.add_argument("--show", default=False)
args = parser.parse_args()

print("Start with config", args.settings)
benchmark = BenchmarkAdapter(args.settings)
print("Benchmark adapter initialized")
collision_checker = BenchmarkCollisionChecker(benchmark, benchmark.bounds())
print("Collision checker initialized")
planner = PlannerFactory.make_constrained_onf_planner(collision_checker, planner_parameters)
# goal_point = benchmark.goal().as_vec()
# goal_point = np.array([95,92,0])
# start_point = benchmark.start().as_vec()
# start_point = np.array([9,7,0])
trajectory_boundaries = benchmark.bounds()
#test_arr = [start_point, goal_point]
#benchmark.evaluate_and_save_results([start_point, goal_point], "constrained_onf_planner")

planner.init(start_point, goal_point, trajectory_boundaries)
device = planner._device
collision_model = planner._collision_model
is_show = args.show
fig = None

if is_show:
    fig = plt.figure(dpi=200)

for i in range(1000):
    planner.step()
    if is_show:
        trajectory = planner.get_path()
        fig.clear()
        prepare_figure(trajectory_boundaries)
        plot_planner_data(trajectory, collision_model, trajectory_boundaries, np.zeros((0, 2)), device=device)
        # plot_nerf_opt_planner(planner)
        # plot_collision_positions(planner.checked_positions, planner.truth_collision)
        plt.pause(0.01)
        

benchmark.evaluate_and_save_results(planner.get_path(), "constrained_onf_planner")
# with open("/home/evgeny/pytorch-motion-planner/notebooks/benchmark/corridor_results.json") as data_file:
#     data = json.load(data_file)
# #    for el in data['runs']:
# del data['runs'][0]
# os.remove("/home/evgeny/pytorch-motion-planner/notebooks/benchmark/corridor_results.json") 
# jsonString = json.dumps(data)
# jsonFile = open("/home/evgeny/pytorch-motion-planner/notebooks/benchmark/corridor_results.json", "w")
# jsonFile.write(jsonString)
# jsonFile.close()