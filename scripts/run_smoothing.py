#! /usr/bin/python3.9
import numpy as np
import torch
from matplotlib import pyplot as plt
import argparse
import json
import os
import time
from pytorch_lightning.utilities import AttributeDict

from neural_field_optimal_planner.planner_factory import PlannerFactory
from neural_field_optimal_planner.plotting_utils import *
from neural_field_optimal_planner.test_environment_builder import TestEnvironmentBuilder

torch.random.manual_seed(100)
np.random.seed(400)

planner_parameters = AttributeDict(
    device="cpu",
    trajectory_length=100,
    collision_model=AttributeDict(
        mean=0,
        sigma=1,
        use_cos=True,
        bias=True,
        use_normal_init=True,
        angle_encoding=True,
        name="ONF"
    ),
    trajectory_initializer=AttributeDict(
        name="TrajectoryInitializer",
        resolution=0.05
    ),
    collision_optimizer=AttributeDict(
        lr=5e-2,
        betas=(0.9, 0.9)
    ),
    trajectory_optimizer=AttributeDict(
        lr=1e-2,
        betas=(0.9, 0.9)
    ),
    planner=AttributeDict(
        name="ConstrainedNERFOptPlanner",
        trajectory_random_offset=0.02,
        collision_weight=1,
        velocity_hessian_weight=0.5,
        random_field_points=10,
        init_collision_iteration=0,
        constraint_deltas_weight=20,
        multipliers_lr=0.1,
        init_collision_points=100,
        reparametrize_trajectory_freq=10,
        optimize_collision_model_freq=1,
        angle_weight=0.5,
        angle_offset=0.3,
        boundary_weight=1,
        collision_multipliers_lr=1e-3
    )
)


parser = argparse.ArgumentParser()
parser.add_argument("settings")
parser.add_argument("--show", default=False)
args = parser.parse_args()

print("start with config", args.settings)

# read result of planners and smoothers
with open(args.settings, 'r') as f:
    initial_results = json.load(f)

results = initial_results.copy()
run_numbers = len(initial_results["runs"])

for run_number in range(run_numbers):
    benchmark = BenchmarkAdapter(args.settings)
    collision_checker = BenchmarkCollisionChecker(benchmark, benchmark.bounds())

    planner = PlannerFactory.make_constrained_onf_planner(collision_checker, planner_parameters)
    goal_point = benchmark.goal().as_vec()
    start_point = benchmark.start().as_vec()
    trajectory_boundaries = benchmark.bounds()

    planners_list = list(initial_results["runs"][run_number]["plans"])
    initial_path = np.array(initial_results["runs"][run_number]["plans"][planners_list[-1]]["trajectory"])

    initial_path = np.array(initial_path, dtype=np.float32)[0:500:5]  # TODO solve 100 - 500 path points tragedy

    planner.init(start_point, goal_point, trajectory_boundaries)
    planner.set_trajectory(initial_path)
    device = planner._device
    collision_model = planner._collision_model
    is_show = args.show
    fig = None
    if is_show:
        fig = plt.figure(dpi=200)

    t1 = time.time()
    for i in range(1000):
        planner.step()
        if is_show:
            trajectory = planner.get_path()
            fig.clear()
            prepare_figure(trajectory_boundaries)
            plot_planner_data(trajectory, collision_model, trajectory_boundaries, np.zeros((0, 2)), device=device)
            plot_nerf_opt_planner(planner)
            plt.pause(0.01)

    t2 = time.time()
    t = t2 - t1
    smoother_name = "constrained_onf_smoother"
    benchmark.evaluate_and_save_results_smoothing(initial_path, planner.get_path(), planners_list[-1],
                                                  smoother_name, time=t)

    with open(args.settings, 'r') as f:
        # read ONF results
        new_results = json.load(f)

    # join initial results and ONF results
    results["runs"][run_number]["plans"][planners_list[-1]]["smoothing"][smoother_name] = \
        new_results["runs"][0]["plans"][planners_list[-1]]["smoothing"][smoother_name]

# write down joint results
with open(args.settings, 'w+') as f:
    f.write(json.dumps(results, indent=4))
