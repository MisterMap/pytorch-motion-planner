#! /usr/bin/python3.9
import numpy as np
import torch
from matplotlib import pyplot as plt
import argparse
import json
import os
import time

from neural_field_optimal_planner.planner_factory import PlannerFactory
from neural_field_optimal_planner.plotting_utils import prepare_figure, plot_planner_data, plot_nerf_opt_planner
from neural_field_optimal_planner.benchmark_adapter import BenchmarkAdapter
from neural_field_optimal_planner.benchmark_adapter.benchmark_collision_checker import BenchmarkCollisionChecker

torch.random.manual_seed(100)
np.random.seed(400)

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

    planner = PlannerFactory.make_constrained_onf_planner(collision_checker)
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
