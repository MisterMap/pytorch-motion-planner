import torch

from .constrained_nerf_opt_planner import ConstrainedNERFOptPlanner
from .nerf_opt_planner import NERFOptPlanner
from .onf_model import ONF


class PlannerFactory(object):
    @staticmethod
    def make_onf_planner(collision_checker):
        device = "cpu"
        collision_model = ONF(1.5, 1).to(device)
        collision_optimizer = torch.optim.Adam(collision_model.parameters(), 1e-3, betas=(0.9, 0.9))
        trajectory = torch.zeros(100, 2, requires_grad=True, device=device)
        trajectory_optimizer = torch.optim.Adam([trajectory], 1e-2, betas=(0.9, 0.999))
        return NERFOptPlanner(trajectory, collision_model, collision_checker, collision_optimizer,
                              trajectory_optimizer, trajectory_random_offset=0.02, collision_weight=0.01,
                              velocity_hessian_weight=3, random_field_points=10, init_collision_iteration=400)

    @staticmethod
    def make_constrained_onf_planner(collision_checker):
        device = "cpu"
        # collision_model = ONF(0, 0.5, use_cos=True, use_normal_init=False, bias=False).to(device)
        collision_model = ONF(1.5, 1).to(device)
        collision_optimizer = torch.optim.Adam(collision_model.parameters(), 2e-3, betas=(0.9, 0.9))
        trajectory = torch.zeros(100, 3, requires_grad=True, device=device)
        trajectory_optimizer = torch.optim.Adam([trajectory], 1e-2, betas=(0.9, 0.999))
        return ConstrainedNERFOptPlanner(trajectory, collision_model, collision_checker, collision_optimizer,
                                         trajectory_optimizer, trajectory_random_offset=0.02, collision_weight=0.1,
                                         velocity_hessian_weight=0.5, random_field_points=10,
                                         init_collision_iteration=200, constraint_deltas_weight=2, multipliers_lr=0.01)
