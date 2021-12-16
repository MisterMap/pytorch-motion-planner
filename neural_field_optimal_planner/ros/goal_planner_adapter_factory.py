import torch

from .collision_checker_adapter import CollisionCheckerAdapter
from .goal_planner_adapter import GoalPlannerAdapter
from .map_adapter import MapAdapter
from .robot_state import RobotState
from .transform_receiver_factory import TransformReceiverFactory
from ..collision_checker import CollisionChecker
from ..nerf_opt_planner import NERFOptPlanner
from ..onf_model import ONF


class GoalPlannerAdapterFactory(object):
    @staticmethod
    def make_goal_planner_adapter():
        transform_receiver_factory = TransformReceiverFactory()
        robot_state = RobotState(transform_receiver_factory)
        map_adapter = MapAdapter(map_topic_name="map")
        collision_checker = CollisionChecker(robot_radius=0.3)
        collision_checker_adapter = CollisionCheckerAdapter(collision_checker, point_topic_name="obstacle_points",
                                                            map_adapter=map_adapter)
        planner = GoalPlannerAdapterFactory.make_onf_planner(collision_checker_adapter)
        return GoalPlannerAdapter(planner, map_adapter, robot_state, goal_topic_name="/move_base_simple/goal",
                                  path_topic_name="/path", planning_timeout=1, planner_rate=1)

    @staticmethod
    def make_onf_planner(collision_checker):
        device = "cpu"
        collision_model = ONF(1.5, 1).to(device)
        collision_optimizer = torch.optim.Adam(collision_model.parameters(), 1e-2)
        trajectory = torch.zeros(100, 2, requires_grad=True, device=device)
        trajectory_optimizer = torch.optim.Adam([trajectory], 1e-2, betas=(0.9, 0.999))
        return NERFOptPlanner(trajectory, collision_model, collision_checker, collision_optimizer,
                              trajectory_optimizer, trajectory_random_offset=0.02, collision_weight=0.01,
                              velocity_hessian_weight=3, random_field_points=10, init_collision_iteration=400)
