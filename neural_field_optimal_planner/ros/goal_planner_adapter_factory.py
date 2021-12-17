import torch

from .collision_checker_adapter import CollisionCheckerAdapter
from .goal_planner_adapter import GoalPlannerAdapter
from .map_adapter import MapAdapter
from .robot_state import RobotState
from .transform_receiver_factory import TransformReceiverFactory
from ..collision_checker import CollisionChecker
from ..nerf_opt_planner import NERFOptPlanner
from ..onf_model import ONF
from ..planner_factory import PlannerFactory


class GoalPlannerAdapterFactory(object):
    @staticmethod
    def make_goal_planner_adapter():
        transform_receiver_factory = TransformReceiverFactory()
        robot_state = RobotState(transform_receiver_factory)
        map_adapter = MapAdapter(map_topic_name="map")
        collision_checker = CollisionChecker(robot_radius=0.3)
        collision_checker_adapter = CollisionCheckerAdapter(collision_checker, point_topic_name="obstacle_points",
                                                            map_adapter=map_adapter)
        # planner = PlannerFactory.make_onf_planner(collision_checker_adapter)
        planner = PlannerFactory.make_constrained_onf_planner(collision_checker_adapter)
        return GoalPlannerAdapter(planner, map_adapter, robot_state, goal_topic_name="/move_base_simple/goal",
                                  path_topic_name="/path", planning_timeout=0.1, planner_rate=10, is_point=False)

