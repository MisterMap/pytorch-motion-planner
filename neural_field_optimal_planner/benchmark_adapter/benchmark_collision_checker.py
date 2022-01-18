import numpy as np

from ..collision_checker import CollisionChecker


class BenchmarkCollisionChecker(CollisionChecker):
    def __init__(self, benchmark):
        super().__init__()
        self._benchmark = benchmark

    def check_collision(self, test_positions):
        return self._benchmark.is_collision(test_positions)
