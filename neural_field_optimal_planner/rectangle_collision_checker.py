from .collision_checker import CollisionChecker


class RectangleCollisionChecker(CollisionChecker):
    def __init__(self, center, box, collision_boundaries=None):
        super().__init__(collision_boundaries)
        self._center = center
        self._box = box

    def check_collision(self, test_positions):
        positions = test_positions.inv().apply(self._obstacle_points)
        result = positions[:, 0] > self._box[0]
        result &= positions[:, 0] < self._box[1]
        result &= positions[:, 1] > self._box[2]
        result &= positions[:, 1] < self._box[3]
        return result | self._check_boundaries_collision(test_positions)
