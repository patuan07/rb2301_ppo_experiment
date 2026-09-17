import math
import unittest

from rb2301_ca1.maze_layout import (
    DIFFICULTIES,
    LayoutConfig,
    find_clear_path,
    generate_layout,
    layout_config_for_difficulty,
)


class MazeLayoutTests(unittest.TestCase):
    def test_seed_is_reproducible_and_pool_size_is_fixed(self):
        config = LayoutConfig()
        first = generate_layout(2301, config)
        second = generate_layout(2301, config)
        self.assertEqual(first, second)
        self.assertEqual(len(first.all_positions), config.obstacle_pool_size)
        self.assertGreaterEqual(len(first.active_positions), config.minimum_obstacles)
        self.assertLessEqual(
            len(first.active_positions),
            config.maximum_active_obstacles,
        )

    def test_accepted_layout_has_inflated_astar_route(self):
        config = LayoutConfig()
        for seed in range(10):
            layout = generate_layout(seed, config)
            path, length = find_clear_path(layout.active_positions, config)
            self.assertTrue(path)
            self.assertEqual(path[0], (config.start_x, config.start_y))
            self.assertEqual(path[-1], (config.goal_x, config.goal_y))
            self.assertAlmostEqual(length, layout.path_length)
            self.assertGreaterEqual(
                max(abs(y) for _, y in path),
                config.minimum_path_lateral_excursion,
            )
            direct = math.hypot(
                config.goal_x - config.start_x,
                config.goal_y - config.start_y,
            )
            self.assertGreaterEqual(length, direct * config.minimum_path_stretch)
            for path_x, path_y in path:
                for obstacle_x, obstacle_y, _ in layout.active_positions:
                    self.assertGreater(
                        math.hypot(path_x - obstacle_x, path_y - obstacle_y),
                        config.obstacle_clearance - 1e-9,
                    )

    def test_no_reserved_centre_lane(self):
        config = LayoutConfig()
        layout = generate_layout(9, config)
        self.assertGreaterEqual(
            layout.straight_blockers,
            config.minimum_straight_blockers,
        )
        self.assertTrue(
            any(
                abs(y - config.start_y) <= config.straight_path_half_width
                for _, y, _ in layout.active_positions
            )
        )

    def test_all_curriculum_levels_generate(self):
        self.assertEqual(DIFFICULTIES, ("easy", "medium", "hard"))
        for difficulty in DIFFICULTIES:
            layout = generate_layout(17, layout_config_for_difficulty(difficulty))
            self.assertTrue(layout.path_centres)

    def test_unused_models_are_parked_below_the_world(self):
        layout = generate_layout(2301)
        parked = layout.all_positions[len(layout.active_positions) :]
        self.assertTrue(all(z < 0.0 for _, _, z in parked))

    def test_astar_does_not_cut_between_blocked_cardinal_cells(self):
        config = LayoutConfig(
            goal_x=0.1,
            goal_y=0.1,
            corridor_half_width=0.3,
            boundary_clearance=0.0,
            planning_resolution=0.1,
            obstacle_clearance=0.01,
        )
        path, length = find_clear_path(
            ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
            config,
        )
        self.assertEqual(path, ())
        self.assertTrue(math.isinf(length))


if __name__ == "__main__":
    unittest.main()
