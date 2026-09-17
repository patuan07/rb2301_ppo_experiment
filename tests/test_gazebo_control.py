import subprocess
import unittest

from rb2301_ca1.gazebo_control import GazeboCommandError, GazeboController


class RecordingRunner:
    def __init__(self, output="data: true", returncode=0):
        self.commands = []
        self.output = output
        self.returncode = returncode

    def __call__(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, self.returncode, self.output, "")


class SequenceRunner:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append((command, kwargs))
        output = next(self.outputs)
        return subprocess.CompletedProcess(command, 0, output, "")


class GazeboControllerTests(unittest.TestCase):
    def test_robot_pose_service_request(self):
        runner = RecordingRunner()
        controller = GazeboController(runner=runner)
        controller.set_entity_pose("nanocar", 1.0, -2.0, yaw=0.5)
        command = runner.commands[0][0]
        self.assertIn("/world/empty/set_pose", command)
        self.assertIn('name: "nanocar"', command[-1])
        self.assertIn("x: 1.00000000", command[-1])

    def test_randomization_updates_all_64_cans_at_once(self):
        runner = RecordingRunner()
        controller = GazeboController(runner=runner)
        layout = controller.randomize_obstacles(2301)
        command = runner.commands[0][0]
        self.assertEqual(len(layout.all_positions), 64)
        self.assertIn("/world/empty/set_pose_vector", command)
        self.assertIn('name: "coke1"', command[-1])
        self.assertIn('name: "coke64"', command[-1])

    def test_false_service_response_raises(self):
        controller = GazeboController(runner=RecordingRunner(output="data: false"))
        with self.assertRaises(GazeboCommandError):
            controller.reset_world()

    def test_timed_out_service_response_is_retried(self):
        runner = SequenceRunner(["Service call timed out", "data: true"])
        controller = GazeboController(
            runner=runner,
            retry_attempts=2,
            retry_delay_seconds=0.0,
        )
        controller.reset_world()
        self.assertEqual(len(runner.commands), 2)

    def test_wait_for_service_checks_worker_partition(self):
        runner = RecordingRunner(output="/world/empty/set_pose_vector\n")
        controller = GazeboController(runner=runner)
        controller.wait_for_service(
            "/world/empty/set_pose_vector",
            timeout_seconds=0.1,
            poll_interval_seconds=0.01,
        )
        self.assertEqual(runner.commands[0][0], ["gz", "service", "-l"])


if __name__ == "__main__":
    unittest.main()
