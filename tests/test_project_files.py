import ast
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class ProjectFileTests(unittest.TestCase):
    def test_python_sources_parse(self):
        source_roots = [PROJECT / "src", PROJECT / "tests"]
        for source_root in source_roots:
            for path in source_root.rglob("*.py"):
                if any(part in {"build", "install", "log"} for part in path.parts):
                    continue
                with self.subTest(path=path):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_xml_and_sdf_files_parse(self):
        suffixes = {".xml", ".urdf", ".sdf"}
        for path in PROJECT.rglob("*"):
            if path.suffix not in suffixes:
                continue
            with self.subTest(path=path):
                ET.parse(path)

    def test_world_contains_boundaries_and_full_obstacle_pool(self):
        world_path = PROJECT / "src/rb2301_gz/worlds/obstacle_world_ca1.sdf"
        world = ET.parse(world_path).getroot().find("world")
        self.assertIsNotNone(world)
        model_names = {model.get("name") for model in world.findall("model")}
        self.assertTrue({"left_boundary", "right_boundary", "start_boundary"} <= model_names)
        includes = world.findall("include")
        self.assertEqual(len(includes), 64)
        self.assertEqual(
            [item.findtext("name") for item in includes],
            [f"coke{index}" for index in range(1, 65)],
        )
        self.assertNotIn("/home/", world_path.read_text(encoding="utf-8"))

    def test_robot_has_velocity_and_odometry_systems(self):
        robot_path = PROJECT / "src/rb2301_gz/urdf/nanocar_description.urdf"
        root = ET.parse(robot_path).getroot()
        filenames = {plugin.get("filename") for plugin in root.iter("plugin")}
        self.assertIn("gz-sim-velocity-control-system", filenames)
        self.assertIn("gz-sim-odometry-publisher-system", filenames)

    def test_bridge_directions_are_explicit(self):
        launch_path = PROJECT / "src/rb2301_gz/launch/ca1_gazebo.launch.py"
        source = launch_path.read_text(encoding="utf-8")
        self.assertIn("/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan", source)
        self.assertIn("/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry", source)
        self.assertIn("/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist", source)

    def test_rl_environment_uses_continuous_holonomic_action_space(self):
        environment_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/ros_gym_env.py"
        source = environment_path.read_text(encoding="utf-8")
        self.assertIn("shape=(3,)", source)
        self.assertIn("max_x_velocity", source)
        self.assertIn("max_y_velocity", source)
        self.assertIn("max_turn_velocity", source)
        self.assertNotIn("spaces.Discrete", source)

    def test_parallel_workers_isolate_ros_and_gazebo(self):
        worker_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/managed_env.py"
        source = worker_path.read_text(encoding="utf-8")
        self.assertIn('os.environ["ROS_DOMAIN_ID"]', source)
        self.assertIn('os.environ["GZ_PARTITION"]', source)
        self.assertIn('"randomize_world:=false"', source)
        self.assertIn("world_path:=", source)
        self.assertIn("worker_config.json", source)
        self.assertIn("cleanup_managed_workers", source)
        self.assertIn('"launch_pid": process.pid', source)
        self.assertIn("wait_for_service", source)
        self.assertIn("start_new_session=True", source)
        self.assertIn("timeout_seconds=30.0", source)
        self.assertIn("retry_attempts=3", source)
        self.assertIn('self._restart("reset_failure"', source)
        self.assertIn('self._restart("step_failure"', source)
        self.assertIn('self._restart("periodic_recycle")', source)
        self.assertIn("restart_events.jsonl", source)

    def test_parallel_training_chooses_fresh_ros_domains_by_default(self):
        runtime_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/runtime_utils.py"
        source = runtime_path.read_text(encoding="utf-8")
        self.assertIn("select_base_ros_domain_id", source)
        self.assertIn("secrets.randbelow", source)
        training_source = (
            PROJECT / "src/rb2301_ca1/rb2301_ca1/train_rl.py"
        ).read_text(encoding="utf-8")
        self.assertIn('default=None', training_source)

    def test_training_supports_resume_and_emergency_save(self):
        training_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/train_rl.py"
        source = training_path.read_text(encoding="utf-8")
        self.assertIn('"--resume"', source)
        self.assertIn("reset_num_timesteps=not resumed", source)
        self.assertIn('_save_recovery_artifacts(model, run_dir, "emergency")', source)
        self.assertIn("emergency_error.txt", source)

    def test_imitation_pipeline_is_installed(self):
        setup_path = PROJECT / "src/rb2301_ca1/setup.py"
        setup_source = setup_path.read_text(encoding="utf-8")
        self.assertIn("collect_demonstrations", setup_source)
        self.assertIn("train_imitation", setup_source)
        self.assertTrue((PROJECT / "run_imitation_pipeline.sh").is_file())

    def test_collection_supports_parallel_exact_expert_execution(self):
        collection_path = (
            PROJECT / "src/rb2301_ca1/rb2301_ca1/collect_demonstrations.py"
        )
        collection_source = collection_path.read_text(encoding="utf-8")
        self.assertIn('"--num-envs"', collection_source)
        self.assertIn("ProcessPoolExecutor", collection_source)
        self.assertIn("environment.step_direct(action)", collection_source)
        self.assertIn("collection_logs", collection_source)
        self.assertIn("shard_checkpoint_episodes", collection_source)
        environment_source = (
            PROJECT / "src/rb2301_ca1/rb2301_ca1/ros_gym_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def step_direct(", environment_source)
        self.assertIn('action_mode = "direct_expert"', environment_source)

    def test_ca1_controller_and_collector_share_one_expert_policy(self):
        controller_source = (
            PROJECT / "src/rb2301_ca1/rb2301_ca1/obstacle_avoidance.py"
        ).read_text(encoding="utf-8")
        collection_source = (
            PROJECT / "src/rb2301_ca1/rb2301_ca1/collect_demonstrations.py"
        ).read_text(encoding="utf-8")
        for source in (controller_source, collection_source):
            self.assertIn("from .expert_policy import ConeExpertPolicy", source)
            self.assertIn("ConeExpertPolicy()", source)

    def test_automatic_imitation_pipeline_uses_parallel_collectors(self):
        source = (PROJECT / "run_imitation_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('collection_envs="${COLLECTION_ENVS:-4}"', source)
        self.assertIn('--num-envs "$collection_envs"', source)
        self.assertIn("index * 100000", source)

    def test_lidar_and_environment_use_twenty_hz_control(self):
        robot_path = PROJECT / "src/rb2301_gz/urdf/nanocar_description.urdf"
        robot_source = robot_path.read_text(encoding="utf-8")
        self.assertIn("<update_rate>20</update_rate>", robot_source)
        environment_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/ros_gym_env.py"
        environment_source = environment_path.read_text(encoding="utf-8")
        self.assertIn("scans_per_action: int = 1", environment_source)

    def test_evaluation_manages_a_headless_simulator_by_default(self):
        evaluation_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/evaluate_rl.py"
        source = evaluation_path.read_text(encoding="utf-8")
        self.assertIn('"--num-envs"', source)
        self.assertIn("ProcessPoolExecutor", source)
        self.assertIn('get_context("spawn")', source)
        self.assertIn("success_rate_wilson_95", source)
        self.assertIn("evaluation_summary.json", source)
        self.assertIn('"--external-sim"', source)
        self.assertIn("make_managed_env", source)
        self.assertIn("cleanup_managed_workers", source)
        self.assertIn("Managed evaluation logs:", source)

    def test_ppo_has_conservative_teacher_anchor_and_controls(self):
        training_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/train_rl.py"
        source = training_path.read_text(encoding="utf-8")
        self.assertIn("TeacherAnchoredPPO", source)
        for flag in (
            '"--teacher-model"',
            '"--teacher-anchor-strength"',
            '"--ppo-clip-range"',
            '"--ppo-n-epochs"',
            '"--ppo-target-kl"',
        ):
            self.assertIn(flag, source)
        anchor_path = PROJECT / "src/rb2301_ca1/rb2301_ca1/anchored_ppo.py"
        anchor_source = anchor_path.read_text(encoding="utf-8")
        self.assertIn("class TeacherAnchoredPPO", anchor_source)
        self.assertIn("teacher_anchor_l2", anchor_source)
        self.assertIn("_excluded_save_params", anchor_source)

    def test_setup_pins_ros_compatible_system_python(self):
        setup_path = PROJECT / "setup_rl.sh"
        source = setup_path.read_text(encoding="utf-8")
        self.assertIn("/usr/bin/python3.12", source)
        self.assertIn('$existing_version" != "3.12"', source)
        self.assertIn("import rclpy", source)

    def test_rl_requirements_respect_colcon_setuptools_constraint(self):
        requirements_path = PROJECT / "requirements-rl.txt"
        source = requirements_path.read_text(encoding="utf-8")
        self.assertIn("setuptools>=68,<80", source)

    def test_deployment_pipeline_is_installed(self):
        setup_source = (
            PROJECT / "src/rb2301_ca1/setup.py"
        ).read_text(encoding="utf-8")
        self.assertIn("deploy_policy", setup_source)
        self.assertIn("export_policy", setup_source)
        for script in ("setup_deploy.sh", "deploy_policy.sh", "export_policy.sh"):
            path = PROJECT / script
            self.assertTrue(path.is_file(), script)
            self.assertTrue(path.stat().st_mode & 0o111, f"{script} is not executable")

    def test_deployment_installs_the_exported_policy(self):
        """The .npz is the deployable artefact, so colcon must install it.

        ``setup.py`` has no ``package_data``, and an empty glob installs nothing
        silently -- the failure would only surface on the robot, at run time.
        """

        setup_source = (
            PROJECT / "src/rb2301_ca1/setup.py"
        ).read_text(encoding="utf-8")
        self.assertIn("glob.glob('policy/*.npz')", setup_source)
        self.assertIn("policy_artifacts", setup_source)

    def test_deployment_dependencies_exclude_the_training_stack(self):
        """The point of the deploy path is that the robot installs no wheels."""

        source = (
            PROJECT / "requirements-deploy.txt"
        ).read_text(encoding="utf-8")
        required = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        self.assertIn("numpy", required)
        for package in ("torch", "stable-baselines3", "gymnasium", "tensorboard"):
            self.assertNotIn(package, required)

    def test_deployment_setup_creates_no_environment(self):
        """It may *mention* .venv and requirements-rl.txt, but never use them."""

        source = (PROJECT / "setup_deploy.sh").read_text(encoding="utf-8")
        self.assertNotIn("python3 -m venv", source)
        self.assertNotIn("pip install", source)
        self.assertNotIn("-r requirements-rl.txt", source)
        self.assertIn("colcon build --symlink-install", source)

    def test_deployment_runtime_avoids_the_training_stack(self):
        for module in ("deploy_policy", "policy_runtime"):
            source = (
                PROJECT / f"src/rb2301_ca1/rb2301_ca1/{module}.py"
            ).read_text(encoding="utf-8")
            with self.subTest(module=module):
                self.assertNotIn("stable_baselines3", source)
                self.assertNotIn("import torch", source)
        exporter = (
            PROJECT / "src/rb2301_ca1/rb2301_ca1/export_policy.py"
        ).read_text(encoding="utf-8")
        # The exporter is the one module allowed to need them, and only off-robot.
        self.assertIn("import torch", exporter)


if __name__ == "__main__":
    unittest.main()
