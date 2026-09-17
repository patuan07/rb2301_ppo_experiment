import glob

from setuptools import find_packages, setup

package_name = 'rb2301_ca1'

# The exported actor is the artefact the robot actually runs, so it ships with
# the package and is found through the package share directory.
#
# The glob stays relative on purpose: colcon asserts that data_files sources are
# relative to the package directory, so an os.path.dirname(__file__)-based path
# aborts the build.  A glob that matches nothing installs nothing *silently* and
# only fails later on the robot, hence the check.  Only *.npz is matched -- the
# Stable-Baselines3 checkpoint in the repository's policy/ directory is a
# training artefact and must never reach a storage-limited robot.
policy_artifacts = sorted(glob.glob('policy/*.npz'))
if not policy_artifacts:
    raise SystemExit(
        "No exported policy found at policy/*.npz.  Generate it on the training "
        "machine with:\n"
        "    python -m rb2301_ca1.export_policy policy/best_model_ppo.zip\n"
        "then build again."
    )

setup(
    name=package_name,
    version='1.1.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/policy', policy_artifacts),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marmot',
    maintainer_email='marmot@todo.todo',
    description='Continuous holonomic LiDAR navigation with ROS 2 and Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'obstacle_avoidance = rb2301_ca1.obstacle_avoidance:main',
            'generate_obstacles = rb2301_ca1.obstacle_generator:main',
            'train_rl = rb2301_ca1.train_rl:main',
            'evaluate_rl = rb2301_ca1.evaluate_rl:main',
            'collect_demonstrations = rb2301_ca1.collect_demonstrations:main',
            'summarize_collection = rb2301_ca1.summarize_collection:main',
            'train_imitation = rb2301_ca1.train_imitation:main',
            'rl_smoke_test = rb2301_ca1.smoke_test:main',
            'deploy_policy = rb2301_ca1.deploy_policy:main',
            'export_policy = rb2301_ca1.export_policy:main',
        ],
    },
)
