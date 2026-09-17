from setuptools import find_packages, setup

package_name = 'rb2301_ca1'

setup(
    name=package_name,
    version='1.1.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
