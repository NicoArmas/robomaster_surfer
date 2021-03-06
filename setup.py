from setuptools import setup
from glob import glob

package_name = 'robomaster_surfer'
vision_package = 'robomaster_surfer.vision'
utils_package = 'robomaster_surfer.vision.utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, vision_package, utils_package],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        ## NOTE: you must add this line to use launch files
        # Instruct colcon to copy launch files during package build 
        ('share/' + package_name + '/launch', glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usi',
    maintainer_email='usi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_switcher = robomaster_surfer.switch_lane:main'
        ],
    },
)
