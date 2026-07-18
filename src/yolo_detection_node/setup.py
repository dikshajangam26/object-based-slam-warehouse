from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo_detection_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diksha',
    maintainer_email='dikshajangam26@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_detector = yolo_detection_node.yolo_detector:main',
            'dataset_harvester = yolo_detection_node.dataset_harvester:main',
            'tracking_node = yolo_detection_node.tracking_node:main',
            'reconstruction_node = yolo_detection_node.reconstruction_node:main'
        ],
    },
)
