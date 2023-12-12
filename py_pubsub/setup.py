from setuptools import setup

package_name = 'py_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zack',
    maintainer_email='zack@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'yolo_detect = py_pubsub.publisher_member_function:main',
                'strategy_sub = py_pubsub.strategy_sub:main',
                'cam_calibration = py_pubsub.cam_cali:main',
                'strategy_client = py_pubsub.strategy_client:main',
                'yolo_service = py_pubsub.yolo_service:main',
                'test_yolo_service = py_pubsub.test_yoloservice_client:main',
                'verify_calibration = py_pubsub.verify_calibration:main',
        ],
    },
)
