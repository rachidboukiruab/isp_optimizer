from setuptools import find_packages, setup

package_name = 'isp_optimizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rachid',
    maintainer_email='rachid@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "data_loader = isp_optimizer.data_loader_node:main",
            "object_detector = isp_optimizer.object_detector_node:main",
            "cv_metrics = isp_optimizer.cv_metrics_node:main",
            "cma_es_optimizer = isp_optimizer.cma_es_optimizer_node:main"
        ],
    },
)
