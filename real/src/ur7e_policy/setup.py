from glob import glob

from setuptools import find_packages, setup

setup(
    name="ur7e_policy",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/ur7e_policy"]),
        ("share/ur7e_policy", ["package.xml"]),
        ("share/ur7e_policy/launch", glob("launch/*.launch.py")),
        ("share/ur7e_policy/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "policy_node = ur7e_policy.policy_node:main",
    ]},
)
