from setuptools import find_packages, setup

setup(
    name="ur7e_state",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/ur7e_state"]),
        ("share/ur7e_state", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "joint_state_node = ur7e_state.joint_state_node:main",
    ]},
)
