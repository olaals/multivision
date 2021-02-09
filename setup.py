from setuptools import setup

setup(
        name='tpktools',
        version='0.0.1',
        description='Tools for personal project in robotics',
        py_modules=["realAPI", "robotics", "simAPI", "sli"],
        package_dir={'':'src'}
    )
