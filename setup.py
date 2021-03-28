from setuptools import setup

setup(
        name='multivision',
        version='0.0.1',
        description='Tools for personal project in multiview computer vision',
        py_modules=["realAPI", "robotics", "simAPI", "sli"],
        package_dir={'':'src'},
        install_requires=['matplotlib']
    )
