from setuptools import setup, find_packages, find_namespace_packages

setup(
    author="Zorro",
    description="This is a utility, written by Zorro",
    name="mlops",
    version="1.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": 'src'},
    include_package_data=True,
)
