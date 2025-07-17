from setuptools import setup, find_packages
import subprocess

# Build message types before installing arktypes
subprocess.check_call(["make"])

# Load requirements
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="arktypes",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
)
