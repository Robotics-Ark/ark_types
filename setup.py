from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

class CustomInstallCommand(install):
    """Custom handler for the install command to run make before installation."""

    def run(self):
        # Run the make command
        subprocess.check_call(["make"])

        # Continue with the standard installation
        super().run()

class CustomBuildPyCommand(build_py):
    def run(self):
        subprocess.check_call(["make"])
        super().run()

class CustomDevelopCommand(develop):
    def run(self):
        subprocess.check_call(["make"])
        super().run()

setup(
    name="arktypes",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "lcm"
    ],
    cmdclass={
        'install': CustomInstallCommand,
        'build_py': CustomBuildPyCommand,
        'develop': CustomDevelopCommand,
    },
)
