
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class CustomInstallCommand(install):
    """Custom handler for the install command to run make before installation."""

    def run(self):
        # Run the make command
        subprocess.check_call(["make"])
        
        # Continue with the standard installation
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
    },
)
