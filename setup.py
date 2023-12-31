from setuptools import find_packages, setup
from typing import List


def get_requirements(filepath: str) -> List[str]:
    """
    Returns the list of required packages
    """

    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', "") for req in requirements]

        if '-e.' in requirements:
            requirements.remove('-e.')

    return requirements


setup(
    name='Movie-collection-prediction',
    version='0.0.1',
    author='Aryan',
    author_email='aryan.10119011621@ipu.ac.in',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
