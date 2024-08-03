from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str)->List[str]:
    """
    Returns a list of requirements.

    Args:
    file_path (str): path to requirements file

    Returns:
    List[str]: list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace('\n', '') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')


setup(
    name='audio_classification',
    version='0.0.0',
    author='saraimdad',
    author_email='saraimdad12@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)