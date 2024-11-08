from setuptools import find_packages, setup
from typing import List

"""
    The function `get_requirements` reads a requirements.txt file and returns a list of required
    packages, excluding the '-e .' entry if present.
    
    :param file_path: The `file_path` parameter in the `get_requirements` function should be a string
    representing the path to the requirements.txt file that you want to read. You should provide the
    full file path as a string when calling the function. For example, if your requirements.txt file is
    located in a directory called
    :type file_path: str
    :return: The function `get_requirements` will return a list of required packages read from the
    requirements.txt file specified by the `file_path` parameter. If the variable `-e .` is found in the
    list of requirements, it will be removed before returning the final list.
"""

variable = '-e .'   
def get_requirements(file_path:str)->List[str]:
    '''
    Function to read requirements.txt file and return a list of required packages.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    
        if variable in requirements:
            requirements.remove('-e .')
    
    return requirements


## project metadata
setup(
    name='E2E_ML',
    version='0.0.1',
    author="Sasank Tanikella",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
    python_requires='>=3.9',
    include_package_data=True,
    description='End-to-End Machine Learning Project',
)