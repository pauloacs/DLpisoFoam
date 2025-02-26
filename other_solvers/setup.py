from setuptools import setup, find_packages

setup(
    name='surrogate_models',
    version='0.1',
    author="Paulo Sousa",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow'
    ],
)
