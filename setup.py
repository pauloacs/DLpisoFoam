from setuptools import setup, find_packages

setup(
    name='surrogate_model',
    version='0.1',
    author = "Paulo Sousa", \
    packages = [ \
        'surrogate_model', \
        'surrogate_model.test_data', \
    ], \
    package_dir={ \
        'surrogate_model': 'surrogate_model', \
        'test_data': 'surrogate_model/test_data', \
    }, \
    install_requires=[
        'numpy',
        'tensorflow'
    ],
)
