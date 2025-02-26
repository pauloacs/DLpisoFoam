from setuptools import setup, find_packages

setup(
    name='pressureSM',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train_script = pressureSM.train_and_eval.entry_point:train_entry_point',
            'evaluation_script = pressureSM.train_and_eval.entry_point:eval_entry_point'
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow'
    ],
)
