from setuptools import setup, find_packages

setup(
    name='pressure_SM',
    version='1.0.0',
    packages=find_packages(include=["pressure_SM*", "pressure_SM_delta_delta*"]),
    entry_points={
        'console_scripts': [
            # pressure_SM entry points
            'train_2d = pressure_SM._2D.train_and_eval.entry_point:train_entry_point',
            'eval_2d = pressure_SM._2D.train_and_eval.entry_point:eval_entry_point',
            'train_3d = pressure_SM._3D.train_and_eval.entry_point:train_entry_point',
            'eval_3d = pressure_SM._3D.train_and_eval.entry_point:eval_entry_point',
            # pressure_SM_delta_delta entry points
            'train_3d_delta_delta = pressure_SM_delta_delta._3D.train_and_eval.entry_point:train_entry_point',
            'eval_3d_delta_delta = pressure_SM_delta_delta._3D.train_and_eval.entry_point:eval_entry_point',
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow'
    ],
)
