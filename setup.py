from setuptools import setup, find_packages

setup(
    name='pressure_SM',
    version='0.3.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train_2d = pressure_SM._2D.train_and_eval.entry_point:train_entry_point',
            'eval_2d = pressure_SM._2D.train_and_eval.entry_point:eval_entry_point',
            'train_3d = pressure_SM._3D.train_and_eval.entry_point:train_entry_point',
            'eval_3d = pressure_SM._3D.train_and_eval.entry_point:eval_entry_point',
            'train_3d_PCA = pressure_SM._3D.train_and_eval_3d_PCA.entry_point:train_entry_point',
            'eval_3d_PCA = pressure_SM._3D.train_and_eval_3d_PCA.entry_point:eval_entry_point',
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow'
    ],
)
