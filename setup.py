from setuptools import setup, find_packages

classic_packages = find_packages(
    where='classical/packages',
    include=('pressure_SM', 'pressure_SM.*', 'pressure_SM_delta_delta', 'pressure_SM_delta_delta.*'),
)
auto_packages = find_packages(
    where='auto/packages',
    include=('pressure_SM_gridded_shift', 'pressure_SM_gridded_shift.*',
             'pressure_SM_mesh_graph', 'pressure_SM_mesh_graph.*'),
)

setup(
    name='pressure_SM',
    version='1.0.0',
    packages=classic_packages + auto_packages,
    package_dir={
        'pressure_SM': 'classical/packages/pressure_SM',
        'pressure_SM_delta_delta': 'classical/packages/pressure_SM_delta_delta',
        'pressure_SM_gridded_shift': 'auto/packages/pressure_SM_gridded_shift',
        'pressure_SM_mesh_graph': 'auto/packages/pressure_SM_mesh_graph',
    },
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
            'train_3d_mesh_graph = pressure_SM_mesh_graph._3D.training:main',
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow'
    ],
    extras_require={'mesh_graph': ['foamlib>=1.7.8,<2', 'scipy', 'numpy<2', 'matplotlib']},
)
