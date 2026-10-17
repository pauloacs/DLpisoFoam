# Thin wrapper — all logic lives in the shared package.
import os, sys
os.environ['TRAIN_SCRIPT_MODE'] = '1'
sys.path.insert(0, os.getcwd())
from python_module import train

if not train:
    print('[train_update] train=False in python_module.py — skipping model update.')
    sys.exit(0)

from python_module import __dict__ as case_config

backend = case_config.get('training_backend', 'lattice')
if backend == 'mesh_graph':
    raise NotImplementedError('Mesh-graph prototype supports initial training only; disable incremental updates.')
if backend != 'lattice':
    raise ValueError(f'Unknown training_backend: {backend}')
from pressure_SM_delta_delta_shift._3D.auto_CFD.train_update import add_new_features_and_train

if __name__ == '__main__':
    add_new_features_and_train()
