# Thin wrapper — all logic lives in the shared package.
import os, sys
os.environ['TRAIN_SCRIPT_MODE'] = '1'
sys.path.insert(0, os.getcwd())
from python_module import train

if not train:
    print('[train_init] train=False in python_module.py — skipping initial training.')
    sys.exit(0)

from python_module import __dict__ as case_config

backend = case_config.get('training_backend', 'lattice')
if backend == 'mesh_graph':
    from pressure_SM_mesh_graph._3D.training import main
elif backend == 'lattice':
    from pressure_SM_delta_delta_shift._3D.auto_CFD.train_init import main
else:
    raise ValueError(f'Unknown training_backend: {backend}')

if __name__ == "__main__":
    main()
