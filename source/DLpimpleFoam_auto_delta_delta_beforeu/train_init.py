# Thin wrapper — all logic lives in the shared package.
import os, sys
os.environ['TRAIN_SCRIPT_MODE'] = '1'
sys.path.insert(0, os.getcwd())
from python_module import train

if not train:
    print('[train_init] train=False in python_module.py — skipping initial training.')
    sys.exit(0)

from pressure_SM_delta_delta_shift._3D.auto_CFD.train_init import main

if __name__ == "__main__":
    main()
