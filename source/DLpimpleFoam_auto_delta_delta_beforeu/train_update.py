# Thin wrapper — all logic lives in the shared package.
import os, sys
os.environ['TRAIN_SCRIPT_MODE'] = '1'
sys.path.insert(0, os.getcwd())
from python_module import train

if not train:
    print('[train_update] train=False in python_module.py — skipping model update.')
    sys.exit(0)

from pressure_SM_delta_delta_shift._3D.auto_CFD.train_update import add_new_features_and_train

if __name__ == '__main__':
    add_new_features_and_train()
