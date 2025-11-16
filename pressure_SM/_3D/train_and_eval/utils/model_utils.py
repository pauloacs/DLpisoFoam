"""
Model architecture definitions and training utilities.
"""

import numpy as np


def define_model_arch(model_architecture: str) -> tuple[int, list]:
    """
    Define neural network architecture parameters.
    
    Args:
        model_architecture: Name of the architecture
        
    Returns:
        tuple: (n_layers, width) where width is list of layer widths
    """
    model_architecture = model_architecture.lower()
    match model_architecture:
        case 'mlp_small':
            n_layers = 3
            width = [512]*3
        case 'mlp_big':
            n_layers = 7
            width = [256] + [512]*5 + [256]
        case 'mlp_huge':
            n_layers = 12
            width = [256] + [512]*10 + [256]
        case 'mlp_small_unet':
            width = [512, 512, 256, 256, 128]
            n_layers = len(width)
        case 'conv1d':
            n_layers = 7
            width = [128, 64, 32, 16, 32, 64, 128]
        case 'mlp_attention':
            n_layers = 3
            width = [512]*3
        case 'mlp_medium':
            n_layers = 5
            width = [256, 512, 512, 512, 256]
        case 'gnn':
            n_layers = 6
            width = [128, 256, 256, 256, 128, 64]
        case 'fno3d':
            n_layers = 4
            width = [64, 128, 128, 64]
        case 'mixer':
            n_layers=3
            width = [512]*3
        case 'cnn':
            n_layers = None
            width = None
        case 'multi_layer_3d':
            n_layers = 5
            width = 128
        case _:
            raise ValueError('Invalid NN model type')

    return n_layers, width


def Callback_EarlyStopping(LossList, min_delta=0.1, patience=20):
    """
    Early stopping callback for training.
    
    Args:
        LossList: List of loss values
        min_delta: Minimum change threshold
        patience: Number of epochs to wait
        
    Returns:
        bool: True if training should stop
    """
    # No early stopping for 2*patience epochs
    if len(LossList)//patience < 2:
        return False
    # Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience])  # second-last
    mean_recent = np.mean(LossList[::-1][:patience])  # last
    # you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous)  # abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta:
        print("*CB_ES* Loss didn't change much from last %d epochs" % (patience))
        print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False
