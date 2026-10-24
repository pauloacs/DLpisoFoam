"""
Pressure surrogate model with shifter formulation.

The shifter predicts 4 channels [ux, uy, uz, s] that reconstruct ddP as:
  ddP_pred = -ux * ∂x(dP_prev) - uy * ∂y(dP_prev) - uz * ∂z(dP_prev) + s

This allows the model to predict local field displacement rather than a free residual.
"""

__version__ = "1.0-shifter"
