"""C001_heat_diffusion: Periodic heat flow with diagnostic energy.

Inputs are constructed on the CPU; forward returns named device arrays.
Four updates reuse one frozen Laplacian; this is a fixed-forcing stage.
Source: ../../../../examples/heat_diffusion_forward.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    axis = np.linspace(-1, 1, 16, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    temperature = np.exp(-8 * (xx * xx + yy * yy))
    return {'u': temperature, 'north': np.roll(temperature, 1, axis=0), 'south': np.roll(temperature, -1, axis=0), 'west': np.roll(temperature, 1, axis=1), 'east': np.roll(temperature, -1, axis=1)}


def forward(u, north, south, west, east):
    if isinstance(u, np.ndarray):
        u = ap.ndarray.from_numpy(u)
    if isinstance(north, np.ndarray):
        north = ap.ndarray.from_numpy(north)
    if isinstance(south, np.ndarray):
        south = ap.ndarray.from_numpy(south)
    if isinstance(west, np.ndarray):
        west = ap.ndarray.from_numpy(west)
    if isinstance(east, np.ndarray):
        east = ap.ndarray.from_numpy(east)

    __numpilot_v_1 = ap.add(north, south)
    __numpilot_v_2 = ap.add(west, east)
    __numpilot_v_3 = ap.add(__numpilot_v_1, __numpilot_v_2)
    __numpilot_v_4 = ap.multiply(u, ap.full(u.shape, 4.0, dtype=u.dtype))
    __numpilot_v_5 = ap.subtract(__numpilot_v_3, __numpilot_v_4)
    __numpilot_v_6 = ap.multiply(__numpilot_v_5, ap.full(__numpilot_v_5.shape, 0.04, dtype=__numpilot_v_5.dtype))
    __numpilot_v_7 = ap.add(u, __numpilot_v_6)
    __numpilot_v_8 = ap.maximum(__numpilot_v_7, ap.full(__numpilot_v_7.shape, 0.0, dtype=__numpilot_v_7.dtype))
    __numpilot_v_9 = ap.multiply(__numpilot_v_5, ap.full(__numpilot_v_5.shape, 0.04, dtype=__numpilot_v_5.dtype))
    __numpilot_v_10 = ap.add(__numpilot_v_8, __numpilot_v_9)
    __numpilot_v_11 = ap.maximum(__numpilot_v_10, ap.full(__numpilot_v_10.shape, 0.0, dtype=__numpilot_v_10.dtype))
    __numpilot_v_12 = ap.multiply(__numpilot_v_5, ap.full(__numpilot_v_5.shape, 0.02, dtype=__numpilot_v_5.dtype))
    __numpilot_v_13 = ap.add(__numpilot_v_11, __numpilot_v_12)
    __numpilot_v_14 = ap.maximum(__numpilot_v_13, ap.full(__numpilot_v_13.shape, 0.0, dtype=__numpilot_v_13.dtype))
    __numpilot_v_15 = ap.multiply(__numpilot_v_5, ap.full(__numpilot_v_5.shape, 0.02, dtype=__numpilot_v_5.dtype))
    __numpilot_v_16 = ap.add(__numpilot_v_14, __numpilot_v_15)
    __numpilot_v_17 = ap.maximum(__numpilot_v_16, ap.full(__numpilot_v_16.shape, 0.0, dtype=__numpilot_v_16.dtype))
    __numpilot_v_18 = ap.square(u)
    __numpilot_v_19 = ap.sum(__numpilot_v_18, axis=1)
    __numpilot_v_20 = ap.square(__numpilot_v_17)
    __numpilot_v_21 = ap.sum(__numpilot_v_20, axis=1)
    __numpilot_v_22 = ap.subtract(__numpilot_v_17, u)
    __numpilot_v_23 = ap.subtract(__numpilot_v_21, __numpilot_v_19)
    return {'out': __numpilot_v_17, 'change': __numpilot_v_22, 'energy_delta': __numpilot_v_23}