"""J011_datetime_arithmetic: Calendar and duration dtypes.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'dates': np.array(['2024-02-28', '2024-02-29', '2024-03-01'], dtype='datetime64[D]'),
        'step': np.array([1, 2, 3], dtype='timedelta64[D]'),
    }


def forward(dates, step):
    return {
        'dates': np.add(dates, step),
        'elapsed': np.subtract(dates, dates[0]),
    }
