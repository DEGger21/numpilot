"""J004_network_io: Read a numeric vector from a loopback HTTP endpoint.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'url': np.array('http://127.0.0.1:8765/vector.txt'),
    }


def forward(url):
    from urllib.request import urlopen
    with urlopen(url.item(), timeout=2) as response:
        payload = response.read().decode("ascii")
    out = np.fromstring(payload, sep=" ", dtype=np.float64)
    return {
        'out': out,
    }
