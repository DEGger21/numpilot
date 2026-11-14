"""C001_heat_diffusion: Periodic heat flow with diagnostic energy.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Four updates reuse one frozen Laplacian; this is a fixed-forcing stage.
Source: ../../../../examples/heat_diffusion_forward.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    axis = np.linspace(-1, 1, 16, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    temperature = np.exp(-8 * (xx * xx + yy * yy))
    return {
        'u': temperature,
        'north': np.roll(temperature, 1, axis=0),
        'south': np.roll(temperature, -1, axis=0),
        'west': np.roll(temperature, 1, axis=1),
        'east': np.roll(temperature, -1, axis=1),
    }


def laplacian(u, north, south, west, east):
    vertical = np.add(north, south)
    horizontal = np.add(west, east)
    neighbours = np.add(vertical, horizontal)
    centre = np.multiply(u, 4.0)
    return np.subtract(neighbours, centre)


def advance(u, lap, rate):
    increment = np.multiply(lap, rate)
    updated = np.add(u, increment)
    return np.maximum(updated, 0.0)


def row_energy(u):
    squared = np.square(u)
    return np.sum(squared, axis=1)


def forward(u, north, south, west, east):
    lap = laplacian(u, north, south, west, east)
    current = u
    for step in range(4):
        if step < 2:
            rate = 0.04
        else:
            rate = 0.02
        current = advance(current, lap, rate)
    before = row_energy(u)
    after = row_energy(current)
    change = np.subtract(current, u)
    return {
        'out': current,
        'change': change,
        'energy_delta': np.subtract(after, before),
    }
