"""C008_portfolio_repricing: Smooth CDF portfolio scenario repricing.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Retains the source example's smooth-CDF approximation, not an exact pricer.
Source: ../../../../examples/black_scholes_portfolio.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def build_inputs():
    return {
        'spot': np.linspace(80, 160, 64, dtype=np.float32),
        'strike': np.linspace(50, 68, 64, dtype=np.float32),
        'expiry': np.linspace(0.1, 2, 64, dtype=np.float32),
        'volatility': np.linspace(0.12, 0.45, 64, dtype=np.float32),
        'rate': np.full(64, 0.03, dtype=np.float32),
    }


def normal_cdf(x):
    scaled = np.multiply(x, 0.79788456)
    smooth = np.tanh(scaled)
    return np.multiply(np.add(1.0, smooth), 0.5)


def d_terms(spot, strike, expiry, volatility):
    sigma_t = np.multiply(volatility, np.sqrt(expiry))
    log_ratio = np.log(np.divide(spot, strike))
    carry = np.multiply(0.5, np.multiply(np.square(volatility), expiry))
    d1 = np.divide(np.add(log_ratio, carry), sigma_t)
    return d1, np.subtract(d1, sigma_t)


def price(spot, strike, expiry, volatility, rate):
    d1, d2 = d_terms(spot, strike, expiry, volatility)
    discount = np.exp(np.negative(np.multiply(rate, expiry)))
    asset = np.multiply(spot, normal_cdf(d1))
    liability = np.multiply(np.multiply(strike, discount), normal_cdf(d2))
    return np.maximum(np.subtract(asset, liability), 0.0)


def forward(spot, strike, expiry, volatility, rate):
    base = price(spot, strike, expiry, volatility, rate)
    scenario = base
    for index in range(3):
        if index == 0:
            shock = 0.9
        else:
            shock = 1.1
        shocked = np.multiply(spot, shock)
        scenario = np.add(scenario, price(shocked, strike, expiry, volatility, rate))
    average = np.divide(scenario, 4.0)
    return {
        'base': base,
        'scenario_mean': average,
        'difference': np.subtract(average, base),
    }
