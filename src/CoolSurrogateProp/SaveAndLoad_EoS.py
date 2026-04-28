# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 13:07:27 2026

@author: pmarques
"""

#%% Import packages

import h5py
import numpy as np
from jax import nn

#%% Define functions

# Map jax.nn functions to string names
ACT_FN_TO_STR = {
    nn.relu:       "relu",
    nn.silu:       "silu",
    nn.tanh:       "tanh",
    nn.elu:        "elu",
    nn.gelu:       "gelu",
    nn.sigmoid:    "sigmoid",
    nn.leaky_relu: "leaky_relu",
}
# Conversion function
def afns_to_str(fns):
    return [ACT_FN_TO_STR[f] for f in fns]

# Reverse list: map strings to jax.nn functions
STR_TO_ACT_FN = {v: k for k, v in ACT_FN_TO_STR.items()}
# Conversion function
def str_to_afns(names):
    return [STR_TO_ACT_FN[s] for s in names]

#%% Save and load JAX models using h5py

def save_jax_model(path, d):
    with h5py.File(path, "w") as f:
        _save_recursive(f, d)

def _save_recursive(group, d):
    for k, v in d.items():
        if isinstance(v, dict):
            _save_recursive(group.create_group(k), v)
        elif isinstance(v, (list, tuple)):
            # Lists of strings stored as a special dtype
            if all(isinstance(i, str) for i in v):
                group[k] = np.array(v, dtype=h5py.string_dtype())
            else:
                group[k] = np.array(v)
        elif isinstance(v, str):
            group[k] = np.array(v, dtype=h5py.string_dtype())
        else:  # int, float, np/jax array
            group[k] = np.array(v)

def load_jax_model(path):
    with h5py.File(path, "r") as f:
        return _load_recursive(f)

def _load_recursive(group):
    d = {}
    for k, v in group.items():
        if isinstance(v, h5py.Group):
            d[k] = _load_recursive(v)
        elif h5py.check_string_dtype(v.dtype):
            val = v[()]
            # Return list of strings or a single string
            d[k] = [s.decode() for s in val] if val.ndim > 0 else val.decode()
        else:
            d[k] = v[()]   # returns a numpy array or scalar
    return d