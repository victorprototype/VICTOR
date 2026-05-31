# ============================================================
# VICTOR v6.0 — Cell 1: Imports + Config + Runtime Setup
# ============================================================
# Assumes Cell 0 + Cell 0-verify already completed successfully.
# ============================================================


# ── 1. Core imports ─────────────────────────────────────────
import os, json, pickle, time, functools
import numpy as np
import scipy.sparse as sp_sci
import matplotlib.pyplot as plt

import jax, jax.numpy as jnp
from jax.experimental.sparse import BCOO
import flax.linen as nn
import optax
from flax.training import train_state

print(f"JAX     : {jax.__version__}")
print(f"Devices : {jax.devices()}")

# ── 2. XLA compilation cache ────────────────────────────────
JAX_CACHE = "/content/jax_cache"
os.environ["JAX_COMPILATION_CACHE_DIR"] = JAX_CACHE
os.makedirs(JAX_CACHE, exist_ok=True)
print(f"JIT cache: {JAX_CACHE}")

from victor import config as cfg

# ── 3. Project paths ────────────────────────────────────────
DATASET_DIR = cfg.DATASET_DIR
CKPT_DIR    = cfg.CKPT_DIR
RESULTS_DIR = cfg.RESULTS_DIR

for d in [CKPT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Dataset  : {DATASET_DIR}")
print(f"Checkpts : {CKPT_DIR}")



# ── 4. Import VICTOR package modules ────────────────────────


from victor import geometry
from victor import data_loader
from victor import model
from victor import losses
from victor import trainer
from victor import checkpoint
from victor import evaluate
print("\nConfig loaded:")
print(f"  Grid={cfg.N_GRID}x{cfg.N_GRID}")
print(f"  Epochs={cfg.N_EPOCHS}")
print(f"  LR={cfg.LR}")