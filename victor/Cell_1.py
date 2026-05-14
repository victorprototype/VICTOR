# ============================================================
# VICTOR v6.0 — Cell 1: Install + Imports + Config + Module Loader
# ============================================================

# ── 1. JAX install (pinned, version-safe) ───────────────────
import subprocess, sys

def pip(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

# Uninstall any mismatched JAX packages from previous sessions
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y",
                       "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt"],
                      stderr=subprocess.DEVNULL)

# Reinstall everything pinned to matching versions
JAX_VERSION = "0.4.30"
for p in [
    f"jax[cuda12]=={JAX_VERSION}",
    f"jaxlib=={JAX_VERSION}",
    "flax",
    "optax",
    "scipy",
    "matplotlib",
    "ott-jax",
    "orbax-checkpoint",
]:
    pip(p)

print(f"JAX {JAX_VERSION} + dependencies installed OK")

# ── 2. Mount Drive ──────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = "/content/drive/MyDrive"
    print("Drive mounted")
except Exception:
    DRIVE = "."
    print("Local mode (Drive not available)")

# ── 3. Core imports ─────────────────────────────────────────
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

# ── 4. XLA compilation cache (persists across sessions) ─────
JAX_CACHE = "/content/jax_cache"
os.environ["JAX_COMPILATION_CACHE_DIR"] = JAX_CACHE
os.makedirs(JAX_CACHE, exist_ok=True)
print(f"JIT cache: {JAX_CACHE}")

# ── 5. Project paths ────────────────────────────────────────
DATASET_DIR  = f"{DRIVE}/victor_dataset"
CKPT_DIR     = f"{DRIVE}/VICTOR_v6"
RESULTS_DIR  = "./ono_results"

for d in [CKPT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Dataset  : {DATASET_DIR}")
print(f"Checkpts : {CKPT_DIR}")



# ── 6. Import VICTOR package modules ────────────────────────
# Call load_modules() at the top of any cell to get fresh imports.
# Autoreload handles this automatically during development,
# but load_modules() is useful after a Drive edit mid-session.

from victor import config as cfg
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