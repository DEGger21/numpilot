"""NumPilot framework demo: dense scientific feature contraction.

The program is deliberately straight-line so the ordinary-script workflow can
show every stage in the terminal.  It keeps one FP32 data type and uses only
fixed-shape NumPy calls.  Ten matrix/trigonometric stages model a repeated
feature contraction found in scientific grid calculations: a dense projection
is followed by a bounded nonlinear transform and fed into the next projection.
"""
import numpy as np

# Keep the shape explicit: the admission analyzer can bind the FP32 2-D contract.
x = np.add(
    np.ones((1024, 1024), dtype=np.float32),
    np.zeros((1024, 1024), dtype=np.float32),
)
w = np.eye(1024, dtype=np.float32)

# NUMPILOT_DEVICE_INPUTS_END

# An identity projection keeps the repeated contraction numerically stable
# while retaining the full large-matrix workload.  The migration focus is the
# repeated matmul + sin chain; each stage is visible to the six-agent UI.

stage0 = np.matmul(x, w)
stage1 = np.sin(stage0)
stage2 = np.matmul(stage1, w)
stage3 = np.sin(stage2)
stage4 = np.matmul(stage3, w)
stage5 = np.sin(stage4)
stage6 = np.matmul(stage5, w)
stage7 = np.sin(stage6)
stage8 = np.matmul(stage7, w)
result = np.sin(stage8)
