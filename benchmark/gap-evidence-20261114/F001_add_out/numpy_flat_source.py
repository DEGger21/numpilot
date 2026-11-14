import numpy as np

x = np.full((1024, 1024), 0.5, dtype=np.float32)
y = np.full((1024, 1024), 0.5, dtype=np.float32)
buffer = np.full((1024, 1024), 0.5, dtype=np.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = np.add(x, y)
__numpilot_v_2 = np.square(__numpilot_v_1)

out = __numpilot_v_1
norm = __numpilot_v_2
