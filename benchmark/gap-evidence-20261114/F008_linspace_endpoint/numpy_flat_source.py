import numpy as np

offset = np.full((1048576,), 0.5, dtype=np.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = np.linspace(0.0, 0.9999990463256836, 1048576, dtype=np.float32)
__numpilot_v_2 = np.add(__numpilot_v_1, offset)

out = __numpilot_v_2
