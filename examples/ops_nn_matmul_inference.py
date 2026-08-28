"""Small NumPy inference case with an intentional AsNumPy operator gap.

``numpy.matmul`` is the gap under test: the audited AsNumPy 0.3.1 public
wrapper currently copies its operands to the host and delegates to NumPy.
The two input constructors already have strict NPU implementations, which
makes this script useful for demonstrating NumPilot's gap-recovery flow.
"""

import numpy as np


features = np.full((8, 16), 0.01, dtype=np.float32)
weights = np.full((16, 4), 0.02, dtype=np.float32)
result = np.matmul(features, weights)
