"""Official-first example: join two float32 feature blocks with concatenate.

The only call outside the pinned AsNumpy capability catalog is
``np.concatenate``.  The straight-line form keeps the source fully qualified
for the deterministic migration gate; the verification driver supplies the
same shapes and dtype to the NumPy and migrated programs.
"""

import numpy as np


x = np.ones((3, 2), dtype=np.float32)
y = np.zeros((3, 4), dtype=np.float32)
joined = np.concatenate((x, y), axis=1)
