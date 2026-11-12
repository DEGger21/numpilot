# Extract from gpu4pyscf commit 5fde981316127cd3506d9d3c7d5633647fcc4508
# gpu4pyscf/gpu4pyscf/dft/numint.py:_dot_ao_dm (lines 2381-2382)
def _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    return cupy.dot(dm.T, ao)
