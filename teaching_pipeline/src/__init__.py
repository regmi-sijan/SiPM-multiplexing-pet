"""Teaching re-implementation of:

    Subedi, S. K., Cherry, S. R., Qiang, Y., & Peng, P. (2025).
    Feasibility study of multiplexing analog signals from SiPMs for a single layer
    monolithic PET detector design.
    Radiation Measurements 182, 107399.  https://doi.org/10.1016/j.radmeas.2025.107399

Written from the earlier preprint (SSRN 4865116, https://ssrn.com/abstract=4865116);
every section/figure/table number in this package refers to the PREPRINT, not to the
published article. See the repository README for the numbering map.

Independent educational re-implementation. Not affiliated with, endorsed by, or verified
by the authors. Please cite the published article above, not this code alone.
"""
import os
# macOS workaround: PyTorch and NumPy/SciPy can each bundle their own libomp, which makes OpenMP abort
# ("OMP: Error #15"). Must be set BEFORE numpy/torch are imported. Harmless on other systems.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
