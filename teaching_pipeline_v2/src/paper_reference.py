"""Published numbers from the source study, for comparison with our runs.

    Subedi, S. K., Cherry, S. R., Qiang, Y., & Peng, P. (2025).
    Feasibility study of multiplexing analog signals from SiPMs for a single layer
    monolithic PET detector design.
    Radiation Measurements 182, 107399.  https://doi.org/10.1016/j.radmeas.2025.107399

PROVENANCE OF THESE NUMBERS -- read before quoting them:

* TABLE2_OPTIMAL is transcribed from **Table 2 of the SSRN preprint 4865116**. That table
  (average FWHM + standard deviation for every multiplexing scheme) does not appear in the
  main text of the published article; the published article may carry it as supplementary
  data, which has not been checked. Cite it as the preprint's Table 2.
* TABLE1_CORNER_MSE is Table 1, which appears in BOTH versions with identical values
  (verified value-by-value against the published article). The published Table 1 adds a
  second half using a fixed 5x5 corner grid for every channel count; only the adaptive-grid
  half is reproduced here.
"""
TABLE2_OPTIMAL = {
    32: (0.50, 0.24, 0.50, 0.24),   # no multiplexing
    28: (0.51, 0.23, 0.52, 0.23),
    24: (0.50, 0.21, 0.50, 0.21),
    20: (0.51, 0.21, 0.51, 0.21),
    16: (0.51, 0.19, 0.51, 0.20),   # Config 16b
    12: (0.52, 0.18, 0.53, 0.18),   # Config 12a
    8:  (0.58, 0.17, 0.58, 0.17),   # Config 8b
    4:  (0.69, 0.25, 0.69, 0.24),   # Config 4b
}

# Table 1: mean square error at the detector corners [mm^2?]; the corner grid size differs per row, so it
# is NOT directly comparable with the RMS error computed over the whole test grid in this code.
TABLE1_CORNER_MSE = {   # channels: (MSE_Y, MSE_X, corner grid)
    32: (0.40, 0.38, "3x3"), 28: (0.41, 0.39, "3x3"), 24: (0.61, 0.60, "4x4"), 20: (0.56, 0.53, "4x4"),
    16: (0.48, 0.49, "4x4"), 12: (1.18, 1.09, "5x5"), 8: (0.99, 0.97, "5x5"), 4: (4.33, 4.51, "5x5"),
}
