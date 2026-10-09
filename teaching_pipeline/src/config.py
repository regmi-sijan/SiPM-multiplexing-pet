"""Configuration and presets for the teaching version of the PET multiplexing study.

Everything that is taken from the paper is marked [paper]; everything that is an
assumption or a simplification made for this teaching version is marked [assumed].
"""
from dataclasses import dataclass, field, replace


@dataclass
class Config:
    # ---- detector geometry (paper, Section 2.1) -------------------------------
    crystal_xy_mm: float = 40.0          # [paper] 40 x 40 mm LSO plate
    crystal_z_mm: float = 4.0            # [paper] 4 mm thick
    n_sipm_per_side: int = 8             # [paper] 8 SiPMs on each of 4 sides = 32
    sipm_size_mm: float = 4.0            # [paper] 4 x 4 mm SiPM
    sipm_pitch_mm: float = 4.6           # [paper] array pitch 4.6 mm

    # ---- optics -----------------------------------------------------------------
    esr_reflectance: float = 0.98        # [paper] crystal/ESR boundary, reflectance 0.98 (top, bottom)
    sipm_pde: float = 0.5                # [paper] crystal/SiPM boundary efficiency 0.5
    side_gap_reflectivity: float = 0.0   # [assumed] bare crystal side between SiPMs absorbs photons
    light_yield_per_kev: float = 30.0    # [assumed] typical LSO value, ~15,300 photons at 511 keV
    gamma_energy_kev: float = 511.0      # [paper] 511 keV
    mu_per_mm: float = 0.088             # [assumed] LSO attenuation at 511 keV (~0.88 /cm)

    # ---- sampling grids (paper, Section 2.2) ---------------------------------------
    train_pitch_mm: float = 1.0          # [paper] 40 x 40 grid, 1 mm pitch -> 1600 classes
    events_per_train_pos: int = 100      # [assumed] paper: ~550-950 events per position
    events_per_test_pos: int = 100       # [assumed]
    # test grid = training grid shifted by half a pitch -> 39 x 39 = 1521 points  [paper]

    # ---- CNN (paper, Section 2.2 and Fig. 3) ------------------------------------------
    n_filters: int = 200                 # [paper] 200 filters
    kernel_size: int = 3                 # [assumed] paper says filter size 1-3 was explored
    epochs: int = 20                     # [assumed] paper explored 20-30 epochs
    batch_size: int = 128                # [assumed]
    lr: float = 0.01                     # [paper] initial LR 0.01 was among tested values (sgdm)
    momentum: float = 0.9                # [assumed] MATLAB sgdm default
    val_fraction: float = 0.10           # [paper] 10 % validation split

    # ---- evaluation --------------------------------------------------------------------
    fit: str = "auto"                    # 'gauss' (histogram fit as in paper), 'robust' (median/MAD), 'auto'
    hist_window_mm: float = 3.0

    # ---- run control -------------------------------------------------------------------
    channels: tuple = (32, 28, 24, 20, 16, 12, 8, 4)
    seed: int = 1
    out_dir: str = "results"

    @property
    def n_sipm(self):
        return 4 * self.n_sipm_per_side

    @property
    def n_photons_mean(self):
        return self.light_yield_per_kev * self.gamma_energy_kev


_BASE = Config()

PRESETS = {
    # Runs in a few minutes on a laptop CPU. Coarser grid, smaller network.
    "quick": replace(_BASE, train_pitch_mm=2.0, events_per_train_pos=40,
                     events_per_test_pos=60, n_filters=32, epochs=15,
                     channels=(32, 16, 8, 4)),
    # Medium: all channel counts, 2 mm grid. Tens of minutes on CPU.
    "medium": replace(_BASE, train_pitch_mm=2.0, events_per_train_pos=100,
                      events_per_test_pos=100, n_filters=64, epochs=20),
    # Paper-like settings (1 mm grid, 1600 classes, 200 filters). Needs a GPU or hours of CPU.
    "paper": replace(_BASE, events_per_train_pos=300, events_per_test_pos=200),
}


def get_config(name: str) -> Config:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return PRESETS[name]
