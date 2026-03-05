import copy
from dataclasses import dataclass, asdict
from typing import Union, Tuple


# ============================================================
# Type aliases
# ============================================================

Number2 = Union[float, Tuple[float, float]]
Int2 = Union[int, Tuple[int, int]]

# ============================================================
# Validation helpers
# ============================================================

def _to_len2(value, name):
    if isinstance(value, (int, float)):
        return (value, value)
    return value

def _validate_len2(name: str, value):
    if not (isinstance(value, tuple) and len(value) == 2):
        raise ValueError(f"{name} must be a tuple of length 2")


def _validate_ratio(name: str, value: float):
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_positive(name: str, value: float):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative(name: str, value: float):
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


# ============================================================
# Accelerator configuration
# ============================================================

@dataclass
class AcceleratorConfig:
    """
    Properties of the accelerator.

    Parameters
    ----------
    particles : tuple[float, float]
        Number of particles per bunch [10^10]

    energy : tuple[float, float]
        Beam energy [GeV]

    espread : tuple[float, float]
        Relative energy spread [1]

    emitt_x, emitt_y : tuple[float, float]
        Horizontal/vertical emittance [μm]

    beta_x, beta_y : tuple[float, float]
        Beta functions at IP [mm]

    sigma_x, sigma_y : tuple[float, float]
        RMS beam sizes [nm]

    sigma_z : tuple[float, float]
        RMS beam sizes [μm]
    
    f_rep : float
        Repetition frequency [Hz]

    n_b : int
        Number of bunches [1]

    offset_x, offset_y : tuple[float, float]
        Horizontal/vertical beam offset [nm]
    
    waist_x, waist_y : tuple[float, float]
        Horizontal/vertical waist shift [μm]

    angle_x, angle_y : tuple[float, float]
        Bunch inclination not compensated by crab scheme [rad]
    """
    
    # Beam properties
    particles: Number2
    energy: Number2
    espread: Number2

    emitt_x: Number2
    emitt_y: Number2
    beta_x: Number2
    beta_y: Number2

    which_espread: Int2 = (1, 1)  # 0,1,2,3
    charge_sign: int = -1

    sigma_x: Number2 = (0.0, 0.0)
    sigma_y: Number2 = (0.0, 0.0)
    sigma_z: Number2 = (0.0, 0.0)

    f_rep: float = 0.0
    n_b: int = 0

    dist_z: Int2 = (0, 0)
    trav_focus: int = 0

    offset_x: Number2 = (0.0, 0.0)
    offset_y: Number2 = (0.0, 0.0)

    waist_x: Number2 = (0.0, 0.0)
    waist_y: Number2 = (0.0, 0.0)

    angle_x: Number2 = (0.0, 0.0)
    angle_y: Number2 = (0.0, 0.0)
    angle_phi: Number2 = (0.0, 0.0)

    # --------------------------------------------------------

    def __post_init__(self):

        # Validate tuple length
        len2_fields = [
            "particles", "energy", "espread",
            "which_espread",
            "sigma_x", "sigma_y", "sigma_z",
            "dist_z",
            "emitt_x", "emitt_y",
            "beta_x", "beta_y",
            "angle_x", "angle_y", "angle_phi",
            "offset_x", "offset_y",
            "waist_x", "waist_y",
        ]

        # --- Auto-expand scalars ---
        for name in len2_fields:
            value = getattr(self, name)
            value = _to_len2(value, name)
            setattr(self, name, value)

        # --- Now validate ---
        for name in len2_fields:
            _validate_len2(name, getattr(self, name))

        # which_espread allowed values
        for v in self.which_espread:
            if v not in (0, 1, 2, 3):
                raise ValueError("which_espread must be 0, 1, 2, or 3")

        # dist_z allowed values
        for v in self.dist_z:
            if v not in (0, 1):
                raise ValueError("dist_z must be 0 or 1")

        # charge sign constraint
        if self.charge_sign not in (-1, 0, 1):
            raise ValueError("charge_sign must be -1, 0, or 1")

        # positivity checks
        for name in ["particles", "energy"]:
            for v in getattr(self, name):
                _validate_positive(name, v)

        # non-negativity checks
        _validate_non_negative("f_rep", self.f_rep)
        _validate_non_negative("n_b", self.n_b)

        for name in ["sigma_x", "sigma_y", "sigma_z"]:
            for v in getattr(self, name):
                _validate_non_negative(name, v)

    def copy(self):
        return copy.copy(self)
    
    def as_dict(self):
        return asdict(self)
         


# ============================================================
# Simulation configuration
# ============================================================

@dataclass
class SimulationConfig:
    # Grid and macroparticles
    n_x: int
    n_y: int
    n_z: int
    n_t: int = 1

    n_m: Int2 = (100000, 100000)

    cut_x: float = -1.0
    cut_y: float = -1.0
    cut_z: float = 0.0

    integration_method: int = 2  # 1,2,3
    force_symmetric: int = 0

    rndm_save: int = 0
    rndm_load: int = 0

    # Luminosity
    do_lumi: int = 0
    num_lumi: int = 10000
    lumi_p: float = 1e-23

    # do_gglumi: int = 0  # apparently not yet implemented in guinea
    # num_gglumi: int = 10000
    # gglumi_p: float = 1e-23

    ecm_min: float = 0.0

    # Storage cuts
    store_beam: int = 0
    # storebeam_ptmin: float = 0.0  # apparently not yet implemented in guinea
    # storebeam_ptmax: float = 1e20
    # storebeam_angmin: float = 0.0
    # storebeam_angmax: float = 1e20

    # Background and ratios
    electron_ratio: float = 1.0
    do_photons: int = 0
    photon_ratio: float = 1.0

    # Hadrons / jets
    do_hadrons: int = 0  # 0,1,2
    store_hadrons: int = 0
    hadron_ratio: float = 1e5

    do_jets: int = 0
    store_jets: int = 0
    jet_ptmin: float = 2.0
    jet_ratio: float = 1e5
    jet_log: int = 1

    # Pairs
    do_pairs: int = 0
    pair_ratio: float = 1.0
    pair_q2: int = 1  # 0,1,2
    track_pairs: int = 0
    grids: int = 0

    pair_ecut: float = 0.511e-3
    pair_step: float = 1.0

    # pairs_ptmin: float = 0.0  # apparently not yet implemented in guinea
    # pairs_ptmax: float = 1e20
    # pairs_angmin: float = 0.0
    # pairs_angmax: float = 1e20

    # Physics switches
    beam_size: int = 1  # 1 or 2
    do_eloss: int = 1
    do_espread: int = 1
    do_isr: int = 0

    do_compt: int = 0
    compt_emax: float = 10000.0
    compt_x_min: float = 0.01

    load_beam: int = 3
    load_photons: int = 0
    store_photons: int = 0

    do_prod: int = 0
    prod_e: float = 0.0
    prod_scal: float = 1e-29

    do_cross: int = 0

    # --------------------------------------------------------

    def __post_init__(self):

        len2_fields = [
            "n_m",
        ]

        # --- Auto-expand scalars ---
        for name in len2_fields:
            value = getattr(self, name)
            value = _to_len2(value, name)
            setattr(self, name, value)

        # --- Now validate ---
        for name in len2_fields:
            _validate_len2(name, getattr(self, name))

        # positivity
        for name in ["n_x", "n_y", "n_z", "n_t"]:
            _validate_positive(name, getattr(self, name))

        # integration method
        if self.integration_method not in (1, 2, 3):
            raise ValueError("integration_method must be 1, 2, or 3")

        # power-of-two requirement
        if self.integration_method == 2:
            if not _is_power_of_two(self.n_x):
                raise ValueError("n_x must be power of two for integration_method=2")
            if not _is_power_of_two(self.n_y):
                raise ValueError("n_y must be power of two for integration_method=2")

        # enumerations
        if self.do_hadrons not in (0, 1, 2):
            raise ValueError("do_hadrons must be 0, 1, or 2")

        if self.pair_q2 not in (0, 1, 2):
            raise ValueError("pair_q2 must be 0, 1, or 2")

        if self.beam_size not in (1, 2):
            raise ValueError("beam_size must be 1 or 2")

        # ratios
        _validate_ratio("electron_ratio", self.electron_ratio)
        _validate_ratio("photon_ratio", self.photon_ratio)
        _validate_ratio("pair_ratio", self.pair_ratio)

        # logical consistency
        if self.track_pairs != 0 and self.grids <= 0:
            raise ValueError("grids must be > 0 when track_pairs is enabled")

    def copy(self):
        return copy.copy(self)

    def as_dict(self):
        return asdict(self)