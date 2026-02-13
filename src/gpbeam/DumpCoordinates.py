import os
import numpy as np
from scipy.stats import truncnorm
from .Settings import AcceleratorConfig

try:
    import xtrack
    xtrack_loaded = True
except ImportError:
    xtrack_loaded = False

try:
    import RF_Track
    rf_track_loaded = True
except ImportError:
    rf_track_loaded = False


def dump_coordinates(cur_workdir, coords: np.ndarray | xtrack.particles.particles.Particles, positron: bool):
    if type(coords) == np.ndarray:
        if not len(coords.shape) == 2 or coords.shape[1] != 6:
            raise ValueError("coordinate array must have shape (N,6)")
    elif xtrack_loaded and isinstance(coords, xtrack.particles.particles.Particles):
        particles = coords
        coords = np.empty((6,len(particles.x)))

        coords[:,0] = particles.energy / 1e9  # [GeV]
        coords[:,1] = particles.x * 1e6  # [um]
        coords[:,2] = particles.y * 1e6  # [um]
        coords[:,3] = particles.zeta * 1e6  # [um]
        coords[:,4] = particles.px * 1e6
        coords[:,5] = particles.py * 1e6
    elif rf_track_loaded and isinstance(coords, RF_Track.Bunch6d):
        particles = coords
        coords = np.empty((6,len(particles.x)))

        coords[:,0] = particles.get_phase_space("%K") / 1e3  # [GeV]
        coords[:,1] = particles.get_phase_space("%x") * 1e3  # [um]
        coords[:,2] = particles.get_phase_space("%y") * 1e3  # [um]
        coords[:,3] = particles.get_phase_space("%t") * 1e3 * RF_Track.clight  # [um]
        coords[:,4] = particles.get_phase_space("%xp") * 1e3
        coords[:,5] = particles.get_phase_space("%yp") * 1e3
    else:
        raise ValueError("coords must be of either np.ndarray or xtrack.particles.particles.Particles type")
    
    if positron:
        fname = "positron.ini"
    else:
        fname = "electron.ini"

    np.savetxt(os.path.join(cur_workdir,fname), coords)

    return


def sampleGaussianWaist(accelerator: AcceleratorConfig, cut: int = 3, N: int = 100000,):
    """
    Sample Gaussian waist particle distributions from an AcceleratorConfig.
    """
    betx_1 = accelerator.beta_x[0] * 1e3  # mm → µm
    betx_2 = accelerator.beta_x[1] * 1e3

    bety_1 = accelerator.beta_y[0] * 1e3
    bety_2 = accelerator.beta_y[1] * 1e3

    sigma_z_1 = accelerator.sigma_z[0]
    sigma_z_2 = accelerator.sigma_z[1]

    energy_1 = accelerator.energy[0]
    energy_2 = accelerator.energy[1]

    espread_1 = accelerator.espread[0]
    espread_2 = accelerator.espread[1]

    lorentz_1 = (energy_1 * 1e9 - 511e3) / 511e3
    lorentz_2 = (energy_2 * 1e9 - 511e3) / 511e3

    ex_1 = accelerator.emitt_x[0] / lorentz_1
    ex_2 = accelerator.emitt_x[1] / lorentz_2

    ey_1 = accelerator.emitt_y[0] / lorentz_1
    ey_2 = accelerator.emitt_y[1] / lorentz_2

    x0_1 = accelerator.offset_x[0]
    x0_2 = accelerator.offset_x[1]

    y0_1 = accelerator.offset_y[0]
    y0_2 = accelerator.offset_y[1]

    which_espread_1 = accelerator.which_espread[0]
    which_espread_2 = accelerator.which_espread[1]

    # sample initial conditions
    beam_1 = [
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ex_1 * betx_1), size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ey_1 * bety_1), size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=sigma_z_1, size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ex_1 * 1/betx_1), size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ey_1 * 1/bety_1), size=N),
    ]

    beam_2 = [
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ex_2 * betx_2), size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ey_2 * bety_2), size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=sigma_z_2, size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ex_2 * 1/betx_2), size=N),
        truncnorm.rvs(-cut, cut, loc=0, scale=np.sqrt(ey_2 * 1/bety_2), size=N),
    ]

    if which_espread_1 == 0:
        beam_1.insert(0, np.ones(N) * energy_1)
    elif which_espread_1 == 1:
        beam_1.insert(0, ((np.random.random(N) - 1/2) * 2 * espread_1 + 1) * energy_1 )
    elif which_espread_1 == 3:
        beam_1.insert(0, truncnorm.rvs(-cut, cut, loc=energy_1, scale=espread_1, size=N))
    else:
        raise NotImplementedError("Beam_1: unknown setting for which_espread")

    if which_espread_2 == 0:
        beam_2.insert(0, np.ones(N) * energy_2)
    elif which_espread_2 == 1:
        beam_2.insert(0, ((np.random.random(N) - 1/2) * 2 * espread_2 + 1) * energy_2 )
    elif which_espread_2 == 3:
        beam_2.insert(0, truncnorm.rvs(-cut, cut, loc=energy_2, scale=espread_2, size=N))
    else:
        raise NotImplementedError("Beam_2: unknown setting for which_espread")

    # add offset
    beam_1, beam_2 = np.array(beam_1).T, np.array(beam_2).T

    beam_1[:,1:] -= beam_1[:,1:].mean(axis=0)
    beam_2[:,1:] -= beam_2[:,1:].mean(axis=0)

    beam_1[:,1] += x0_1
    beam_2[:,1] += x0_2
    beam_1[:,2] += y0_1
    beam_2[:,2] += y0_2

    return beam_1, beam_2
