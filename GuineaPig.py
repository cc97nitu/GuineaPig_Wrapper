import os
import re
import time
import shutil
import subprocess
import itertools
import atexit

import numpy as np
#from scipy.stats import truncnorm

##########################
### general config
##########################
WORKDIR = os.path.dirname("/dev/shm/workdir_guinea/")


##########################
### static methods
##########################
def default_accelerator():
    def_acc = {
        "default": {
            "beta_x": 9.1,
            "beta_y": 0.14,
            "dist_z": 0,
            "emitt_x.1": 0.66,
            "emitt_x.2": 0.66,
            "emitt_y.1": 0.02,
            "emitt_y.2": 0.02,
            "energy": 750,
            "espread.1": 0.003,
            "espread.2": 0.003,
            "f_rep": 50.0,
            "n_b": 312,
            "offset_x.1": 0,
            "offset_x.2": 0,
            "offset_y.1": 0,
            "offset_y.2": 0,
            "particles": 0.372,
            "sigma_z.1": 44.0,
            "sigma_z.2": 44.0,
            "waist_y": 0,
        },
    }

    return def_acc


def default_config():
    config = {
        "default": {
            "charge_sign": -1.0,
            "cut_x": -1,
            "cut_y": -1,
            "cut_z": "3.0*sigma_z.1",
            "do_coherent": 1,
            "do_compt": 0,
            "do_hadrons": 0,
            "do_lumi": 0,
            "do_pairs": 0,
            "do_photons": 1,
            "ecm_min": 1485.0,
            "electron_ratio": 0.2,
            "force_symmetric": 0,
            "grids": 0,
            "hist_ee_bins": 1010,
            "hist_ee_max": "2.02*energy.1",
            "load_beam": 3,
            "n_m": 100000,
            "n_t": 1,
            "n_x": 110,
            "n_y": 100,
            "n_z": 50,
            "num_lumi": 100000,
            "photon_ratio": 0.2,
            "rndm_load": 1,
            "rndm_save": 1,
            "store_hadrons": 0,
            "store_pairs": 0,
            "store_photons": 0,
            "track_pairs": 0,
        },
    }

    return config


def parse_guinea_config(filepath):
    """
    Provide a guinea-style configuration file as dictionary.
    """
    with open(filepath, "r") as f:
        text = f.read()

    # Pattern to match sections like $SECTION:: name { ... }
    section_pattern = re.compile(
        r"\$(\w+)::\s*(\w+)\s*\{([^}]*)\}", re.DOTALL
    )

    # Dictionary to store all parsed data
    config = {}

    for match in section_pattern.finditer(text):
        section, name, body = match.groups()
        section_dict = {}

        # Split key=value; pairs
        for entry in body.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                key, value = entry.split("=", 1)
                key, value = key.strip(), value.strip()

                # Try to convert to number if possible
                try:
                    value_eval = eval(value, {"__builtins__": None}, {})
                    section_dict[key] = value_eval
                except Exception:
                    section_dict[key] = value  # keep as string if expression

        config.setdefault(section, {})[name] = section_dict

    return config


def dump_guinea_config(filepath, config):
    """
    Write a guinea-style configuration dictionary back to text.
    """
    lines = []
    for section, subsections in config.items():
        for name, params in subsections.items():
            lines.append(f"${section}:: {name}")
            lines.append("{")

            # Write key=value; pairs — preserving formatting
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    lines.append(f"    {key}={value};")
                else:
                    lines.append(f"    {key}={value};")

            lines.append("}")
    text = "\n".join(lines) + "\n"

    with open(filepath, "w") as f:
        f.write(text)

    return


def sampleGaussianWaist(acc_dat: dict, accelerator: str, cut: int = 3, N: int = 100000):
    try:
        config = acc_dat["ACCELERATOR"][accelerator]
    except KeyError:
        raise ValueError("Accelerator {} not found".format(accelerator))
    
    try:
        if "beta_x.1" in config.keys():
            betx_1 = config["beta_x.1"] * 1e3  # [um]
            betx_2 = config["beta_x.2"] * 1e3  # [um]
        else:
            betx_1 = config["beta_x"] * 1e3  # [um]
            betx_2 = config["beta_x"] * 1e3  # [um]

        if "beta_y.1" in config.keys():
            bety_1 = config["beta_y.1"] * 1e3  # [um]
            bety_2 = config["beta_y.2"] * 1e3  # [um]
        else:
            bety_1 = config["beta_y"] * 1e3  # [um]
            bety_2 = config["beta_y"] * 1e3  # [um]

        if "sigma_z.1" in config.keys():
            sigma_z_1 = config["sigma_z.1"]
            sigma_z_2 = config["sigma_z.2"]
        else:
            sigma_z_1 = config["sigma_z"]
            sigma_z_2 = config["sigma_z"]

        if "energy.1" in config.keys():
            energy_1 = config["energy.1"]
            energy_2 = config["energy.2"]
        else:
            energy_1 = config["energy"]
            energy_2 = config["energy"]

        if "espread.1" in config.keys():
            espread_1 = config["espread.1"]
            espread_2 = config["espread.2"]
        else:
            espread_1 = config["espread"]
            espread_2 = config["espread"]

        lorentz_1 = (energy_1 * 1e9 - 511e3) / 511e3
        lorentz_2 = (energy_2 * 1e9 - 511e3) / 511e3

        if "emitt_x.1" in config.keys():
            ex_1 = config["emitt_x.1"] / lorentz_1
            ex_2 = config["emitt_x.2"] / lorentz_2
        else:
            ex_1 = config["emitt_x"] / lorentz_1
            ex_2 = ex_1

        if "emitt_y.1" in config.keys():
            ey_1 = config["emitt_y.1"] / lorentz_1
            ey_2 = config["emitt_y.2"] / lorentz_2
        else:
            ey_1 = config["emitt_y"] / lorentz_1
            ey_2 = ey_1

        if "offset_x.1" in config.keys():
            x0_1 = config["offset_x.1"]
            x0_2 = config["offset_x.2"]
        elif "offset_x" in config.keys():
            x0_1 = config["offset_x"]
            x0_2 = x0_1
        else:
            x0_1 = 0
            x0_2 = x0_1

        if "offset_y.1" in config.keys():
            y0_1 = config["offset_y.1"]
            y0_2 = config["offset_y.2"]
        elif "offset_y" in config.keys():
            y0_1 = config["offset_y"]
            y0_2 = y0_1
        else:
            y0_1 = 0
            y0_2 = y0_1

        if "which_espread.1" in config.keys():
            which_espread_1 = config["which_espread.1"]
            which_espread_2 = config["which_espread.2"]
        elif "which_espread" in config.keys():
            which_espread_1 = config["which_espread"]
            which_espread_2 = which_espread_1
        else:
            which_espread_1 = 0
            which_espread_2 = which_espread_1

    except KeyError:
        raise ValueError("Incomplete description of accelerator {}".format(accelerator))

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

        





class GuineaPig(object):

    def __init__(self, accelerators: dict, config: dict, use_guineaPig_legacy: bool = True):
        self._rmtree = shutil.rmtree
        self._in_context = False

        self.accelerators = accelerators
        self.config = config

        self.use_guineaPig_legacy = use_guineaPig_legacy

        # create work dir
        self.unique_id = abs(hash((os.getpid(), time.time())))

        return

    def cleanup(self):
        self._rmtree(self.WORKDIR, ignore_errors=True)

    def __enter__(self):
        id = abs(hash((self.unique_id, time.time())))
        self.WORKDIR = os.path.normpath(WORKDIR +"_" + str(id))
        os.makedirs(self.WORKDIR, exist_ok=False)

        self._in_context = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._in_context = False
        self.cleanup()
        return
    
    def dumpCoordinates(self, coords: np.array, positron: bool):
        if not len(coords.shape) == 2 or coords.shape[1] != 6:
            raise RuntimeError("coordinate array must have shape (N,6)")

        if not self._in_context:
            raise RuntimeError("simulate can only be called within a GuineaPig context")

        if positron:
            fname = "positron.ini"
        else:
            fname = "electron.ini"
        
        np.savetxt(os.path.join(self.WORKDIR,fname), coords)
        return
            
    def simulate(self, accelerator: str = "default", particles_1 = None, particles_2 = None):
        if not self._in_context:
            raise RuntimeError("simulate can only be called within a GuineaPig context")
            
        acc_dat = {"ACCELERATOR": self.accelerators, "PARAMETERS": self.config}
        dump_guinea_config(os.path.join(self.WORKDIR, "acc.dat"), acc_dat)

        # initial conditions
        if particles_1 is not None and particles_2 is not None:
            self.dumpCoordinates(particles_1,positron=False)
            self.dumpCoordinates(particles_2,positron=True)
        elif particles_1 is not None and particles_2 is None:
            self.dumpCoordinates(particles_1,positron=False)
            self.dumpCoordinates(particles_1,positron=True)
        elif particles_1 is None and particles_2 is None:
            particles_1, particles_2 = sampleGaussianWaist(acc_dat, accelerator)

            self.dumpCoordinates(particles_1,positron=False)
            self.dumpCoordinates(particles_2,positron=True)
        else:
            raise ValueError("particles_2 must be None if particles_1 is not provided")

        if self.use_guineaPig_legacy:
            lumi_ee, lumi_ee_high, miss = self._call_guineaLegacy(accelerator)
        else:
            lumi_ee, lumi_ee_high, miss = self._call_guineaPlusPlus(accelerator)

        return lumi_ee, lumi_ee_high, miss

    def _call_guineaLegacy(self, accelerator: str):
        # call guinea <- guinea seems to read from stdin always as interactive program <- feed it with fake stdin
        with open(os.devnull, "r") as nullin:
            result = subprocess.run(
                ["guinea", accelerator, "default", os.path.join(self.WORKDIR, "result.out")],
                stdin=nullin,
                stdout=subprocess.PIPE,    # Capture stdout
                stderr=subprocess.STDOUT,  # Capture stderr too
                text=True,                 # Decode bytes -> str
                cwd=self.WORKDIR,
                check=True
            )
        
        with open(os.path.join(self.WORKDIR, "guinea.log"), "w") as logfile:
            logfile.write(result.stdout)

        # Use regex to find both lumi_ee and lumi_ee_high
        pattern = r"lumi_ee\s*=\s*([0-9.eE+-]+).*?lumi_ee_high\s*=\s*([0-9.eE+-]+)"
        match = re.search(pattern, result.stdout, re.DOTALL)
        
        if match:
            lumi_ee = float(match.group(1))
            lumi_ee_high = float(match.group(2))
        else:
            print("Luminosities not found in guinea output")

        # and the proportion of beam being outside the grid
        pattern = r"miss_1=([0-9.]+);.*?miss_2=([0-9.]+);"
        match = re.search(pattern, result.stdout, re.DOTALL)
        
        if match:
            miss_1 = float(match.group(1))
            miss_2 = float(match.group(2))

            if miss_1 > 0.3 or miss_2 > 0.3:
                print("missing", miss_1, miss_2)

        else:
            print("miss values not found in guinea output")

        return lumi_ee, lumi_ee_high, max(miss_1, miss_2)

    def _call_guineaPlusPlus(self, accelerator: str):
        # call guinea <- guinea seems to read from stdin always as interactive program <- feed it with fake stdin
        with open(os.devnull, "r") as nullin:
            result = subprocess.run(
                ["/opt/SimulationCodes/guinea-pig/bin/guinea", accelerator, "default", os.path.join(self.WORKDIR, "g++_result.out")],
                stdin=nullin,
                stdout=subprocess.PIPE,    # Capture stdout
                stderr=subprocess.STDOUT,  # Capture stderr too
                text=True,                 # Decode bytes -> str
                cwd=self.WORKDIR,
                check=True
            )
        
        with open(os.path.join(self.WORKDIR, "guinea.log"), "w") as logfile:
            logfile.write(result.stdout)

        # Use regex to find both lumi_ee and lumi_ee_high
        with open(os.path.join(self.WORKDIR, "g++_result.out"), "r") as f:
            result_file = f.read()
        
        pattern = r"lumi_ee\s*=\s*([0-9.eE+-]+).*?lumi_ee_high\s*=\s*([0-9.eE+-]+)"
        match = re.search(pattern, result_file, re.DOTALL)
        
        if match:
            lumi_ee = float(match.group(1))
            lumi_ee_high = float(match.group(2))
        else:
            print("Luminosities not found in guinea output")

        # and the proportion of beam being outside the grid
        pattern = r"miss\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        matches = re.findall(pattern, result_file)
        
        if matches:
            miss_1 = float(matches[0])
            miss_2 = float(matches[1])
        
            if miss_1 > 0.3 or miss_2 > 0.3:
                print("missing", miss_1, miss_2)
        
        else:
            print("miss values not found in guinea output")

        return lumi_ee, lumi_ee_high, max(miss_1, miss_2)

