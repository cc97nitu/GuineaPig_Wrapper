import os
import time
import re
import subprocess
import shutil

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from pathlib import Path

from .Settings import AcceleratorConfig, SimulationConfig
from .DumpCoordinates import dump_coordinates, sampleGaussianWaist
from .Result import parse_guinea_output


##########################
### general config
##########################
from .config import WORKDIR, EXECUTABLE_GUINEA_PLUSPLUS


def _promote_scalars(cls, raw_dict):
    """
    Promote scalar values to (value, value) if dataclass field expects tuple.
    """

    from typing import get_origin

    promoted = {}

    for field in cls.__dataclass_fields__.values():
        name = field.name
        expected_type = field.type

        if name not in raw_dict:
            continue

        value = raw_dict[name]

        # If expected type is Tuple[...] but value is scalar → promote
        if (
            get_origin(expected_type) is tuple
            and not isinstance(value, tuple)
        ):
            promoted[name] = (value, value)
        else:
            promoted[name] = value

    return promoted


@dataclass
class GuineaPig:
    """
    Container for a full GuineaPig configuration project.

    Holds multiple accelerator definitions and multiple simulation
    parameter sets.
    """

    accelerators: Dict[str, "AcceleratorConfig"] = field(default_factory=dict)
    simulations: Dict[str, "SimulationConfig"] = field(default_factory=dict)

    active_accelerator: Optional[str] = None
    active_simulation: Optional[str] = None

    # ============================================================
    # Basic management
    # ============================================================

    def add_accelerator(self, name: str, config: "AcceleratorConfig"):
        self.accelerators[name] = config

    def add_simulation(self, name: str, config: "SimulationConfig"):
        self.simulations[name] = config

    def set_active(self, accelerator: str, simulation: str):
        if accelerator not in self.accelerators:
            raise KeyError(f"Unknown accelerator '{accelerator}'")
        if simulation not in self.simulations:
            raise KeyError(f"Unknown simulation '{simulation}'")

        self.active_accelerator = accelerator
        self.active_simulation = simulation

    # ============================================================
    # Accessors
    # ============================================================
    @property
    def accelerator(self) -> "AcceleratorConfig":
        if self.active_accelerator is None:
            # implicitly activate only choice
            if len(self.accelerators) == 1:
                self.active_accelerator = tuple(self.accelerators.keys())[0]
                return self.accelerators[self.active_accelerator]
            else:
                raise RuntimeError("No active accelerator selected")
        return self.accelerators[self.active_accelerator]

    @property
    def simulation(self) -> "SimulationConfig":
        if self.active_simulation is None:
            # implicitly activate only choice
            if len(self.simulations) == 1:
                self.active_simulation = tuple(self.simulations.keys())[0]
                return self.simulations[self.active_simulation]
            else:
                raise RuntimeError("No active simulation selected")
        return self.simulations[self.active_simulation]

    # ============================================================
    # File I/O
    # ============================================================
    @classmethod
    def load(cls, filepath: str | Path) -> "GuineaPig":
        with open(filepath, "r") as f:
            text = f.read()

        section_pattern = re.compile(
            r"\$(\w+)::\s*(\w+)\s*\{([^}]*)\}",
            re.DOTALL
        )

        project = cls()

        for section, name, body in section_pattern.findall(text):

            raw = {}
            tuple_accumulator = {}

            for entry in body.split(";"):
                entry = entry.strip()
                if not entry:
                    continue

                if "=" not in entry:
                    continue

                key, value = entry.split("=", 1)
                key = key.strip()
                value = value.strip()

                # DO NOT evaluate expressions like 3.0*sigma_z.1
                try:
                    value_eval = eval(value, {"__builtins__": None}, {})
                except Exception:
                    value_eval = value  # keep expression string

                # convert GP naming → python naming
                key = key.replace(" ", "_")

                # handle .1 / .2
                if "." in key:
                    base, idx = key.split(".")
                    idx = int(idx) - 1

                    tuple_accumulator.setdefault(base, [None, None])
                    tuple_accumulator[base][idx] = value_eval
                else:
                    raw[key] = value_eval

            # finalize tuples
            for k, v in tuple_accumulator.items():
                if v[0] is None or v[1] is None:
                    raise ValueError(f"Incomplete tuple for {k}")
                raw[k] = tuple(v)

            # Promote scalar to tuple if dataclass expects tuple
            if section.upper() == "ACCELERATOR":
                obj = AcceleratorConfig(**_promote_scalars(AcceleratorConfig, raw))
                project.accelerators[name] = obj

            elif section.upper() == "PARAMETERS":
                obj = SimulationConfig(**_promote_scalars(SimulationConfig, raw))
                project.simulations[name] = obj

        return project

    def save(self, filepath: str | Path):
        lines = []

        def write_section(section_name, objects):
            for name, obj in objects.items():

                lines.append(f"${section_name}:: {name}")
                lines.append("{")

                data = asdict(obj)

                for key, value in data.items():
                    # gp_key = key.replace("_", " ")
                    gp_key = key

                    if isinstance(value, tuple) and len(value) == 2:
                        lines.append(f"    {gp_key}.1={value[0]};")
                        lines.append(f"    {gp_key}.2={value[1]};")
                    else:
                        lines.append(f"    {gp_key}={value};")

                lines.append("}")

        write_section("ACCELERATOR", self.accelerators)
        write_section("PARAMETERS", self.simulations)

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")    
            
    
    # ============================================================
    # Convenience utilities
    # ============================================================
    def remove_accelerator(self, name: str):
        if name in self.accelerators:
            del self.accelerators[name]
            if self.active_accelerator == name:
                self.active_accelerator = None

    def remove_simulation(self, name: str):
        if name in self.simulations:
            del self.simulations[name]
            if self.active_simulation == name:
                self.active_simulation = None

    def list_accelerators(self):
        return list(self.accelerators.keys())

    def list_simulations(self):
        return list(self.simulations.keys())

    def to_dict(self):
        """
        Export project as nested dictionary (dataclasses preserved).
        """
        return {
            "ACCELERATOR": self.accelerators,
            "PARAMETERS": self.simulations,
        }

    # ============================================================
    # Validation
    # ============================================================
    def validate(self):
        """
        Re-run dataclass validation explicitly.
        Useful if configs were modified after creation.
        """
        for acc in self.accelerators.values():
            acc.__post_init__()

        for sim in self.simulations.values():
            sim.__post_init__()

    # ============================================================
    # Representation
    # ============================================================
    def __repr__(self):
        return (
            f"GuineaPigProject("
            f"{len(self.accelerators)} accelerators, "
            f"{len(self.simulations)} simulations, "
            f"active=({self.active_accelerator}, {self.active_simulation})"
            f")"
        )

    # ============================================================
    # Run Guinea-Pig
    # ============================================================
    def simulate(self, particles_1 = None, particles_2 = None, retain_files: bool = False):
        unique_id = abs(hash((os.getpid(), time.time())))
        cur_workdir = os.path.normpath(WORKDIR +"/workdir_guinea_" + str(unique_id))
        os.makedirs(cur_workdir, exist_ok=False)

        # ensure proper configuration
        _ = self.accelerator
        _ = self.simulation

        self.save(os.path.join(cur_workdir, "acc.dat"))
                  
        # initial conditions
        if particles_1 is not None and particles_2 is not None:
            np_1 = dump_coordinates(cur_workdir, particles_1,positron=False)
            np_2 = dump_coordinates(cur_workdir, particles_2,positron=True)
        elif particles_1 is not None and particles_2 is None:
            np_1 = dump_coordinates(cur_workdir, particles_1,positron=False)
            np_2 = dump_coordinates(cur_workdir, particles_1,positron=True)
        elif particles_1 is None and particles_2 is None:
            particles_1, particles_2 = sampleGaussianWaist(self.accelerator)

            np_1 = dump_coordinates(cur_workdir, particles_1,positron=False)
            np_2 = dump_coordinates(cur_workdir, particles_2,positron=True)
        else:
            raise ValueError("particles_2 must be None if particles_1 is not provided")

        # enforce total bunch charge
        self.simulation.n_m = (np_1, np_2)
        
        # call guinea <- guinea seems to read from stdin always as interactive program <- feed it with fake stdin
        out_file = os.path.join(cur_workdir, "g++_result.out")
        with open(os.devnull, "r") as nullin:
            result = subprocess.run(
                [EXECUTABLE_GUINEA_PLUSPLUS(), self.active_accelerator, self.active_simulation, out_file],
                stdin=nullin,
                stdout=subprocess.PIPE,    # Capture stdout
                stderr=subprocess.STDOUT,  # Capture stderr too
                text=True,                 # Decode bytes -> str
                cwd=cur_workdir,
                check=True
            )
        
        simulation_result = parse_guinea_output(out_file)

        if simulation_result.extra["miss"] > 0.3:
            print("Warning: particles outside of grid, miss rate = ", simulation_result.extra["miss"])

        if retain_files:
            print("INFO: current working directory " + cur_workdir)
            with open(os.path.join(cur_workdir, "guinea.log"), "w") as logfile:
                logfile.write(result.stdout)
        else:
            shutil.rmtree(cur_workdir, ignore_errors=True)

        return simulation_result

