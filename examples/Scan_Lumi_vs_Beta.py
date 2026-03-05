import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import gpbeam as gp


def max_beamsize(beta, ex, l):
    return np.sqrt(beta * ex) + l * np.sqrt(1/beta * ex)



# accerlerator setup
acc = gp.AcceleratorConfig(
    particles=(2,2),
    energy=(275,275),
    espread=(1.5e-3,1.5e-3),
    emitt_x=(5,5),
    emitt_y=(0.035,0.035),
    sigma_z=(300,300),
    beta_x=(13,13),
    beta_y=(0.41,0.41),
    f_rep=10,
    n_b=1312,
)

# calculate 1% cut-off
energy = max(acc.energy)  # unit GeV
ecm_min = energy * 2 * 0.99  # cut 1% lumi at this energy 
lorentz = (energy * 1e9 + 511e3) / 511e3 

def_beta_x, def_beta_y = max(acc.beta_x), max(acc.beta_y)  # [m]

max_ex = max(acc.emitt_x) * 1e-6 / lorentz  # geo. emittance [m]
max_ey = max(acc.emitt_y) * 1e-6 / lorentz  # geo. emittance [m]
max_sigma_z = max(acc.sigma_z) * 1e-6  # rms bunch length [m]

# guinea setup
sim = gp.SimulationConfig(
    n_x=128,
    n_y=128,
    n_z=100,
    cut_z=3*acc.sigma_z[0],
    ecm_min=ecm_min,
)


# --- helper function executed in each thread ---
def run_point(beta_x, beta_y):
    # setup
    acc_conf = acc.copy()
    acc_conf.beta_x = (beta_x * 1e3, beta_x * 1e3)
    acc_conf.beta_y = (beta_y * 1e3, beta_y * 1e3)

    gp_setup = sim.copy()
    gp_setup.cut_x = 6 * max_beamsize(beta_x, max_ex, 6 * max_sigma_z) * 1e9
    gp_setup.cut_y = 6 * max_beamsize(beta_y, max_ey, 6 * max_sigma_z) * 1e9

    print("dimensions of the grid: beta_x={:.1e}, x={:.0e}[nm], beta_y={:.1e}, y={:.0e}[nm]".format(beta_x, gp_setup.cut_x, beta_y, gp_setup.cut_y))

    # simulate
    guinea = gp.GuineaPig()
    guinea.add_accelerator("LCF_CLIC", acc_conf)
    guinea.add_simulation("default", gp_setup)

    simulation_result = guinea.simulate()

    # result row
    res = {
        "beta_x": beta_x,
        "beta_y": beta_y,
        "lumi_ee": simulation_result.general.lumi_ee,
        "lumi_ee_high": simulation_result.general.lumi_ee_high,
        "miss_value": simulation_result.extra["miss"],
    }
    res = acc_conf.as_dict() | gp_setup.as_dict() | res

    return res

 
if __name__ == "__main__":
    #####################################
    # scan beta functions
    #####################################
    verBeta_range = np.logspace(-6,-2,2)
    horBeta_range = np.logspace(-7,-3,2) * def_beta_x / def_beta_y

    results = list()

    points = list(itertools.product(horBeta_range, verBeta_range))
    totalLoopTraversals = len(points)
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_point, *point): i
            for i, point in enumerate(points)
        }

        for future in as_completed(futures):
            i = futures[future]
            res = future.result()

            print(
                f"iteration {i+1}/{totalLoopTraversals}  "
                f"bet_x: {res['beta_x']:.2e} [m],  "
                f"bet_y: {res['beta_y']:.2e} [m],  "
                f"lumi_ee: {res['lumi_ee']:.4e},  "
                f"lumi_ee_high: {res['lumi_ee_high']:.4e}"
            )

            results.append(res)


    # save results
    results = pd.DataFrame(results)
    #results.to_csv("Lumi_vs_BetaFunction.csv")
