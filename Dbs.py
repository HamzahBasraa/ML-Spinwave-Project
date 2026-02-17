

import numpy as np
import pandas as pd
from subprocess import run, PIPE, STDOUT
from glob import glob
from os import path
from numpy import load
import os
import re
import shutil
import time



SCRIPT_TEMPLATE = "task1.mx3"
GRID = 0.1e-6
LIMIT = 0.5e-6

N_RESTARTS = 3
MAX_LOCAL_STEPS = 15



position_dict = {
    "c1_x": 3e-7,  "c1_y": -7e-7,
    "c2_x": -1e-7, "c2_y": 4e-7,
    "c3_x": 4e-7,  "c3_y": -2e-7,
    "c4_x": -8e-7, "c4_y": 0.0e-7,
    "c5_x": 5e-7,  "c5_y": 6e-7,
    "c6_x": -4e-7, "c6_y": -1e-7,
    "c7_x": 0.0e-7,"c7_y": -2e-7,
    "c8_x": 3e-7,  "c8_y": 5e-7,
    "c9_x": -2e-7, "c9_y": 3e-7,
    "c10_x": 4e-7, "c10_y": -4e-7
}

PARAM_KEYS = list(position_dict.keys())
N_PARAMS = len(PARAM_KEYS)

def read_mumax3_table(filename):
    table = pd.read_table(filename)
    table.columns = ' '.join(table.columns).split()[1::2]
    return table


def read_mumax3_ovffiles(outputdir):
    """
    Converts OVF files to numpy and loads them.
    EXACTLY like your GA version.
    """

    # Convert all OVF files
    p = run(["mumax3-convert", "-numpy", outputdir + "/*.ovf"],
            stdout=PIPE, stderr=STDOUT)

    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    # Load converted NPY files
    fields = {}
    for npyfile in glob(outputdir + "/*.npy"):
        key = path.splitext(path.basename(npyfile))[0]
        fields[key] = load(npyfile)

    return fields


def run_mumax3(script, name):

    scriptfile = name + ".txt"
    outputdir  = name + ".out"

    with open(scriptfile, 'w') as f:
        f.write(script)

    p = run(["mumax3", "-f", scriptfile],
            stdout=PIPE, stderr=STDOUT)

    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    # Run conversion exactly like your GA workflow
    read_mumax3_ovffiles(outputdir)

    return outputdir


# ==========================================================
# SCRIPT UPDATE (UNCHANGED FROM YOUR VERSION)
# ==========================================================

def update_parameters(input_file, position_dict):

    with open(input_file, 'r') as f:
        text = f.read()

    for key, value in position_dict.items():
        value_str = f"{float(value):.6g}"
        pattern = rf"(\b{re.escape(key)}\b\s*:=\s*)([-+0-9.eE]+)"
        text, _ = re.subn(pattern, rf"\g<1>{value_str}", text)

    with open(input_file, 'w') as f:
        f.write(text)


# ==========================================================
# FFT EXTRACTION (UNCHANGED LOGIC)
# ==========================================================

def extract_detector_fft(output_path,
                         dt=100e-12,
                         cellsize=5e-9,
                         f_drive=2.6e9):

    files = sorted(glob(output_path + '/m*.npy'))

    n_components, n_z, n_y, n_x = np.load(files[0]).shape
    n_time = len(files)

    data_5d = np.zeros((n_time, n_components, n_z, n_y, n_x))

    for i, f in enumerate(files):
        data_5d[i] = np.load(f)

    z_middle = int(n_z / 2)
    my_t = data_5d[:, 1, z_middle, :, :]
    my_t -= np.mean(my_t, axis=0)

    discard = int(5e-9 / dt)
    my_t = my_t[discard:]
    n_time = my_t.shape[0]

    fast_transform = np.fft.fft(my_t, axis=0)
    freqs = np.fft.fftfreq(n_time, dt)
    idx = np.argmin(np.abs(freqs - f_drive))

    amplitude = np.abs(fast_transform[idx])

    Lx = n_x * cellsize
    Ly = n_y * cellsize

    def phys_to_ix(x): return int((x + Lx / 2) / cellsize)
    def phys_to_iy(y): return int((y + Ly / 2) / cellsize)

    detectors = {
        "output1": {"cx": 0.75e-6, "cy":  0.35e-6, "w": 0.2e-6, "h": 0.3e-6},
        "output2": {"cx": 0.75e-6, "cy": -0.35e-6, "w": 0.2e-6, "h": 0.3e-6},
    }

    results = {}

    for name, det in detectors.items():
        cx, cy, w, h = det.values()

        ix_min = phys_to_ix(cx - w / 2)
        ix_max = phys_to_ix(cx + w / 2)
        iy_min = phys_to_iy(cy - h / 2)
        iy_max = phys_to_iy(cy + h / 2)

        ix_min = max(ix_min, 0)
        iy_min = max(iy_min, 0)
        ix_max = min(ix_max, n_x - 1)
        iy_max = min(iy_max, n_y - 1)

        region = amplitude[iy_min:iy_max + 1,
                           ix_min:ix_max + 1]

        results[name] = np.mean(region)

    return results["output1"], results["output2"]


# ==========================================================
# FITNESS (UNCHANGED LOGIC)
# ==========================================================

def evaluate_fitness(candidate, run_id):

    for i, key in enumerate(PARAM_KEYS):
        position_dict[key] = candidate[i]

    update_parameters(SCRIPT_TEMPLATE, position_dict)

    with open(SCRIPT_TEMPLATE, 'r') as f:
        script_content = f.read()

    output_dir = run_mumax3(script_content, run_id)

    out1_f1, out2_f1 = extract_detector_fft(output_dir, f_drive=2.6e9)
    score_f1 = max(out1_f1 - out2_f1, 0)

    out1_f2, out2_f2 = extract_detector_fft(output_dir, f_drive=2.8e9)
    score_f2 = max(out2_f2 - out1_f2, 0)

    total = score_f1 * score_f2

    shutil.rmtree(output_dir, ignore_errors=True)
    os.remove(run_id + ".txt")

    print(f"{run_id} | Fitness = {total:.4e}")

    return total


# ==========================================================
# COORDINATE DBS
# ==========================================================

def create_random_individual():
    n_steps = int(2 * LIMIT / GRID)
    random_steps = np.random.randint(-n_steps//2,
                                     n_steps//2 + 1,
                                     size=N_PARAMS)
    return random_steps * GRID


def coordinate_local_search(start_candidate, restart_id):

    current = start_candidate.copy()
    best_fitness = evaluate_fitness(current,
                                    f"dbs_r{restart_id}_start")

    step = 0

    while step < MAX_LOCAL_STEPS:

        improved = False

        for i in range(N_PARAMS):

            for direction in [-1, 1]:

                trial = current.copy()
                trial[i] += direction * GRID
                trial[i] = np.clip(trial[i], -LIMIT, LIMIT)

                fitness = evaluate_fitness(
                    trial,
                    f"dbs_r{restart_id}_s{step}_p{i}_{direction}"
                )

                if fitness > best_fitness:
                    current = trial
                    best_fitness = fitness
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

        step += 1

    return current, best_fitness


def run_dbs():

    global_best = None
    global_best_fitness = -1e99

    for r in range(N_RESTARTS):

        print(f"\n===== RESTART {r} =====")

        start = create_random_individual()
        candidate, fitness = coordinate_local_search(start, r)

        if fitness > global_best_fitness:
            global_best = candidate
            global_best_fitness = fitness
            np.save("dbs_best.npy", global_best)

    print("\nDBS COMPLETE")
    print("Best Fitness:", global_best_fitness)

    return global_best, global_best_fitness


run_dbs()
