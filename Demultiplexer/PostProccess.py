import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from subprocess import run, PIPE, STDOUT
from glob import glob
from os import path
from numpy import load
import os
import re
import time 
import random
import shutil

file = 'task1.out/table.txt'
output = "scriptcopy.out"
name = "scriptcopy"
file_path = 'task1.mx3'


position_dict = {"c1_x": 3e-7,
                "c1_y": -7e-7,
                "c2_x": -1e-7,
                "c2_y": 4e-7,
                "c3_x": 4e-7,
                "c3_y": -2e-7,
                "c4_x": -8e-7,
                "c4_y": 0.0e-7,
                "c5_x": 5e-7,
                "c5_y": 6e-7,
                "c6_x": -4e-7,
                "c6_y": -1e-7,
                "c7_x": 0.0e-7,
                "c7_y": -2e-7,
                "c8_x": 3e-7,
                "c8_y": 5e-7,
                "c9_x": -2e-7,
                "c9_y": 3e-7,
                "c10_x": 4e-7,
                "c10_y": -4e-7,
                "c11_x": 1e-7,
                "c11_y": -4e-7,
                "c12_x": 1e-7,
                "c12_y": 4e-7,
                "c13_x": 2e-7,
                "c13_y": 3e-7,
                "c14_x": 5e-7,}
PARAM_KEYS = list(position_dict.keys())

with open(file_path, 'r') as file: #script 
    MumaxScript = file.read()

def read_mumax3_table(filename):
    """Puts the mumax3 output table in a pandas dataframe"""

    from pandas import read_table
    
    table = read_table(filename)
    table.columns = ' '.join(table.columns).split()[1::2]
    
    return table

def read_mumax3_ovffiles(outputdir):
    """Load all ovffiles in outputdir into a dictionary of numpy arrays 
    with the ovffilename (without extension) as key"""

    # convert all ovf files in the output directory to numpy files
    p = run(["mumax3-convert","-numpy",outputdir+"/*.ovf"], stdout=PIPE, stderr=STDOUT)
    print(p.stdout.decode('utf-8'))
    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    # read the numpy files (the converted ovf files)
    fields = {}
    for npyfile in glob(outputdir+"/*.npy"):
        key = path.splitext(path.basename(npyfile))[0]
        fields[key] = load(npyfile)
    
    return fields

def run_mumax3(script, name, verbose=False):
    """ Executes a mumax3 script and convert ovf files to numpy files
    
    Parameters
    ----------
      script:  string containing the mumax3 input script
      name:    name of the simulation (this will be the name of the script and output dir)
      verbose: print stdout of mumax3 when it is finished
    """
    start = time.time()
    scriptfile = name + ".txt" 
    outputdir  = name + ".out"

    # write the input script in scriptfile
    with open(scriptfile, 'w' ) as f:
        f.write(script)
    
    # call mumax3 to execute this script
    p = run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
    if verbose or p.returncode != 0:
        print(p.stdout.decode('UTF-8'))
        
    if path.exists(outputdir + "/table.txt"):
        table = read_mumax3_table(outputdir + "/table.txt")
    else:
        table = None
        
    fields = read_mumax3_ovffiles(outputdir)
    
    end = time.time()
    print(end-start)
    return table, fields


def fft(output_path):

    files = sorted(glob(output_path + '/m*.npy')) #ensures theyre in order 
    n_components, n_z, n_y, n_x = np.load(files[4]).shape
    n_time = len(files)
    dt = 100e-12 

    # Create empty 5D array
    data_5d = np.zeros((n_time, n_components, n_z, n_y, n_x))
        
    for i,f in enumerate(files): #for each file 
        data_5d[i] = np.load(f) #this is the 5d array
    My_initial = data_5d[:,1,0,:,:] #3d array with time, x , y 

    z_middle = int(n_z / 2) # added this as de multiplexer is 3d
    my_t = data_5d[:, 1, z_middle, :, :]
    
    discard = int(5e-9 / dt)     # 5 ns
    my_t = my_t[discard:]
    n_time = my_t.shape[0]
    #performing fast fourier transform
    #------------------------------------------------
    fast_transform = np.fft.fft(my_t, axis=0)  
    freqs = np.fft.fftfreq(n_time, dt)


    f_drive = 2.6e9
    idx = np.argmin(np.abs(freqs - f_drive))
    amplitude = np.abs(fast_transform[idx, :, :])

    
    #-------------------------------------------------

    amp_norm = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
    plt.imshow(amp_norm, origin='lower', cmap='jet')
    plt.colorbar(label='Normalized |My| Amplitude')
    plt.show()

def visualise(output_path):
    # so what i want to do now is 
        # use nested for loops and to plot a the graph
        # an array should store the intesity of y at each point in the matrix
        # the matrix's coordinates ie [x][y] will store the normalised intensity of y at each point

    files = sorted(glob(output_path + '/m*.npy')) #ensures theyre in order 
        
    for f in files: #for each file 
        a = np.load(f) #load the numpy arrays  in to python

        first_index = 1 if a.shape[0] > 1 else 0

        # My_first = a[first_index, 0, :, :]  # shape will be (32, 128)
        My_test = a[first_index, :, :, :]          # (Nz, Ny, Nx)
        My_first = My_test.mean(axis=0)

        My = (My_first-My_first.min())/(My_first.max()-My_first.min())

        plt.imshow(My, origin='lower', cmap='jet')
        plt.title("My across the film")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="My")
        plt.show()


def update_parameters(input_file, position_dict):
    with open(input_file, 'r') as f:
        text = f.read()

    for key, value in position_dict.items():
        value_str = f"{float(value):.6g}"


        pattern = rf"(\b{re.escape(key)}\b\s*:=\s*)([-+0-9.eE]+)"
        text, n = re.subn(pattern, rf"\g<1>{value_str}", text)

        if n == 0:
            print(f"WARNING: parameter '{key}' not found")

    with open(input_file, 'w') as f:
        f.write(text)

    print("Parameters updated successfully")


def extract_detector_fft(output_path, dt=100e-12, cellsize=5e-9, f_drive=2.6e9):
    import numpy as np
    import matplotlib.pyplot as plt
    from glob import glob

    # -----------------------
    # Load time-series data
    # -----------------------
    files = sorted(glob(output_path + '/*.npy'))
    n_components, n_z, n_y, n_x = np.load(files[0]).shape
    n_time = len(files)

    data_5d = np.zeros((n_time, n_components, n_z, n_y, n_x))

    for i, f in enumerate(files):
        data_5d[i] = np.load(f)

    # Use My component (index 1) at z = 0
    
    z_middle = int(n_z / 2) # added this as de multiplexer is 3d
    my_t = data_5d[:, 1, z_middle, :, :]
    # Remove DC component (better than subtracting first frame)
    my_t = my_t - np.mean(my_t, axis=0)
    discard = int(5e-9 / dt)     # 5 ns
    my_t = my_t[discard:]
    n_time = my_t.shape[0]
    # -----------------------
    # FFT
    # -----------------------
    fast_transform = np.fft.fft(my_t, axis=0)
    freqs = np.fft.fftfreq(n_time, dt)

    idx = np.argmin(np.abs(freqs - f_drive))

    amplitude = np.abs(fast_transform[idx]) # focused on power instead of amplitude

    # -----------------------
    # Geometry / coordinates
    # -----------------------
    Lx = n_x * cellsize
    Ly = n_y * cellsize

    def phys_to_ix(x):
        return int((x + Lx / 2) / cellsize)

    def phys_to_iy(y):
        return int((y + Ly / 2) / cellsize)

    # -----------------------
    # Detector definitions
    # -----------------------
    detectors = {
        "output1": {"cx": 0.75e-6, "cy":  0.35e-6, "w": 0.2e-6, "h": 0.3e-6},
        "output2": {"cx": 0.75e-6, "cy": -0.35e-6, "w": 0.2e-6, "h": 0.3e-6},
    }

    results = {}

    # -----------------------
    # Plot FFT amplitude
    # -----------------------
    # plt.figure(figsize=(8, 3))
    # plt.imshow(amplitude, origin="lower", cmap="inferno")
    # plt.colorbar(label="|My(f)|")
    # plt.title("Detector Regions on FFT Amplitude")

    # -----------------------
    # Extract detector values
    # -----------------------
    for name, det in detectors.items():
        cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]

        ix_min = phys_to_ix(cx - w / 2)
        ix_max = phys_to_ix(cx + w / 2)
        iy_min = phys_to_iy(cy - h / 2)
        iy_max = phys_to_iy(cy + h / 2)

        # Clamp to array bounds
        ix_min = max(ix_min, 0)
        iy_min = max(iy_min, 0)
        ix_max = min(ix_max, n_x - 1)
        iy_max = min(iy_max, n_y - 1)

        # Average FFT amplitude inside detector
        region = amplitude[iy_min:iy_max + 1, ix_min:ix_max + 1]
        results[name] = np.mean(region)

    #     # ---- DRAW RECTANGLE ----
    #     plt.gca().add_patch(
    #         plt.Rectangle(
    #             (ix_min, iy_min),
    #             ix_max - ix_min,
    #             iy_max - iy_min,
    #             fill=False,
    #             edgecolor="cyan",
    #             linewidth=2
    #         )
    #     )

    #     # ---- DRAW CENTER POINT ----
    #     plt.scatter(
    #         phys_to_ix(cx),
    #         phys_to_iy(cy),
    #         c="white",
    #         s=30
    #     )

    # plt.xlabel("x (cells)")
    # plt.ylabel("y (cells)")
    # plt.tight_layout()
    # plt.show()

    print("Detector values:")
    print("Output 1:", results["output1"])
    print("Output 2:", results["output2"])

    return results["output1"], results["output2"]


def evaluate_fitness(candidate_vector, run_id):
    # --- 1. UPDATE YOUR EXISTING DICTIONARY ---
    # We take the numbers from the GA and put them into position_dict
    for i, key in enumerate(PARAM_KEYS):
        position_dict[key] = candidate_vector[i]


    temp_script_path = f"task1.mx3"


    update_parameters(temp_script_path, position_dict)

    # # Read the updated script back into memory to pass to the runner
    with open(temp_script_path, 'r') as f:
        script_content = f.read()

    # --- 3. RUN SIMULATION ---
    output_dir = run_id + ".out"
    total_fitness = -100.0 

    try:
        # Run Mumax using the script content we just prepared
        run_mumax3(script_content, run_id)
        
        # --- 4. ANALYZE (Same as before) ---
        # Frequency 1 (2.6 GHz) -> Target: Output 1
        out1_f1, out2_f1 = extract_detector_fft(output_dir, f_drive=2.6e9, dt = 100e-12)
        score_f1 = (out1_f1 - out2_f1)
        score_f1 = max(score_f1, 0)

        # Frequency 2 (2.8 GHz) -> Target: Output 2
        out1_f2, out2_f2 = extract_detector_fft(output_dir, f_drive=2.8e9, dt = 100e-12)
        score_f2 = (out2_f2 - out1_f2)
        score_f2 = max(score_f2, 0)

        total_fitness = score_f1 * score_f2
        print(f"Run {run_id}: F1_Score={score_f1:.2e} | F2_Score={score_f2:.2e} | Total={total_fitness:.4f}")

    except Exception as e:
        print(f"Run {run_id} Failed: {e}")

        # We create a list containing [ID, Fitness, c1_x, c1_y, c2_x, ...]
    log_data = [run_id, total_fitness] + list(candidate_vector)
    columns = ['run_id', 'fitness'] + PARAM_KEYS
    
    # Convert to DataFrame
    df_log = pd.DataFrame([log_data], columns=columns)
    
    # Append to CSV (header=True only if file doesn't exist yet)
    log_file = "ga_history.csv"
    file_exists = os.path.isfile(log_file)
    df_log.to_csv(log_file, mode='a', header=not file_exists, index=False)


    return total_fitness


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Select one individual from the population using tournament selection.

    population: list of candidate vectors
    fitnesses: list of fitness values (same order as population)
    tournament_size: number of individuals competing
    """
    selected_indices = random.sample(range(len(population)), tournament_size)

    best_idx = selected_indices[0]
    best_fitness = fitnesses[best_idx]

    for idx in selected_indices[1:]:
        if fitnesses[idx] > best_fitness:   # MAXIMISATION
            best_idx = idx
            best_fitness = fitnesses[idx]

    return population[best_idx]

def uniform_crossover(parent1, parent2, crossover_rate=0.6):
    """
    Uniform crossover for real-valued vectors.
    
    parent1, parent2: numpy arrays of same length
    crossover_rate: probability of taking gene from parent1
    """
    child = parent1.copy()

    for i in range(len(parent1)):
        if np.random.rand() > crossover_rate:
            child[i] = parent2[i]

    return child


def mutate(candidate,
           mutation_strength=0.1e-6,
           mutation_rate=0.15,
           grid=0.1e-6,
           limit=0.5e-6):
    """
    Mutation with:
    - Gaussian noise
    - snapping to discrete grid
    - hard bounds enforcement

    candidate: numpy array [c1_x, c1_y, ...]
    """

    mutant = candidate.copy()

    for i, key in enumerate(PARAM_KEYS):

        if np.random.rand() < mutation_rate:
            # --- 1. Gaussian perturbation ---
            mutant[i] += mutation_strength * np.random.randn()

            # --- 2. Snap to allowed grid ---
            mutant[i] = np.round(mutant[i] / grid) * grid

            # --- 3. Enforce bounds ---
            mutant[i] = np.clip(mutant[i], -limit, +limit)

    return mutant

def elitist_replacement(population, fitnesses, new_population):
    """
    Keep the best individual from the old population
    and insert it into the new population.
    """
    best_idx = np.argmax(fitnesses)
    elite = population[best_idx]

    # Replace a random child (or worst, if you track it)
    replace_idx = np.random.randint(len(new_population))
    new_population[replace_idx] = elite

    return new_population

def run_genetic_algorithm():
    # --- 1. Settings ---
    POP_SIZE = 5
    GENERATIONS = 25
    GRID = 0.1e-6
    LIMIT = 0.5e-6
    # Create initial random population (random values between -0.5e-6 and 0.5e-6)
    def create_random_individual():
        """Create a random individual with grid-snapped values"""
        n_params = len(PARAM_KEYS)
        n_steps = int(2 * LIMIT / GRID)  # Number of grid points from -limit to +limit
        random_steps = np.random.randint(-n_steps//2, n_steps//2 + 1, size=n_params)
        return random_steps * GRID
    
    population = [create_random_individual() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        print(f"\n=== GENERATION {gen} ===")
        
        # --- 2. Evaluation Step ---
        fitness_scores = []
        for i, individual in enumerate(population):
            # Run simulation for this individual
            run_name = f"1gen_{gen}_ind_{i}"
            score = evaluate_fitness(individual, run_name)
            fitness_scores.append(score)
            shutil.rmtree(run_name+".out")  # delete folder after finishing

        # Track progress
        best_score = max(fitness_scores)
        print(f"Best Score: {best_score:.4f}")
        
        # --- 3. Breeding Step (Selection, Crossover, Mutation) ---
        new_population = []
        
        # Fill the new population
        while len(new_population) < POP_SIZE:
            # A. Select Parents
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            # B. Crossover
            child = uniform_crossover(parent1, parent2)
            
            # C. Mutate
            child = mutate(child)
            
            new_population.append(child)

        # --- 4. Elitism Step ---
        # Ensure the absolute best from the old generation survives
        population = elitist_replacement(population, fitness_scores, new_population)
        
        # Save the best individual to a file so you don't lose progress
        best_idx = np.argmax(fitness_scores)
        np.save(f"best_candidate_gen_{gen}.npy", population[best_idx])



# run_mumax3(MumaxScript,name)
# read_mumax3_ovffiles(output)
# visualise(output)
fft(output)
# print("Starting Genetic Algorithm")
# run_genetic_algorithm()

