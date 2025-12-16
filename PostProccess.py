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
                "c10_y": -4e-7}


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
    dt = 200e-12 #known time step should work to change this so it is extracted from the script

    # Create empty 5D array
    data_5d = np.zeros((n_time, n_components, n_z, n_y, n_x))
        
    for i,f in enumerate(files): #for each file 
        data_5d[i] = np.load(f) #this is the 5d array
    My_initial = data_5d[:,1,0,:,:] #3d array with time, x , y 

    my_t = My_initial - My_initial[0, :, :] #only containts oscillating data now as we removed the intial state
    

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


def extract_detector_fft(output_path, dt=200e-12, cellsize=5e-9):
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
    my_t = data_5d[:, 1, 0, :, :]

    # Remove DC component (better than subtracting first frame)
    my_t = my_t - np.mean(my_t, axis=0)

    # -----------------------
    # FFT
    # -----------------------
    fast_transform = np.fft.fft(my_t, axis=0)
    freqs = np.fft.fftfreq(n_time, dt)

    f_drive = 2.6e9  # 2.6 GHz
    idx = np.argmin(np.abs(freqs - f_drive))

    amplitude = np.abs(fast_transform[idx])

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





                
run_mumax3(MumaxScript,name)
read_mumax3_ovffiles(output)
visualise(output)
fft(output)
# update_parameters(file_path,position_dict)

# extract_detector_fft(output)

# differential evolution bumps library 
# 0.6 for crossover and mutation rate reference value 