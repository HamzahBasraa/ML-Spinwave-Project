import pandas as pd
from glob import glob
from numpy import load
from subprocess import run, PIPE, STDOUT
from os import path
import time
import numpy as np
import matplotlib.pyplot as plt
import Audio_initial
import Audio_proccess


detectors = {
    "measure1": {"cx": 100e-9,  "cy": 100e-9,  "w": 50e-9, "h": 50e-9},
    "measure2": {"cx": -100e-9, "cy": 200e-9,  "w": 50e-9, "h": 50e-9},
    "measure3": {"cx": 0.0,     "cy": -200e-9, "w": 50e-9, "h": 50e-9},
}
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

def extract_detector_fft(output_path, dt=100e-12, cellsize=10e-9):

    import numpy as np
    from glob import glob

    # -----------------------
    # Load time-series data
    # -----------------------
    files = sorted(glob(output_path + '/m*.npy'))
    
    sample = np.load(files[0])
    n_components, n_z, n_y, n_x = sample.shape
    n_time = len(files)

    data_5d = np.zeros((n_time, n_components, n_z, n_y, n_x))

    data_5d[0] = sample

    for i, f in enumerate(files[1:], start=1):
        data_5d[i] = np.load(f)

    # use My component
    z_middle = int(n_z / 2)
    my_t = data_5d[:, 1, z_middle, :, :]

    # remove dc offset so the signal is centred
    my_t = my_t - np.mean(my_t, axis=0)

    # discard early transient part of simulation
    discard = int(5e-9 / dt)
    my_t = my_t[discard:]
    n_time = my_t.shape[0]

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
    # detector positions from mumax script
    # -----------------------
    detectors = {
        "measure1": {"cx": 500e-9,  "cy": 400e-9,  "w": 50e-9, "h": 50e-9},
        "measure2": {"cx": 0e-9, "cy": 0e-9,  "w": 50e-9, "h": 50e-9},
        "measure3": {"cx": -500e-9,     "cy": -400e-9, "w": 50e-9, "h": 50e-9},
    }

    results = {}

    # -----------------------
    # Extract detector signals
    # -----------------------
    #this converts co ordinates from mumax(where 0,0 is there centre) to python (where 0,0 is top left)
    for name, det in detectors.items():

        cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]

        ix_min = phys_to_ix(cx - w / 2)
        ix_max = phys_to_ix(cx + w / 2)
        iy_min = phys_to_iy(cy - h / 2)
        iy_max = phys_to_iy(cy + h / 2)

        ix_min = max(ix_min, 0)
        iy_min = max(iy_min, 0)
        ix_max = min(ix_max, n_x - 1)
        iy_max = min(iy_max, n_y - 1)

        # --------- changed part ----------
        # average magnetisation inside detector square
        # this gives detector signal vs time
        signal = np.mean(
            my_t[:, iy_min:iy_max+1, ix_min:ix_max+1],
            axis=(1,2)
        )

        # sample 20 evenly spaced time points
        # this compresses the signal into reservoir features
        idx = np.linspace(0, len(signal)-1, 20).astype(int)
        samples = signal[idx]

        results[name] = samples

    # combine all detector samples into one feature vector
    features = np.concatenate([
        results["measure1"],
        results["measure2"],
        results["measure3"]
    ])

    print("feature vector length:", len(features))

    plt.figure(figsize=(6,6))
    plt.imshow(my_t[0], origin='lower', cmap='gray')

    for name, det in detectors.items():

        cx, cy = det["cx"], det["cy"]

        ix = phys_to_ix(cx)
        iy = phys_to_iy(cy)

        plt.scatter(ix, iy, s=200, label=name)

    plt.legend()
    plt.title("Detector locations on magnetisation grid")
    plt.show()
    return features


def train_reservoir(dataset_path, mumax_script):

    x = []   # feature vectors
    y = []   # labels

    wav_files = glob(dataset_path + "/*.wav")

    print("Total audio samples:", len(wav_files))

    for wav in wav_files:

        print("Processing:", wav)
        # 1 Generate envelope--
        envelope = Audio_initial.create_envelope(wav, 5, 0.01, "input_signal.txt")

        # 2 Write MuMax input file
        Audio_proccess(envelope)

        # 3 Run MuMax simulation
        run_mumax3(mumax_script, "simulation")
        # 4 Extract reservoir features
        features = extract_detector_fft("simulation.out")

        x.append(features)

        # label is the spoken digit (first character of filename)
        label = int(path.basename(wav)[0])
        y.append(label)

    x = np.array(x)
    y = np.array(y)

    print("Dataset shape:",  x.shape)

    # save dataset so you don't need to rerun simulations
    np.save("reservoir_features.npy", x)
    np.save("labels.npy", y)

    return  x, y




output_path = "non-linear.out"
output = "non-linear.out"
name = "non-linear"
filepath = "non-linear.mx3"
with open(filepath, 'r') as file: #script 
    MumaxScript = file.read()
# run_mumax3(MumaxScript,name)
# read_mumax3_ovffiles(output)
# extract_detector_fft(output_path, dt=100e-12, cellsize=10e-9)
Audio_initial.create_envelope("free-spoken-digit-dataset/recordings/0_george_21.wav", 5, 0.01, "input_signal.txt")
