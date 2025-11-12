import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from subprocess import run, PIPE, STDOUT
from glob import glob
from os import path
from numpy import load
import os

file = 'task1.out/table.txt'
output = "scriptcopy.out"
name = "scriptcopy"
file_path = 'task1.txt'

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
    
    return table, fields


def fft(output_path):

    files = sorted(glob(output_path + '/*.npy')) #ensures theyre in order 
    n_components, n_z, n_y, n_x = np.load(files[0]).shape
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


    f_drive = 1e9
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

    files = sorted(glob(output_path + '/*.npy')) #ensures theyre in order 
        
    for f in files: #for each file 
        a = np.load(f) #load the numpy arrays  in to python

        first_index = 1 if a.shape[0] > 1 else 0

        My_first = a[first_index, 0, :, :]  # shape will be (32, 128)

        My = (My_first-My_first.min())/(My_first.max()-My_first.min())

        plt.imshow(My, origin='lower', cmap='jet')
        plt.title("My across the film")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="My")
        plt.show()
                
# run_mumax3(MumaxScript,name)
read_mumax3_ovffiles(output)
visualise(output)
fft(output)