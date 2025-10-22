import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from subprocess import run, PIPE, STDOUT
from glob import glob
from os import path
from numpy import load
import os

file = 'task1.out/table.txt'
output = "test1.out"
name = "test1"
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


def visualise(random):
    # so what i want to do now is 
        # use nested for loops and to plot a the graph
        # an array should store the intesity of y at each point in the matrix
        # the matrix's coordinates ie [x][y] will store the normalised intensity of y at each point

    files = sorted(glob(output + '/*.npy')) #ensures theyre in order 
        
    for f in files: #for each file 
        a = np.load(f) #load the numpy arrays  in to python
        #print(a.shape)
        My_initial = a[1,0,:,:] #accesing just my accross the whole grid this is a matrix of values we want to use

        My = (My_initial-My_initial.min())/(My_initial.max()-My_initial.min())

        plt.imshow(My, origin='lower', cmap='RdBu')
        plt.title("My across the film")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="My")
        plt.show()
                

#read_mumax3_ovffiles(output)
# run_mumax3(MumaxScript,name)
read_mumax3_ovffiles(output)
visualise(output)
