import numpy as np

# SETTINGS
input_file = "input_signal.txt"
output_mx3 = "non-linear1.mx3"
dt = 5e-12
target_len = 4000

signal = np.loadtxt(input_file)

# force fixed length
if len(signal) > target_len:
    signal = signal[:target_len]
else:
    signal = np.pad(signal, (0, target_len - len(signal)))

# normalise
signal = signal / np.max(np.abs(signal))
signal = signal * 0.01  # 10 mT max

with open(output_mx3, "w") as f:

    f.write("// iteration xxx of ml \n\n")
    f.write("SetGridsize(128,128,1)\n")
    f.write("SetCellsize(5e-9,5e-9,5e-9)\n\n")

    f.write("Msat  = 8e5\n")
    f.write("Aex   = 1.3e-11\n")
    f.write("alpha = 0.02\n\n")

    f.write("m = uniform(1,0,0)\n\n")
    f.write("TableAdd(m)\n\n")
    f.write("autosave(m, 1e-10)\n\n")
    f.write("input := cuboid(200e-9,640e-9,5e-9).transl(-200e-9, 0,0)\n")
    f.write("defregion(2,input)\n")
    for i, val in enumerate(signal):
        start = i * dt
        end   = (i + 1) * dt

        # Update this line to use setRegion(2, ...)
        f.write(f'B_ext.setRegion(2, vector(0, 0, {val}))\n')
        f.write(f'run({dt})\n')

print("Generated non-linear.mx3 successfully.")