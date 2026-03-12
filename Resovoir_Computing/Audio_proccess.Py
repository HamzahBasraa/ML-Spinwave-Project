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

    # header
    f.write("// creating a device for non-linear spin wave dynamics\n\n")

    # grid
    f.write("// Grid and Cell Size Setup\n")
    f.write("SetGridsize(200, 200, 1)\n")
    f.write("SetCellsize(10e-9, 10e-9, 10e-9)\n\n")

    # measurement square positions
    f.write("// Square positions\n")
    f.write("x1 := 500e-9\n")
    f.write("y1 := 400e-9\n")
    f.write("x2 := 0e-9\n")
    f.write("y2 := 0e-9\n")
    f.write("x3 := -500e-9\n")
    f.write("y3 := -400e-9\n\n")

    # square sizes
    f.write("sq_size := 50e-9\n")
    f.write("sq_size2 := 100e-9\n\n")

    # geometry
    f.write("// Device geometry\n")
    f.write("sphere := cylinder(1.5e-6, 10e-9).transl(0,0,0)\n")
    f.write("input := cuboid(0.25e-6, 0.5e-6, 10e-9).transl(-0.875e-6,0,0)\n\n")

    f.write("dodge := cuboid(sq_size, sq_size, 10e-9).transl(350e-9,-250e-9,0)\n")
    f.write("dodge2 := cuboid(sq_size, sq_size, 10e-9).transl(-407e-9,146e-9,0)\n")
    f.write("dodge3 := cuboid(sq_size, sq_size, 10e-9).transl(116e-9,400e-9,0)\n")
    f.write("dodge4 := cuboid(sq_size, sq_size, 10e-9).transl(-209e-9,-349e-9,0)\n")
    f.write("dodge5 := cuboid(sq_size, sq_size, 10e-9).transl(254e-9,234e-9,0)\n\n")

    f.write("device := input.add(sphere).sub(dodge).sub(dodge2).sub(dodge3).sub(dodge4).sub(dodge5)\n\n")

    # measurement regions
    f.write("// Measurement regions\n")
    f.write("measure1 := cuboid(sq_size2, sq_size2, 10e-9).transl(x1,y1,0)\n")
    f.write("measure2 := cuboid(sq_size2, sq_size2, 10e-9).transl(x2,y2,0)\n")
    f.write("measure3 := cuboid(sq_size2, sq_size, 10e-9).transl(x3,y3,0)\n\n")

    # geometry setup
    f.write("setgeom(device)\n")
    f.write("saveas(geom, \"device\")\n\n")

    # regions
    f.write("// region definitions\n")
    f.write("defregion(1, device)\n")
    f.write("defregion(2, input)\n")
    f.write("defregion(3, measure1)\n")
    f.write("defregion(4, measure2)\n")
    f.write("defregion(5, measure3)\n\n")

    # material parameters
    f.write("// YIG parameters\n")
    f.write("Msat = 1.4e5\n")
    f.write("Aex = 3.5e-12\n")
    f.write("alpha = 2e-4\n\n")

    f.write("// Permalloy region\n")
    f.write("Msat.SetRegion(6, 8e5)\n")
    f.write("Aex.SetRegion(6, 1.3e-11)\n")
    f.write("alpha.SetRegion(6, 0.013)\n\n")

    f.write("Msat.SetRegion(1, 1.45e5)\n")
    f.write("Aex.SetRegion(1, 3.5e-12)\n")
    f.write("alpha.SetRegion(1, 2e-4)\n\n")

    # initial magnetisation
    f.write("// initial state\n")
    f.write("m = uniform(1, .1, 0)\n")
    f.write("B_ext = vector(0,0,0.2)\n")
    f.write("relax()\n")
    f.write("save(m)\n\n")

    # autosave settings
    f.write("// autosave\n")
    f.write("autosave(m, 100e-12)\n")
    f.write("tableautosave(5e-12)\n\n")

    # signal injection section
    f.write("// input signal injection\n")

    for val in signal:
        f.write(f"B_ext.setRegion(2, vector(0, 0, {val} + 0.02))\n")
        f.write(f"run({dt})\n")

print("Generated MuMax script successfully.")