import re
import matplotlib.pyplot as plt


log_file = "project5.out"


totals = []


with open(log_file, "r") as f:
    for line in f:

        match = re.search(r"Total=([0-9.eE+-]+)", line)
        if match:
            totals.append(float(match.group(1)))

if not totals:
    print("No Total= values found.")
    exit()

x_values = list(range(1, len(totals) + 1))


plt.figure()
plt.scatter(x_values, totals)
plt.xlabel("Run Order")
plt.ylabel("Total Value")
plt.title("Total Values per Run")
plt.grid(True)

plt.show()
