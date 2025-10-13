import matplotlib.pyplot as plt

y_array = []
t_array = []
f = open ('task1.out/table.txt')

for line in f:
    if '#' in line:
        continue 
    else:
        sorty = line.split('\t')
        y_array.append(float(sorty[2]))
        t_array.append(float(sorty[0]))

plt.plot(t_array,y_array)
plt.show()


