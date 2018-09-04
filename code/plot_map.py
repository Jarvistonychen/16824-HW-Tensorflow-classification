import matplotlib.pyplot as plt
import numpy as np

filename = 'q32_result.txt'
meanAP_array = []
test_interval = 400
with open(filename, 'r') as f:
    for line in f.readlines():
        if 'Obtained' in line:
            print line
            map_line = line.strip().split()
            meanAP = float(map_line[1])*100
            meanAP_array.append(meanAP)

plt.plot(test_interval*(1+np.arange(len(meanAP_array))),meanAP_array)
plt.ylabel('mAP')
plt.xlabel('#iterations')
plt.title('mAP of Q3.2')
plt.show()



