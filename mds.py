# Multi-dimensional Scaling analyis of NBA data

import numpy as np

# read in data from csv to structured numpy array
data = np.genfromtxt('Data/stats.csv', delimiter=',', dtype=None)

categories =  data[0]

names = []
for row in data:
    names.append(row[0])
    row = row[1:]

# delete names and titles from the data
data  = np.delete(data, 0, 0)
data = np.delete(data, 0, 1)
print(data)

# MDS:
# 1. Create squared proximity matrix
# 2. Apply double centering
# 3. Determine the m largest eigenvalues
# 4. matrix multiplication

datasquared = data * data
print(datasquared)





