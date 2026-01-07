import numpy as np


# Test function

a = np.array([1,2,3,4,5])

n_a = len(a)

n_b = 4
# Extend Test Function to n*m

nm = np.tile(a, (n_b,1))
print(nm)