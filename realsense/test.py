import numpy as np

npzfile = np.load('test_2_depth.npz')

for k in npzfile.files:
     print(k)

print(len(npzfile.files))