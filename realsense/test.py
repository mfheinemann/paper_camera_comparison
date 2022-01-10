import numpy as np

npzfile = np.load('test_depth.npz')

for k in npzfile.files:
     print(k)