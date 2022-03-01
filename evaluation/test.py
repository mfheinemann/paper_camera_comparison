import numpy as np
import math
gt = 2.0
angle = np.radians(20)
w = 225

t = np.linspace( - w/2 * math.sin(angle),  + w/2 * math.sin(angle), num=91)
t2 = np.array([- w/2 * math.sin(angle),  + w/2 * math.sin(angle)])

st_d = np.std(t)
st_d2 = np.std(t2)

m1 = np.mean(t)
m2 = np.mean(t2)

print(st_d)
print(st_d2)

print(m1)
print(m2)

x = -1
y = 1

print(np.sign(x)*y)

print(math.sin(angle))
print(math.cos(np.deg2rad(90)-angle))