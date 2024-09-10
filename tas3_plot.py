from a2 import jump_on_circle

import random
import matplotlib.pyplot as plt

x = 0.5; y = 0.5; radius = 0.5
xs = []; ys = []
random.seed(23)
for i in range(100):
    (px, py) = jump_on_circle(x, y, radius)
    xs.append(px)
    ys.append(py)

fig, ax = plt.subplots()
ax.plot(xs, ys, '.', markersize=4)
ax.set_aspect('equal')
circ = plt.Circle((x, y), radius, color='r', fill=False)
ax.add_patch(circ)
plt.savefig('jump-on-circle-test.png', dpi=300)
