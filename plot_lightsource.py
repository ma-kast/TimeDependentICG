import numpy as np

import matplotlib.pyplot as plt


x_values = np.linspace(-2.5, 2.5, 100)
xx, yy = np.meshgrid(x_values, x_values)

light_source = 10 * np.exp(-(xx**2+ yy**2)/10)


plt.figure()

plt.imshow(light_source)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.savefig("light_source.png")

