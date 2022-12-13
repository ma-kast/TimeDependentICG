import numpy as np

import matplotlib.pyplot as plt

a=20

#ICG_curve = np.load("point_data.npy")
ICG_curve = np.load("point_data_no_ICG.npy")
t = [0,4,8,12, 16,20,24,28,32,36,40]
t_special = np.linspace(0,a,6)
plt.figure()
origin_shape = ICG_curve.shape

ICG_plot = np.zeros([origin_shape[0]+1, origin_shape[1]])

ICG_plot[1:] = ICG_curve

labels = ["far corner", "above tumor", "centre", "close side"]

plt.plot(t,ICG_plot,'x-', label =labels)

#plt.plot(t, ICG_curve,'x-', label = "without_env")
plt.vlines(t_special[1:], 0, np.max(ICG_curve.flatten())*1.01, colors='k', label = "observation times")
plt.xlabel("t")
plt.ylabel("Intensity")
#plt.ylim([0,np.max(c_tumor)+0.1])
plt.xlim([0,50])
plt.legend()
#plt.savefig("Intensity.png")
plt.savefig("Intensity_no_ICG.png")
