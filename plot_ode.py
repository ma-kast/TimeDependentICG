import numpy as np

import matplotlib.pyplot as plt


n= 100
f_tumor = 0.001*( 1+5)
f_env = 0.001
k_tumor = 0.006*(1-0.5)
k_env = 0.006
a= 20

n_x = 100

t= np.linspace(0, a, n_x)
t_after = np.linspace(a, 50, 2*n_x)
t_special = np.linspace(0,a,6)

def ode_sol(t, a, f, k ):

    return -np.exp(-k*t) * f*(1+a *k +np.exp(k*t) * (-1- a*k +k*t))/k**2



def ode_sol_cont(t, a, u_t, k ):

    return np.exp(-k*(t-a)) *u_t

c_tumor = ode_sol(t, a, f_tumor, k_tumor)
c_tumor_cont = ode_sol_cont(t_after, a, c_tumor[-1], k_tumor )

c_env = ode_sol(t, a, f_env, k_env)
c_env_cont = ode_sol_cont(t_after, a, c_env[-1], k_env )

plt.figure()

plt.plot(t, c_tumor,'b', label = "tumor")
plt.plot(t_after, c_tumor_cont,'b')
plt.plot(t, c_env,'r', label = "environment")
plt.plot(t_after, c_env_cont,'r')
plt.vlines(t_special[1:], 0, 12, colors='k', label = "observation times")
plt.xlabel("t")
plt.ylabel("ICG concentration")
plt.ylim([0,np.max(c_tumor)+0.1])
plt.legend()
plt.savefig("ode_sol.png")

print (ode_sol(t_special,a,f_tumor, k_tumor))
print (ode_sol(t_special,a,f_env, k_env))