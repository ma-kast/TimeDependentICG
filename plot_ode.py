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
t_special = np.linspace(0,a*1.5,6)

def ode_sol(t, a, f, k ):

    return -np.exp(-k*t) * f*(1+a *k +np.exp(k*t) * (-1- a*k +k*t))/k**2

def ode_sol_tanh(t, k, a_, c_max, p ):

    return c_max * (np.exp(-k*t) *( np.tanh(a_* t)-p) +p)

def ode_sol_cont(t, a, u_t, k ):

    return np.exp(-k*(t-a)) *u_t

c_tumor = ode_sol(t, a, f_tumor, k_tumor)
c_tumor_cont = ode_sol_cont(t_after, a, c_tumor[-1], k_tumor )

c_env = ode_sol(t, a, f_env, k_env)
c_env_cont = ode_sol_cont(t_after, a, c_env[-1], k_env )

plt.figure()

print(t_special)
print (t_special[1:4],ode_sol(t_special[1:4],a,f_env, k_env))
print (t_special[4:],ode_sol_cont(t_special[4:],a,c_env[-1], k_env))
print (t_special[1:4],ode_sol(t_special[1:4],a,f_tumor, k_tumor))
print (t_special[4:],ode_sol_cont(t_special[4:],a,c_tumor[-1], k_tumor))


plt.plot(t, c_tumor,'b', label = "tumor")
plt.plot(t_after, c_tumor_cont,'b')
plt.plot(t, c_env,'r', label = "environment")
plt.plot(t_after, c_env_cont,'r')
plt.vlines(t_special[1:], 0, 12, colors='k', label = "observation times")
plt.plot(t_special[1:4],ode_sol(t_special[1:4],a,f_tumor, k_tumor),"x", color= "green")
plt.plot(t_special[4:],ode_sol_cont(t_special[4:],a,c_tumor[-1], k_tumor),"x", color= "green")
plt.xlabel("t")
plt.ylabel("ICG concentration")
plt.ylim([0,np.max(c_tumor)+0.1])
plt.legend()
plt.savefig("ode_sol2.png")



t_all = np.vstack((t.reshape(-1,1), t_after.reshape(-1,1)))


k_env = 0.03
k_tumor = k_env/2
c_env = ode_sol_tanh(t_all, k_env, 0.3, 1, 0.01 )
c_tumor = ode_sol_tanh(t_all, k_tumor, 0.3, 1.5, 0.8 )


plt.figure()

plt.plot(t_all, c_tumor,'b', label = "tumor")

plt.plot(t_all, c_env,'r', label = "environment")
plt.vlines(t_special[1:], 0, 12, colors='k', label = "observation times")
plt.xlabel("t")
plt.ylabel("ICG concentration")
plt.ylim([0,np.max(c_tumor)+0.1])
plt.legend()
plt.savefig("ode_sol_tanh.png")