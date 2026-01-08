import numpy as np
import matplotlib.pyplot as plt


# Physical parameters

kB = 1.38e-23       
gamma = 1.9e-8      
k = 1e-6               

Temp_msd = [100, 300, 600]                 # for Figure 3
Temp_var = [50, 100, 300, 600, 1000] # for Figure 4

# Simulation parameters

dt = 1e-4
N = 10000
t = np.arange(N) * dt
n_traj = 1000


n_T = len(Temp_msd)
x = np.zeros((n_T, n_traj, N))

for i, T in enumerate(Temp_msd):
    for j in range(n_traj):
        for n in range(N - 1):
            eta = np.random.randn()
            x[i, j, n+1] = (
                x[i, j, n]
                - (k / gamma) * x[i, j, n] * dt
                + np.sqrt(2 * kB * T / gamma * dt) * eta
            )

# MSD calculation
x0 = x[:, :, 0][:, :, None]
msd = np.mean((x - x0)**2, axis=1)


N_var = 100000
x_var = np.zeros((len(Temp_var), N_var))

for i, T in enumerate(Temp_var):
    for n in range(N_var - 1):
        eta = np.random.randn()
        x_var[i, n+1] = (
            x_var[i, n]
            - (k / gamma) * x_var[i, n] * dt
            + np.sqrt(2 * kB * T / gamma * dt) * eta
        )

variances = np.var(x_var, axis=1)

#Plot
plt.figure(figsize=(14, 5))

#  Figure 3
plt.subplot(1, 2, 1)
for i, T in enumerate(Temp_msd):
    plt.plot(t, msd[i], label=f"T = {T} K")
    plt.axhline(kB * T / k, linestyle="--", alpha=0.6)

plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Mean-square displacement", fontsize=14)
plt.title("Figure 3: MSD relaxation", fontsize=14)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(loc=(0.8, 0.45))
plt.tick_params(labelsize=12)

# Figure 4
plt.subplot(1, 2, 2)
plt.plot(Temp_var, variances, 'o', label="Simulation")
plt.plot(Temp_var, kB * np.array(Temp_var) / k, '--', label="Theory")

plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Position variance", fontsize=14)
plt.title("Figure 4: Equilibrium variance vs temperature", fontsize=14)
plt.legend()
plt.tick_params(labelsize=12)

plt.tight_layout()
plt.show()

plt.show()
