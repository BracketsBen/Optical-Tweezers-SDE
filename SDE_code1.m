import numpy as np
import matplotlib.pyplot as plt


# Physical parameters

kB = 1.38e-23
Temp = [100, 300, 600]  
gamma = 1.9e-8
k = 1e-6


# Simulation parameters

dt = 1e-4
N = 100000
t = np.arange(N) * dt


x = np.zeros((len(Temp), N))

# Euler–Maruyama

for i, T in enumerate(Temp):
    for n in range(N - 1):
        eta = np.random.randn()
        x[i, n+1] = (
            x[i, n]
            - (k / gamma) * x[i, n] * dt
            + np.sqrt(2 * kB * T / gamma * dt) * eta
        )

# Plotting

plt.figure(figsize=(10, 4))


for i, T in enumerate(Temp):
    plt.subplot(1, len(Temp), i + 1)
    plt.ylim(-5e-7, 5e-7)
    plt.plot(t, x[i])
    plt.axhline(0, linestyle='--', linewidth=1, color = 'black')
    plt.xlabel("Time (s)", fontsize = 16)
    plt.ylabel("Position (m)", fontsize = 16)
    plt.title(f"T = {T} K", fontsize = 16)
    plt.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()
