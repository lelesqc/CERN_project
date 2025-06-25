import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

def plot_both():
    phase_file = f"action_angle/a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz"
    evol_file = f"../code/action_angle/last_a0.025-0.050_nu0.90-0.80.npz"

    data_phase = np.load(phase_file)
    data_evol = np.load(evol_file)

    plt.figure(figsize=(7,7))
    plt.scatter(data_phase['x'], data_phase['y'], s=10, label="Phase space", alpha=0.7)
    plt.scatter(data_evol['x'], data_evol['y'], s=10, label="Evolution", alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_both()