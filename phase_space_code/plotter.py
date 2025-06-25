import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

def plot(mode):
    phase_space = np.load(f"action_angle/a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

    x = phase_space['x']
    y = phase_space['y']

    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
