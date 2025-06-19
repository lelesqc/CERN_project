import numpy as np
import matplotlib.pyplot as plt
import sys
import os
if os.environ.get("PHASE_SPACE", "0") == "1":
    import params_fixed as par
else:
    import params as par

def plot():
    if os.environ.get("PHASE_SPACE", "0") == "1":
        a_val = par.a_lambda(0)
        omega_val = par.omega_lambda(0)

        a_str = f"{a_val:.3f}"
        omega_str = f"{omega_val:.2f}"
        phase_space = np.load(f"action_angle/cartesian_a{a_str}_omega{omega_str}.npz")

        x = phase_space['x']
        y = phase_space['y']
        
    else:
        data = np.load("action_angle/cartesian.npz")

        x = data['x']
        y = data['y']

    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot()