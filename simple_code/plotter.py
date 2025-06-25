import numpy as np
import matplotlib.pyplot as plt
import os
if os.environ.get("PHASE_SPACE", "0") == "1":
    import params_fixed as par
else:
    import params as par

def plot():
    if os.environ.get("PHASE_SPACE", "0") == "1":
        a_val = par.a_lambda(0)
        omega_val = par.omega_m

        a_str = f"{a_val:.3f}"
        omega_str = f"{omega_val:.2f}"
        phase_space = np.load(f"action_angle/cartesian_a{a_str}_nu{float(omega_str)/par.omega_s:.2f}.npz")

        x = phase_space['x']
        y = phase_space['y']
        
    else:
        a_start = par.a_lambda(par.T_percent)
        omega_start = par.omega_lambda(par.T_percent)
        a_end = par.a_lambda(par.T_tot)
        omega_end = par.omega_lambda(par.T_tot)

        a_start_str = f"{a_start:.3f}"
        omega_start_str = f"{omega_start:.2f}"
        a_end_str = f"{a_end:.3f}"
        omega_end_str = f"{omega_end:.2f}"

        data = np.load(f"action_angle/cartesian_a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.2f}-{float(omega_end_str)/par.omega_s:.2f}.npz")

        x = data['x']
        y = data['y']

    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot()