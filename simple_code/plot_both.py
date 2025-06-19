import numpy as np
import matplotlib.pyplot as plt
import params as par
import params_fixed as par_fixed

def plot_both():
    # --- phase space ---
    a_val = par_fixed.a_lambda(0)
    omega_val = par_fixed.omega_lambda(0)
    a_str = f"{a_val:.3f}"
    omega_str = f"{omega_val:.2f}"
    phase_file = f"action_angle/cartesian_a{a_str}_nu{float(omega_str)/par_fixed.omega_s:.2f}.npz"

    # --- evolution ---
    a_start = par.a_lambda(par.T_percent)
    omega_start = par.omega_lambda(par.T_percent)
    a_end = par.a_lambda(par.T_tot)
    omega_end = par.omega_lambda(par.T_tot)
    a_start_str = f"{a_start:.3f}"
    omega_start_str = f"{omega_start:.2f}"
    a_end_str = f"{a_end:.3f}"
    omega_end_str = f"{omega_end:.2f}"
    evol_file = f"action_angle/cartesian_a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.2f}-{float(omega_end_str)/par.omega_s:.2f}.npz"

    data_phase = np.load(phase_file)
    data_evol = np.load(evol_file)

    plt.figure(figsize=(7,7))
    plt.scatter(data_phase['x'], data_phase['y'], s=10, label="Phase space", alpha=0.7)
    plt.scatter(data_evol['x'], data_evol['y'], s=10, label="Evolution", alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(r"a = 0.0 - 0.05, $\nu_m$ = 0.9 - 0.8")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_both()
