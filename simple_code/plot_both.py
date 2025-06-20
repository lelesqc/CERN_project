import numpy as np
import matplotlib.pyplot as plt
import params as par
import params_fixed as par_fixed
import glob

# phase space parameters
a_phase = 0.050 
nu_phase = 0.80

# evolution parameters
a_start = 0.025  
a_end = 0.050    
nu_start = 0.90
nu_end = 0.80 

def print_available_files():
    print("File disponibili in 'action_angle/':")
    for f in sorted(glob.glob("action_angle/cartesian*.npz")):
        print("  ", f)

def plot_both():
    phase_file = f"action_angle/cartesian_a{a_phase:.3f}_nu{nu_phase:.2f}.npz"
    evol_file = f"action_angle/cartesian_a{a_start:.3f}-{a_end:.3f}_nu{nu_start:.2f}-{nu_end:.2f}.npz"

    data_phase = np.load(phase_file)
    data_evol = np.load(evol_file)

    plt.figure(figsize=(7,7))
    plt.scatter(data_phase['x'], data_phase['y'], s=10, label="Phase space", alpha=0.7)
    plt.scatter(data_evol['x'], data_evol['y'], s=10, label="Evolution", alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(rf"a = {a_start:.3f} - {a_end:.3f}, $\nu_m$ = {nu_start:.2f} - {nu_end:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        print_available_files()
    else:
        plot_both()
