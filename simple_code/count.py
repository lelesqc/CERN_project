import numpy as np
import params as par
import os
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

poincare_mode = sys.argv[1]

# -------- FFT ----------

def count_particles(data):
    x = data['x']
    y = data['y']

    z = x - 1j * y

    print(z.shape, par.n_steps)

    max_freq_list = []

    n_particles = len(z) // par.n_steps
    for i in range(n_particles):
        start = i * par.n_steps
        end = (i + 1) * par.n_steps
        z_part = z[start:end]
        freq_ampl = np.abs(fft(z_part))
        #freq = fftfreq(len(z_part), d=par.dt)

        max_freq = np.max(freq_ampl)
        max_freq_list.append(max_freq)

    plt.hist(max_freq_list, bins=n_particles, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Max Frequency")
    plt.ylabel("Density")
    plt.title("Distribution of Max Frequencies")
    plt.grid()
    plt.xlim(0, np.max(max_freq_list) * 1.1)
    plt.ylim(0, np.max(np.histogram(max_freq_list, bins=len(z))[0]) * 1.1)
    plt.tight_layout()
    plt.show()    
    
if __name__ == "__main__":
    is_phase_space = os.environ.get("PHASE_SPACE", "0") == "1"

    if is_phase_space and poincare_mode == "none":
        data = np.load(f"action_angle/cartesian_full_steps.npz")

    elif is_phase_space and poincare_mode != "none":    
        data = np.load(f"action_angle/cartesian_a{par.a_lambda(0):.3f}_nu{par.omega_lambda(0)/par.omega_s:.2f}.npz")

    elif not is_phase_space:
        data = np.load(f"action_angle/cartesian_full_steps.npz")

    count_particles(data)




