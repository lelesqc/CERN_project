import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

def plot(mode):
    if mode == "phasespace":
        data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = data['x']
        y = data['y']

        plt.scatter(x, y)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    if mode == "tune":
        data = np.load("tune_analysis/fft_results.npz")
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = action_angle['x']
        actions = action_angle['actions_list']

        x_init = x[0, :]            
        actions_init = actions[0, :] 
        
        mask = x_init > 0

        spectra = data['spectra']
        freqs_list = data['freqs_list']
        tunes_list = data['tunes_list']

        actions_init_pos = actions_init[mask]
        tunes_list_pos = tunes_list[mask]

        for i in range(len(actions_init_pos)):
            print(f"action: {actions_init_pos[i]}, tune: {tunes_list_pos[i]}")
        print(np.max(tunes_list))

        plt.scatter(actions_init, tunes_list)
        plt.tight_layout()
        plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
