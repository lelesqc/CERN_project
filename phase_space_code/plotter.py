import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

def plot(mode):
    """if mode == "phasespace":
        data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = data['x']
        y = data['y']

        plt.figure(figsize=(7,7))
        plt.scatter(x, y, s=3, label=r"Phase Space for final distr.", alpha=1.0)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=20)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.legend(fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()"""

    if mode == "phasespace":
        data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
        tune_analysis = np.load(f"tune_analysis/fft_results.npz")

        tunes_list = tune_analysis['tunes_list']

        x = data['x']
        y = data['y']

        print(tunes_list.size)

        plt.figure(figsize=(7,7))
        scatter = plt.scatter(x, y, s=3, c=tunes_list, cmap='viridis', alpha=1.0)
        plt.colorbar(scatter, label='Tune')
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=20)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.title("Phase Space colored by Tune", fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

    if mode == "tune":
        data = np.load("tune_analysis/fft_results.npz")
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = action_angle['x']
        actions = action_angle['actions_list']

        x_init = x[0, :]            
        actions_init = actions[0, :] 
        
        spectra = data['spectra']
        freqs_list = data['freqs_list']
        tunes_list = data['tunes_list']

        actions_init_pos = actions_init
        tunes_list_pos = tunes_list

        print(x.shape)

        for i in range(len(actions_init_pos)):
            print(f"action: {actions_init_pos[i]:.3f}, tune: {tunes_list_pos[i]:.6f}")

        plt.scatter(x_init, tunes_list)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Tune", fontsize=20)
        plt.title("Tune vs X", fontsize=22)
        plt.grid(True)    
        plt.tight_layout()
        plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)