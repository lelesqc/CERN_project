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
        plt.tight_layout()

    if mode == "tune":
        data = np.load("tune_analysis/fft_results.npz")
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        tunes_list = data['tunes_list']
        spectra = data['spectra']
        freqs_list = data['freqs_list']

        mask = tunes_list > 0
        tunes_pos = tunes_list[mask]

        idx_sorted = np.argsort(tunes_pos)[::-1]
        tunes_sorted = tunes_pos[idx_sorted]

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(tunes_sorted)), tunes_sorted, 'o-')
        plt.xlabel("Indice (tune decrescente)")
        plt.ylabel("Tune")

        # Scrivi il valore del tune su ogni punto
        for i, t in enumerate(tunes_sorted):
            plt.text(i, t, f"{t:.3f}", ha='center', va='bottom', fontsize=8, rotation=45)

        plt.tight_layout()


    plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
