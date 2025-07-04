import numpy as np
import matplotlib.pyplot as plt
import params as par

def tune_analysis():
    data = np.load("tune_analysis/fft_results.npz")
    action_angle = np.load(f"action_angle/tune_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

    tunes_list = data['tunes_list']
    freqs_list = data['freqs_list']
    amplitudes = data['amplitudes']

    x = action_angle['x']
    x_init = x[0, :]

    # Combina le maschere
    mask_combined = (x_init > 0) & (x_init < 11)
    
    amplitudes_masked = amplitudes[mask_combined]
    freqs_list_masked = freqs_list[mask_combined]
    tunes_masked = tunes_list[mask_combined]

    # Applica la maschera sui tune
    mask3 = tunes_masked < 0.8
    amplitudes_final = amplitudes_masked[mask3]
    freqs_final = freqs_list_masked[mask3]
    tunes_final = tunes_masked[mask3]


if __name__ == "__main__":
    tune_analysis()
