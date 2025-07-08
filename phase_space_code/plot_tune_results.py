import numpy as np
import matplotlib.pyplot as plt

import params as par

tune_data = np.load("tune_analysis/fft_results.npz")
action_angle_data = np.load(f"action_angle/tune_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

spectrum = tune_data['freqs_list']
spectrum = spectrum[0, :]

freq_mask = (spectrum > 0.3) & (spectrum < 1.2)
spectrum = spectrum[freq_mask]

amplitudes = tune_data['amplitudes']
amplitudes = amplitudes[:, freq_mask]

x = action_angle_data['x']
x_init = x[0, :]

x_mask = (x_init > 5) & (x_init < 12.7)  
x_init = x_init[x_mask]
amplitudes = amplitudes[x_mask, :]


print(x_init)

X, Y = np.meshgrid(spectrum, x_init)

plt.figure(figsize=(10, 8))
im = plt.pcolormesh(X, Y, amplitudes, cmap='viridis', shading='nearest',
                    edgecolors='black', linewidth=0.1)
plt.colorbar(im, label='Amplitude')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Initial X Position', fontsize=14)
plt.title('Amplitude vs Frequency and Initial Position', fontsize=16)
plt.tight_layout()
plt.show()