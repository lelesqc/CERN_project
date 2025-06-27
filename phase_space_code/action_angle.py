import os
import sys
import numpy as np
from tqdm.auto import tqdm

import params as par
import functions as fn


def run_action_angle(mode):
    data = np.load(f"integrator/evolved_qp_{mode}.npz")

    q = data['q']
    p = data['p']

    n_steps, n_particles = q.shape

    actions_list = np.zeros((n_steps, n_particles))
    theta_list = np.zeros((n_steps, n_particles))
    sign_list = np.zeros((n_steps, n_particles))
    
    for j in tqdm(range(n_particles)):
        for i in range(n_steps):
            h_0 = fn.H0_for_action_angle(q[i, j], p[i, j])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

            if 0 < kappa_squared < 1:
                Q = (q[i, j] + np.pi) / par.lambd
                P = par.lambd * p[i, j]

                action, theta = fn.compute_action_angle(kappa_squared, P)
                actions_list[i, j] = action
                theta_list[i, j] = theta
                sign_list = np.sign(q[i, j]-np.pi)  

    actions_list = np.array(actions_list)
    theta_list = np.array(theta_list)

    x = np.sqrt(2 * np.array(actions_list)) * np.cos(theta_list)
    y = - np.sqrt(2 * np.array(actions_list)) * np.sin(theta_list) * np.array(sign_list)

    return x, y, actions_list

# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    x, y, actions_list = run_action_angle(mode)

    output_dir = "action_angle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
    np.savez(file_path, x=x, y=y, actions_list=actions_list)
