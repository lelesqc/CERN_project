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

    action_list = []
    theta_list = []
    sign_list = []

    for i in tqdm(range(len(q))):
        h_0 = fn.H0_for_action_angle(q[i], p[i])
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        if 0 < kappa_squared < 1:
            Q = (q[i] + np.pi) / par.lambd
            P = par.lambd * p[i]

            action, theta = fn.compute_action_angle(kappa_squared, P)
            action_list.append(action)
            theta_list.append(theta)
            sign_list.append(np.sign(q[i]-np.pi))

    action_list = np.array(action_list)
    theta_list = np.array(theta_list)

    x = np.sqrt(2 * np.array(action_list)) * np.cos(theta_list)
    y = - np.sqrt(2 * np.array(action_list)) * np.sin(theta_list) * np.array(sign_list)

    return x, y


# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    x, y = run_action_angle(mode)

    output_dir = "action_angle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
    np.savez(file_path, x=x, y=y)
