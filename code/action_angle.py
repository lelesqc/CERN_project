import os
import sys
import numpy as np
from tqdm.auto import tqdm

import params as par
import functions as fn


def run_action_angle(poincare_mode):
    data = np.load(f"integrator/evolved_qp_{poincare_mode}.npz")

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
    poincare_mode = sys.argv[1]
    x, y = run_action_angle(poincare_mode)

    a_start = par.a_lambda(par.T_percent)
    omega_start = par.omega_lambda(par.T_percent)
    a_end = par.a_lambda(par.T_tot)
    omega_end = par.omega_lambda(par.T_tot)
    
    a_start_str = f"{a_start:.3f}"
    omega_start_str = f"{omega_start:.2f}"
    a_end_str = f"{a_end:.3f}"
    omega_end_str = f"{omega_end:.2f}"

    str_title = f"a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.2f}-{float(omega_end_str)/par.omega_s:.2f}"

    output_dir = "action_angle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{poincare_mode}_{str_title}.npz")
    np.savez(file_path, x=x, y=y)