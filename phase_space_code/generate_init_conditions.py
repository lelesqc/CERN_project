import os
import sys
import numpy as np
from scipy.special import ellipk

import params as par
import functions as fn

def generate_grid(grid_lim, n_particles):
    X = np.linspace(-grid_lim, grid_lim, n_particles)
    Y = 0

    action, theta = fn.compute_action_angle_inverse(X, Y)

    kappa_squared_list = []
    Omega_list = []

    for act in action:
        h_0 = fn.find_h0_numerical(act)
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        kappa_squared_list.append(kappa_squared)
        Omega_list.append(np.pi / 2 * (par.A / ellipk(kappa_squared)))
        
    Q_list = []
    P_list = []

    for angle, freq, k2 in zip(theta, Omega_list, kappa_squared_list):
        Q, P = fn.compute_Q_P(angle, freq, k2)
        Q_list.append(Q)
        P_list.append(P)

    Q = np.array(Q_list)
    P = np.array(P_list) 

    phi, delta = fn.compute_phi_delta(Q, P)
    phi = np.mod(phi, 2 * np.pi) 
    q_init = np.array(phi)
    p_init = np.array(delta)

    return q_init, p_init

def load_data(filename):
    data = np.load(filename)
    q = data['q']
    p = data['p']

    Q, P = np.meshgrid(q, p)
    q_init, p_init = Q.ravel(), P.ravel()

    q_init = np.array(q_init)
    p_init = np.array(p_init)

    return q_init, p_init


# ---------------------------------------


if __name__ == "__main__":
    grid_lim = float(sys.argv[1])
    n_particles = int(sys.argv[2])
    loaded_data = sys.argv[3] if len(sys.argv) > 3 else None

    if loaded_data is not None:
        q_init, p_init = load_data(loaded_data)
    else:
        q_init, p_init = generate_grid(grid_lim, n_particles) 

    output_dir = "init_conditions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"qp.npz")
    np.savez(file_path, q=q_init, p=p_init)