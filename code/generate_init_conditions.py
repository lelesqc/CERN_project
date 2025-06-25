import os
import sys
import random
import numpy as np
import functions as fn
from scipy.special import ellipk

import params as par

def generate_init(radius, n_particles):
    X_list = []
    Y_list = []

    while len(X_list) < n_particles:
        X = random.uniform(-radius, radius)
        Y = random.uniform(-radius, radius)

        if X**2 + Y**2 <= radius**2:
            X_list.append(X)
            Y_list.append(Y)

    X_list = np.array(X_list)
    Y_list = np.array(Y_list)

    action, theta = fn.compute_action_angle_inverse(X_list, Y_list)

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


# ----------------------------------------


if __name__ == "__main__":
    radius = float(sys.argv[1])
    n_particles = int(sys.argv[2])
    q_init, p_init = generate_init(radius, n_particles)

    output_dir = "init_conditions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "init_distribution.npz")
    np.savez(file_path, q=q_init, p=p_init)