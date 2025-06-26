import os
import sys
import numpy as np

def phase_space(grid_lim, n_particles):
    q = np.pi
    p = np.linspace(-grid_lim, grid_lim, n_particles)
    Q, P = np.meshgrid(q, p)
    q_init, p_init = Q.ravel(), P.ravel()

    q_init = np.array(q_init)
    p_init = np.array(p_init)

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
        q_init, p_init = phase_space(grid_lim, n_particles) 

    output_dir = "init_conditions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"qp.npz")
    np.savez(file_path, q=q_init, p=p_init)