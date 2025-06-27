import os
import sys
import numpy as np

import params as par
import functions as fn

def run_integrator(mode):
    data = np.load("init_conditions/qp.npz")

    q_init = data['q']
    p_init = data['p']

    q = q_init.copy()
    p = p_init.copy()

    q_traj = []
    p_traj = []

    q_sec = []
    p_sec = []

    psi = par.phi_0

    for _ in range(par.n_steps):
        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

        if mode == "tune":
            q_traj.append(q.copy())
            p_traj.append(p.copy())

        elif mode == "phasespace":
            if np.cos(psi) > 1.0 - 1e-3:              
                q_sec.append(q.copy())
                p_sec.append(p.copy())
                
        psi += par.omega_m * par.dt
        par.t += par.dt

    if mode == "tune":
        q = np.array(q_traj)
        p = np.array(p_traj)

    elif mode == "phasespace":
        q = np.concatenate(q_sec)
        p = np.concatenate(p_sec)

        q = np.array(q)
        p = np.array(p)

    return q, p


# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    q, p = run_integrator(mode)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{mode}.npz")
    np.savez(file_path, q=q, p=p)