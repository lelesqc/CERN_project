import os
import sys
import numpy as np

import params as par
import functions as fn

def run_integrator(poincare_mode):
    data = np.load("init_conditions/init_distribution.npz")

    q_init = data['q']
    p_init = data['p']

    q = q_init.copy()
    p = p_init.copy()

    q_single = None
    p_single = None

    q_traj = []
    p_traj = []
    q_traj.append(q.copy())
    p_traj.append(p.copy())

    q_sec = []
    p_sec = []

    step = 0
    psi = par.phi_0
    find_poincare = False
    fixed_params = False
    done = False

    n_extra = 4096
    if poincare_mode == "none":
        while done == False:
            q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)
            q_traj.append(q.copy())
            p_traj.append(p.copy())

            psi += par.omega_lambda(par.t) * par.dt
            par.t += par.dt

            if par.t >= par.T_tot:
                    fixed_params = True

            if fixed_params == True:
                for _ in range(n_extra):
                    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)
                    psi += par.omega_lambda(par.T_tot) * par.dt
                    par.t += par.dt

                    if _ == n_extra - 1:
                        done = True

    else:
        while not find_poincare:
            q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

            if par.t >= par.T_tot:
                fixed_params = True

            if np.cos(psi) > 1.0 - 1e-3:
                if poincare_mode == "first":
                    if q_single == None:
                        q_single = q.copy()
                        p_single = p.copy()
                        find_poincare = True                
                elif poincare_mode == "all":
                    q_sec.append(q.copy())
                    p_sec.append(p.copy())
                    if fixed_params:
                        find_poincare = True
                elif poincare_mode == "last" and fixed_params:
                    q_single = q.copy()
                    p_single = p.copy()
                    find_poincare = True

            psi += par.omega_lambda(par.t) * par.dt
            par.t += par.dt
            step += 1

            if step == par.n_steps // 4:
                print(r">>> 25% completed")
            elif step == par.n_steps // 2:
                print(r">>> 50% completed")
            elif step == 3 * par.n_steps // 4:
                print(r">>> 75% completed")

    if not poincare_mode in ["all", "none"]:
        q = q_single
        p = p_single
    else:
        q = np.concatenate(q_sec)
        p = np.concatenate(p_sec)

    q = np.array(q)
    p = np.array(p)

    return q, p


# --------------- Save results ----------------


if __name__ == "__main__":
    poincare_mode = sys.argv[1]
    q, p = run_integrator(poincare_mode)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{poincare_mode}.npz")
    np.savez(file_path, q=q, p=p)