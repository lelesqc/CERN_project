import numpy as np
import functions as fn
import os
import sys
from tqdm.auto import tqdm
if os.environ.get("PHASE_SPACE", "0") == "1":
    import params_fixed as par
else:
    import params as par

is_phase_space = os.environ.get("PHASE_SPACE", "0") == "1"

def run_integrator(poincare_mode="last", poincare_every=1):
    data = np.load("init_conditions/initial.npz")

    q_init = data['q']
    p_init = data['p']

    q = q_init.copy()
    p = p_init.copy()

    step = 0
    count = 0

    q_sec, p_sec = [], []
    q_single = None
    p_single = None
    q_full, p_full = [], []

    found_last_poincare = False

    while not found_last_poincare:
        if poincare_mode in ["last", "all", "none"] and step >= par.n_steps:
            if np.cos(par.omega_lambda(par.t) * par.t) > 1.0 - 1e-6:
                #print("entrato", par.t, step, par.a_lambda(par.t), par.omega_lambda(par.t) / par.omega_s, np.cos(par.omega_lambda(par.t) * par.t))
                q_sec.append(q.copy())
                p_sec.append(p.copy())
                q_single = q.copy()
                p_single = p.copy()
                q_full.append(q.copy())
                p_full.append(p.copy())
                found_last_poincare = True

                par.a_lambda = orig_a_lambda
                par.omega_lambda = orig_omega_lambda 
        if is_phase_space:
            if poincare_mode == "all":
                if np.cos(par.omega_lambda(par.t) * par.t) > 1.0 - 1e-6:
                    q_sec.append(q.copy())
                    p_sec.append(p.copy())
            elif poincare_mode == "none":
                q_full.append(q.copy())
                p_full.append(p.copy())
            orig_a_lambda = par.a_lambda
            orig_omega_lambda = par.omega_lambda
            a_const = par.a_lambda(par.T_tot)
            omega_const = par.omega_lambda(par.T_tot)
            par.a_lambda = lambda t: a_const
            par.omega_lambda = lambda t: omega_const

            q, p = fn.integrator_step(q, p, par.t, par.dt, fn.Delta_q, fn.dV_dq)

        else: 
            if step == par.n_steps:
                orig_a_lambda = par.a_lambda
                orig_omega_lambda = par.omega_lambda
                a_const = par.a_lambda(par.T_tot)
                omega_const = par.omega_lambda(par.T_tot)
                par.a_lambda = lambda t: a_const
                par.omega_lambda = lambda t: omega_const

            q, p = fn.integrator_step(q, p, par.t, par.dt, fn.Delta_q, fn.dV_dq)

            if poincare_mode == "none" and step < par.n_steps:
                q_full.append(q.copy())
                p_full.append(p.copy())            

            if np.cos(par.omega_lambda(par.t) * par.t) > 1.0 - 1e-6:
                if poincare_mode == "first":
                    if q_single is None:
                        q_single = q.copy()
                        p_single = p.copy()
                elif poincare_mode == "all":
                    q_sec.append(q.copy())
                    p_sec.append(p.copy())
                elif poincare_mode == "number" and (count % poincare_every == 0):
                    q_sec.append(q.copy())
                    p_sec.append(p.copy())                             

        if step == par.n_steps // 4:
            print(r">>> 25% completed")
        elif step == par.n_steps // 2:
            print(r">>> 50% completed")
        elif step == 3 * par.n_steps // 4:
            print(r">>> 75% completed") 

        par.t += par.dt
        step += 1

    if poincare_mode in ["first", "last"]:
            q = q_single
            p = p_single

    elif poincare_mode in ["all", "every"]:
        if q_sec and p_sec:
            q = np.concatenate(q_sec)
            p = np.concatenate(p_sec)

    elif poincare_mode == "none":
        q = np.concatenate(q_full)
        p = np.concatenate(p_full)
        
    q = np.array(q)
    p = np.array(p)
        
# --------------- Save results -----------------

    output_dir = "integrator"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{poincare_mode}.npz")
    np.savez(file_path, q=q, p=p)

# ----------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        poincare_mode = sys.argv[1]
        run_integrator(poincare_mode)
    else:
        run_integrator()