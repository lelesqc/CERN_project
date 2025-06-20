import numpy as np
import functions as fn
import os
import sys
from tqdm.auto import tqdm
if os.environ.get("PHASE_SPACE", "0") == "1":
    import params_fixed as par
else:
    import params as par

def run_integrator(poincare_mode="last", poincare_every=1):
    data = np.load("init_conditions/initial.npz")

    q_init = data['q']
    p_init = data['p']

    q = q_init.copy()
    p = p_init.copy()

    count = 0
    q_sec, p_sec = [], []
    q_single = None
    p_single = None

    for _ in tqdm(range(par.n_steps)):
        q += fn.Delta_q(p, par.t, par.dt/2)
        q = np.mod(q, 2 * np.pi)
        
        t_mid = par.t + par.dt/2

        p += par.dt * fn.dV_dq(q)

        q += fn.Delta_q(p, t_mid, par.dt/2)
        q = np.mod(q, 2 * np.pi)
               
        if np.cos(par.omega_lambda(par.t) * par.t) > 1.0-1e-5:
            if poincare_mode == "first":
                if q_single == None:
                    q_single = q.copy()
                    p_single = p.copy()
            #elif poincare_mode == "last":
            #    q_single = q.copy()
            #    p_single = p.copy()
            elif poincare_mode == "all":
                q_sec.append(q.copy())
                p_sec.append(p.copy())
            elif poincare_mode == "number" and (count % poincare_every == 0):
                q_sec.append(q.copy())
                p_sec.append(p.copy())
                
            count += 1

        par.t += par.dt

        if par.t == par.T_tot:
            print(f"{par.a_lambda(par.t):.3f}, {par.omega_lambda(par.t)/par.omega_s:.3f}")


    if poincare_mode == "last":
        orig_a_lambda = par.a_lambda
        orig_omega_lambda = par.omega_lambda
        a_const = par.a_lambda(par.T_tot)
        omega_const = par.omega_lambda(par.T_tot)
        par.a_lambda = lambda t: a_const
        par.omega_lambda = lambda t: omega_const

        found = False
        while not found:
            q += fn.Delta_q(p, par.t, par.dt/2)
            q = np.mod(q, 2 * np.pi)
            t_mid = par.t + par.dt/2
            p += par.dt * fn.dV_dq(q)
            q += fn.Delta_q(p, t_mid, par.dt/2)
            q = np.mod(q, 2 * np.pi)

            if np.cos(par.omega_lambda(par.t) * par.t) > 1.0-1e-5:
                q_single = q.copy()
                p_single = p.copy()
                found = True

                print(f"par.omega_lambda(par.t) = {par.omega_lambda(par.t)/par.omega_s:.3f}, par.a_lambda(par.t) = {par.a_lambda(par.t):.3f}")

            par.t += par.dt

        par.a_lambda = orig_a_lambda
        par.omega_lambda = orig_omega_lambda

    elif os.environ.get("PHASE_SPACE", "0") == "1":
        orig_a_lambda = par.a_lambda
        orig_omega_lambda = par.omega_lambda
        a_const = par.a_lambda(0)
        omega_const = par.omega_m
        par.a_lambda = lambda t: a_const
        par.omega_lambda = lambda t: omega_const

        found = False
        while not found:
            q += fn.Delta_q(p, par.t, par.dt/2)
            q = np.mod(q, 2 * np.pi)
            t_mid = par.t + par.dt/2
            p += par.dt * fn.dV_dq(q)
            q += fn.Delta_q(p, t_mid, par.dt/2)
            q = np.mod(q, 2 * np.pi)

            if np.cos(par.omega_lambda(par.t) * par.t) > 1.0-1e-5:
                q_sec.append(q.copy())
                p_sec.append(p.copy())
                found = True

                print(f"par.omega_lambda(par.t) = {par.omega_lambda(par.t)/par.omega_s:.3f}, par.a_lambda(par.t) = {par.a_lambda(par.t):.3f}")

            par.t += par.dt

        par.a_lambda = orig_a_lambda
        par.omega_lambda = orig_omega_lambda
    
    if poincare_mode in ["first", "last"]:
            q = q_single
            p = p_single

    elif poincare_mode in ["all", "every"]:
        if q_sec and p_sec:
            q = np.concatenate(q_sec)
            p = np.concatenate(p_sec)
        
    q = np.array(q)
    p = np.array(p)
        
# --------------- Save results -----------------

    output_dir = "integrator"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "evolved_qp.npz")
    np.savez(file_path, q=q, p=p)

# ----------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        poincare_mode = sys.argv[1]
        run_integrator(poincare_mode)
    else:
        run_integrator()