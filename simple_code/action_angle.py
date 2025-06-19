import numpy as np
import functions as fn
import os
import sys
from tqdm.auto import tqdm
if os.environ.get("PHASE_SPACE", "0") == "1":
    import params_fixed as par
else:
    import params as par

def run_action_angle():
    data = np.load("integrator/evolved_qp.npz")
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

# --------------- Save results ----------------

    is_phase_space = os.environ.get("PHASE_SPACE", "0") == "1"
    output_dir = "action_angle"

    print(is_phase_space)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if is_phase_space:
        a_val = par.a_lambda(0)
        omega_val = par.omega_lambda(0)

        a_str = f"{a_val:.3f}"
        omega_str = f"{omega_val:.2f}"
        file_path = os.path.join(output_dir, f"cartesian_a{a_str}_omega{omega_str}.npz")

    else:
        file_path = os.path.join(output_dir, "cartesian.npz")      

    np.savez(file_path, x=x, y=y)
    
# ---------------------------------------------

if __name__ == "__main__":
    run_action_angle()