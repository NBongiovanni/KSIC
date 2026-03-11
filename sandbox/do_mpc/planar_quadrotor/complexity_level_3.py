from casadi import *
import matplotlib.pyplot as plt
import do_mpc
import pdb

from KSIC.closed_loop_eval import NonLinearPlanarQuadrotor

"""
Basic script implementing a control algorithm with the do-mpc library.
Plant: linear planar quadrotor
Controller: MPC using the nonlinear quadrotor model.
"""

def main():
    plant = NonLinearPlanarQuadrotor()
    rk_order = 1
    dt = 0.01
    n_steps = 600
    model_type = 'discrete' # either 'discrete' or 'continuous'
    n_horizon = 1
    u_constraints = True
    cost_u = 1e-3

    model = do_mpc.model.Model(model_type)
    _x = model.set_variable(var_type='_x', var_name='x', shape=(6,1))
    _u = model.set_variable('_u', 'u', (2,1))

    g = 9.81
    m = 1.0
    inertia = 0.15

    a_matrix = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -g, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    b_matrix = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1 / m, 0],
        [0, 1 / inertia]
    ])

    c_vec = np.array([0, 0, 0, 0, -g, 0])

    if rk_order == 1:
        # Runge-Kutta 1
        x_next = _x + dt*(a_matrix @ _x + b_matrix @ _u + c_vec)
    elif rk_order == 2:
        # Runge-Kutta 2
        f = a_matrix @ _x + b_matrix @ _u + c_vec
        k1 = f
        x_mid = _x + (dt/2)*k1
        f_mid = a_matrix @ x_mid + b_matrix @ _u + c_vec
        k2 = f_mid
        x_next = _x + dt*k2
    else:
        print("WARNING: problem in the name of the RK order")
        x_next = None

    model.set_rhs('x', x_next)

    x_ref = np.array([0, 0.5, 0, 0, 0, 0])

    q_matrix = 1 * np.diag([1, 1, 0, 0, 0, 0])
    q_matrix_f = 5 * np.diag([1, 1, 0, 1, 1, 0])

    c_cost = sum1((_x - x_ref).T @ q_matrix @ (_x - x_ref))
    f_cost = sum1((_x - x_ref).T @ q_matrix_f @ (_x - x_ref))
    model.set_expression(expr_name='c_cost', expr=c_cost)
    model.set_expression(expr_name='f_cost', expr=f_cost)

    # Build the model
    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': n_horizon,
        't_step': dt,
        'state_discretization': 'discrete',
        'store_full_solution':True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mterm = model.aux['f_cost'] # terminal cost
    lterm = model.aux['c_cost'] # running cost

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(u=cost_u) # input penalty

    if u_constraints:
        max_u = np.array([[20.0], [2.0]])
        min_u = np.array([[3.0], [-2.0]])
        mpc.bounds['lower', '_u', 'u'] = min_u
        mpc.bounds['upper', '_u', 'u'] = max_u

    mpc.setup()

    estimator = do_mpc.estimator.StateFeedback(model)

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = dt)
    simulator.setup()

    x_traj = []

    # Initial state
    x0 = np.zeros((6, 1))
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = plant.update_state(y_next.squeeze(), u0.squeeze())
        x_traj.append(x0)

    # Extract trajectories
    time = np.arange(mpc.data['_x'].shape[0]) * dt
    ref_traj = np.zeros((n_steps, 6))
    ref_traj[:] = x_ref

    x_traj = np.array(x_traj)
    inputs = mpc.data['_u']  # shape: (timesteps, 6)
    labels = ['y', 'z', 'theta', "y_dot", "z_dot", "theta_dot", "F", "T"]

    fig, axs = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
    for i, ax in enumerate(axs):
        if i < 6:
            ax.plot(time, x_traj[:, i])
            if i < 2:
                ax.plot(time, ref_traj[:, i])
        else:
            ax.plot(time, inputs[:, i-6])
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return None

if __name__ == '__main__':
    main()
