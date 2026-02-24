from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Callable

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np

from .solver_backend import SolverBackend
from .mpc_problem import MPCProblem

@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

class _TVPTemplate:
    """Minimal TVP template to interoperate with the existing tvp_provider.
    It captures z_ref over the horizon when the provider assigns to
    template['_tvp', k, 'z_ref'].
    """

    def __init__(self, N: int, z_dim: int):
        self.N = N
        self.z_dim = z_dim
        self.data = np.zeros((N + 1, z_dim))

    def __setitem__(self, key, value):
        # expected key format: ('_tvp', k, 'z_ref')
        _, k, name = key
        if name != 'z_ref':
            return
        arr = np.asarray(value).reshape(-1)
        self.data[k, : arr.shape[0]] = arr

    # optional: allow provider to read as well if needed later
    def __getitem__(self, key):
        _, k, name = key
        if name != 'z_ref':
            raise KeyError(name)
        return self.data[k]


@dataclass
class _AcadosInternals:
    ocp: AcadosOcp
    solver: AcadosOcpSolver


class AcadosBackend(SolverBackend):
    """
    Solver backend using ACADOS with a *discrete-time* model and LINEAR_LS costs.

    Drop-in replacement for DoMPCBackend with the same public API expected by
    MPCControllerBase.
    """
    def __init__(self, control_params: dict):
        super().__init__()  # Appel du __init__ de SolverBackend
        self.ctrl_params = control_params
        self.solver_options = control_params["solver_options"]

        self.problem: Optional[MPCProblem] = None
        self._internals: Optional[_AcadosInternals] = None

        self._tvp_provider: Optional[Callable] = None
        self._k: int = 0  # internal time step counter

        # cached sizes
        self.z_dim = None
        self.u_dim = None
        self.N = None
        self.dt = None

    # --------------- public API (SolverBackend) ---------------
    def build(self, problem: MPCProblem) -> None:
        self.problem = problem
        self.z_dim = problem.z_dim
        self.u_dim = problem.u_dim
        self.N = problem.N
        self.dt = problem.dt
        self._tvp_provider = problem.tvp_provider

        ocp = AcadosOcp()

        model = AcadosModel()

        # casadi symbols
        z = ca.MX.sym('z', self.z_dim, 1)
        u = ca.MX.sym('u', self.u_dim, 1)

        u_prev = ca.MX.sym('u_prev', self.u_dim, 1)

        # discrete dynamics: z_{k+1} = f(z_k, u_k)
        assert problem.f_discrete is not None, "MPCProblem.f_discrete must be set"
        z_next = problem.f_discrete(z, u)

        model.x = z
        model.u = u
        model.disc_dyn_expr = z_next
        model.name = 'koopman_latent_disc'
        ocp.model = model

        # horizon
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = self.solver_options["nlp_solver_type"]
        ocp.solver_options.print_level = self.solver_options["print_level"]

        if self.solver_options["nlp_solver_type"] == "SQP_RTI":
            ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # TODO: check this
            ocp.solver_options.qp_solver_iter_max = self.solver_options["qp_solver_iter_max"]
            ocp.solver_options.qp_tol = self.solver_options["qp_tol"]

        elif self.solver_options["nlp_solver_type"] == "SQP":
            ocp.solver_options.nlp_solver_max_iter = self.solver_options["nlp_solver_iter_max"]  # Nom correct !
            ocp.solver_options.nlp_tol = self.solver_options["nlp_tol"]
            ocp.solver_options.qp_solver_iter_max = self.solver_options["qp_solver_iter_max"]
            ocp.solver_options.qp_tol = self.solver_options["qp_tol"]
            ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # TODO: check this

        # -------- Costs (LINEAR_LS) --------
        # Si S est fourni, on utilise NONLINEAR_LS pour accéder aux slack variables
        if problem.S is not None and np.any(problem.S > 0):
            self._setup_cost_with_rate_penalty(ocp, problem)
        else:
            self._setup_standard_cost(ocp, problem)

        # Terminal cost: y_e = z, W_e = Qf, yref_e = z_ref
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.Vx_e = np.eye(self.z_dim)
        ocp.cost.W_e = problem.Qf
        ocp.cost.yref_e = np.zeros(self.z_dim)

        # -------- Input constraints
        if problem.use_inputs_constraints:
            ocp.constraints.lbu = np.asarray(problem.u_min).reshape(-1)
            ocp.constraints.ubu = np.asarray(problem.u_max).reshape(-1)
            ocp.constraints.idxbu = np.arange(self.u_dim, dtype=int)

        ocp.constraints.idxbx_0 = np.arange(self.z_dim, dtype=int)
        ocp.constraints.lbx_0 = np.zeros(self.z_dim)
        ocp.constraints.ubx_0 = np.zeros(self.z_dim)

        build_dir = self.ctrl_params["control_runs_dir"]
        json_path = str(build_dir / "acados_ocp.json")

        try:
            ocp.code_export_directory = str(build_dir)
        except AttributeError:
            pass

        print("\n[ACADOS COST DEBUG]")
        print("nx =", ocp.dims.nx, "nu =", ocp.dims.nu)

        print("[END ACADOS COST DEBUG]\n")

        with _pushd(build_dir):
            solver = AcadosOcpSolver(ocp, json_file=json_path)
        self._internals = _AcadosInternals(ocp=ocp, solver=solver)

        # Initialize u_prev for rate penalty (AJOUT)
        if problem.u_guess is not None:
            for i in range(self.N):
                solver.set(i, 'u', problem.u_guess)
            self._u_prev = np.asarray(problem.u_guess).reshape(-1)
        else:
            self._u_prev = np.zeros(self.u_dim)

        # set a decent initial state and guess if provided
        if problem.u_guess is not None:
            for i in range(self.N):
                solver.set(i, 'u', problem.u_guess)

    def _setup_standard_cost(self, ocp: AcadosOcp, problem: MPCProblem) -> None:
        """Setup standard LINEAR_LS cost without rate penalty."""
        ny = self.z_dim + self.u_dim

        Vx = np.zeros((ny, self.z_dim))
        Vx[:self.z_dim, :self.z_dim] = np.eye(self.z_dim)

        Vu = np.zeros((ny, self.u_dim))
        Vu[self.z_dim:, :self.u_dim] = np.eye(self.u_dim)

        W = self._blkdiag(problem.Q, problem.R)
        yref = np.zeros(ny)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.W = W
        ocp.cost.yref = yref

        # ---- FIX: define stage-0 cost explicitly (prevents ny_0 mismatch)
        ocp.cost.cost_type_0 = 'LINEAR_LS'
        ocp.cost.Vx_0 = Vx
        ocp.cost.Vu_0 = Vu
        ocp.cost.W_0 = W
        ocp.cost.yref_0 = yref

    def _setup_cost_with_rate_penalty(self, ocp: AcadosOcp, problem: MPCProblem) -> None:
        """
        Setup cost with input rate penalty using NONLINEAR_LS.

        This allows us to penalize Δu without augmenting the state.
        The cost becomes: ||z - z_ref||²_Q + ||u||²_R + ||u - u_prev||²_S
        """
        z = ocp.model.x
        u = ocp.model.u

        # Create parameter for u_prev (updated at each time step)
        u_prev = ca.MX.sym('u_prev', self.u_dim, 1)
        ocp.model.p = u_prev  # u_prev as runtime parameter

        # AJOUT ESSENTIEL : Valeur initiale des paramètres
        if problem.u_guess is not None:
            ocp.parameter_values = np.asarray(problem.u_guess).reshape(-1)
        else:
            ocp.parameter_values = np.zeros(self.u_dim)

        # Cost output: y = [z; u; u - u_prev]
        y_expr = ca.vertcat(z, u, u - u_prev)

        ny = self.z_dim + self.u_dim + self.u_dim
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = y_expr

        # Weight matrix W = diag(Q, R, S)
        W = np.zeros((ny, ny))
        W[:self.z_dim, :self.z_dim] = problem.Q
        W[self.z_dim:self.z_dim + self.u_dim, self.z_dim:self.z_dim + self.u_dim] = problem.R
        W[self.z_dim + self.u_dim:, self.z_dim + self.u_dim:] = problem.S

        ocp.cost.W = W
        ocp.cost.yref = np.zeros(ny)

    def reset(self) -> None:
        """
        Reset Acados solver for a new independent simulation.

        Uses the official Acados reset() API which clears:
        - QP solver memory and state
        - Primal and dual variable trajectories
        - Internal iteration counters
        """
        # Reset our time counter
        self._k = 0
        self.sqp_iters.clear()

        if self._internals is None:
            return

        # Use official Acados reset method
        solver = self._internals.solver
        solver.reset(reset_qp_solver_mem=1)
        print("[ACADOS] Solver reset via official API (cold start)")

        # Re-apply initial guess after cold start
        if self.problem is not None and self.problem.u_guess is not None:
            u0 = np.asarray(self.problem.u_guess).reshape(-1)
            for i in range(self.N):
                solver.set(i, "u", u0)

        if self.problem is not None and self.problem.u_guess is not None:
            self._u_prev = np.asarray(self.problem.u_guess).reshape(-1)
        else:
            self._u_prev = np.zeros(self.u_dim)

    def set_initial_condition(self, z0, u_guess=None) -> None:
        assert self._internals is not None
        solver = self._internals.solver
        z0 = np.asarray(z0).reshape(-1)
        solver.set(0, 'x', z0)
        # propagate a reasonable initial trajectory
        for i in range(1, self.N + 1):
            solver.set(i, 'x', z0)
        if u_guess is not None:
            for i in range(self.N):
                solver.set(i, 'u', np.asarray(u_guess).reshape(-1))

    def set_tvp_provider(self, provider) -> None:
        # Accept a provider of the same signature as in DoMPCBackend
        self._tvp_provider = provider

    def make_step(self, z_k):
        assert self._internals is not None
        solver = self._internals.solver
        problem = self.problem
        assert problem is not None

        # 1) set current state
        z_k = np.asarray(z_k).reshape(-1)
        solver.set(0, 'x', z_k)

        # 2) If using rate penalty, update u_prev parameter at each stage (AJOUT)
        if problem.S is not None and np.any(problem.S > 0):
            for i in range(self.N):
                solver.set(i, 'p', self._u_prev)

        # 2) build yref sequence from tvp_provider or keep defaults
        if self._tvp_provider is not None:
            template = _TVPTemplate(self.N, self.z_dim)
            t_now = self._k * self.dt
            _ = self._tvp_provider(t_now, template)
            z_refs = template.data  # shape (N+1, z_dim)

            # Determine ny based on cost type (CORRECTION ESSENTIELLE)
            if problem.S is not None and np.any(problem.S > 0):
                ny = self.z_dim + 2 * self.u_dim  # [z, u, Δu]
            else:
                ny = self.z_dim + self.u_dim  # [z, u]

            # stage refs
            for i in range(self.N):
                yref = np.zeros(ny)
                yref[: self.z_dim] = z_refs[i]
                solver.set(i, 'yref', yref)
            # terminal ref
            solver.set(self.N, 'yref', z_refs[self.N])

        solver.constraints_set(0, 'lbx', z_k)
        solver.constraints_set(0, 'ubx', z_k)
        solver.set(0, 'x', z_k)

        status = solver.solve()
        sqp_iter = solver.get_stats("sqp_iter")
        self.sqp_iters.append(sqp_iter)
        u0 = solver.get(0, 'u')
        self.print_acados_stats(solver)

        if problem.S is not None and np.any(problem.S > 0):
            self._u_prev = np.asarray(u0).reshape(-1)

        # 6) advance internal time
        self._k += 1
        return u0

    # --------------- helpers ---------------
    @staticmethod
    def _blkdiag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A)
        B = np.asarray(B)
        shape = (A.shape[0] + B.shape[0], A.shape[1] + B.shape[1])
        out = np.zeros(shape)
        out[: A.shape[0], : A.shape[1]] = A
        out[A.shape[0] :, A.shape[1] :] = B
        return out

    def print_acados_stats(self, solver):
        keys = [
            "status", "sqp_iter", "qp_iter", "qp_status",
            "qp_res_max", "qp_obj", "time_tot", "time_lin", "time_qp",
            "res_stat", "res_eq", "res_ineq", "res_comp"
        ]
        print("=== ACADOS Solver Statistics ===")
        for k in keys:
            try:
                print(f"{k:>12s}: {solver.get_stats(k)}")
            except Exception:
                pass
        print("================================")