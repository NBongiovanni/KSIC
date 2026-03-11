from __future__ import annotations

import numpy as np

from .plant import Plant

G = 9.81

class PlanarQuad(Plant):
    """
    2D planar quad model.
    State: [y, z, theta, y_dot, z_dot, theta_dot] (6)
    Input: [F, tau] (2)
    """

    def __init__(self, dt: float, mass: float = 1.0, inertia: float = 0.15):
        super().__init__(dt, discrete_or_continuous="continuous")
        self.drone_dim = 2
        self.x_dim = 6
        self.u_dim = 2

        self.mass = float(mass)
        self.inertia = float(inertia)

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        y, z, theta, y_dot, z_dot, theta_dot = x
        F, tau = u

        y_ddot = -(F / self.mass) * np.sin(theta)
        z_ddot = (F / self.mass) * np.cos(theta) - G
        theta_ddot = tau / self.inertia

        return np.array([y_dot, z_dot, theta_dot, y_ddot, z_ddot, theta_ddot], dtype=float)


class Quad3D(Plant):
    """
    Simple 3D quadrotor model.
    State (12): [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    Input (4): [F, tau_x, tau_y, tau_z]
    - Uses Euler angles (phi roll, theta pitch, psi yaw)
    - Rotational kinematics: [phi_dot, theta_dot, psi_dot] = T(angles) * [p, q, r]
    - Translational acceleration: a = (R * [0,0,F]/m) - [0,0,g]
    - Rotational dynamics: omega_dot = I^{-1}(tau - omega x I omega)
    This is a standard "minimal" model (good for simulation/control prototypes).
    """

    def __init__(
        self,
        dt: float,
        mass: float = 28e-3,
        inertia: tuple[float, float, float] = (16.6e-6, 16.6e-6, 29.3e-6),
    ):
        super().__init__(dt, discrete_or_continuous="continuous")
        self.drone_dim = 3
        self.x_dim = 12
        self.u_dim = 4

        self.mass = float(mass)
        self.Ix, self.Iy, self.Iz = (float(inertia[0]), float(inertia[1]), float(inertia[2]))

    @staticmethod
    def _rot_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)

        # R = Rz(psi) Ry(theta) Rx(phi)
        Rz = np.array([[cps, -sps, 0.0], [sps, cps, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cph, -sph], [0.0, sph, cph]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _euler_rates_matrix(phi: float, theta: float) -> np.ndarray:
        """
        [phi_dot, theta_dot, psi_dot]^T = T(phi,theta) [p,q,r]^T
        """
        cth = np.cos(theta)
        if abs(cth) < 1e-6:
            # avoid numerical blow-up near +-90deg pitch
            cth = np.sign(cth) * 1e-6 if cth != 0 else 1e-6

        tth = np.tan(theta)
        cph, sph = np.cos(phi), np.sin(phi)

        return np.array(
            [
                [1.0, sph * tth, cph * tth],
                [0.0, cph, -sph],
                [0.0, sph / cth, cph / cth],
            ],
            dtype=float,
        )

    def _dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Unpack state
        px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x
        F, tau_x, tau_y, tau_z = u

        # Rotation matrix and thrust in world frame
        R = self._rot_matrix(phi, theta, psi)
        thrust_world = R @ np.array([0.0, 0.0, F], dtype=float)

        ax, ay, az = thrust_world / self.mass - np.array([0.0, 0.0, G], dtype=float)

        # Euler angle rates
        T = self._euler_rates_matrix(phi, theta)
        euler_dot = T @ np.array([p, q, r], dtype=float)
        phi_dot, theta_dot, psi_dot = euler_dot

        # Rotational dynamics (rigid body with diagonal inertia)
        omega = np.array([p, q, r], dtype=float)
        tau = np.array([tau_x, tau_y, tau_z], dtype=float)

        I = np.diag([self.Ix, self.Iy, self.Iz])
        omega_dot = np.linalg.solve(I, (tau - np.cross(omega, I @ omega)))
        p_dot, q_dot, r_dot = omega_dot

        # Assemble derivative
        return np.array(
            [
                vx,
                vy,
                vz,
                phi_dot,
                theta_dot,
                psi_dot,
                ax,
                ay,
                az,
                p_dot,
                q_dot,
                r_dot,
            ],
            dtype=float,
        )
