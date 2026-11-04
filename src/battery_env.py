import qutip as qt
import numpy as np
import math
import gymnasium as gym
from typing import Any
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Physical quantities from the supplied Dicke battery source
# ---------------------------------------------------------------------------

def rho_b1(rho_b: qt.Qobj) -> np.ndarray:
    """Reduced density matrix of one TLS, using the symmetric Dicke basis."""
    n = int(rho_b.dims[0][0] - 1)
    rho = np.zeros((2, 2), dtype=complex)

    for e in range(n):
        rho[1, 1] += (n - e) / n * rho_b[n - e, n - e]
        rho[1, 0] += (
            np.sqrt((e + 1) * (n - e)) / n
            * rho_b[n - e - 1, n - e]
        )
        rho[0, 0] += (e + 1) / n * rho_b[n - e - 1, n - e - 1]

    rho[0, 1] = np.conjugate(rho[1, 0])
    return rho


def ergotropy_1tls(rho_b: qt.Qobj, omega0: float) -> float:
    """Single-TLS ergotropy from the symmetric Dicke state."""

    rho = rho_b1(rho_b)

    # Energy of the single TLS
    energy = omega0 * np.real(rho[1, 1])

    # Passive-state energy
    eigenvalues = np.linalg.eigvalsh(rho)
    passive_energy = omega0 * eigenvalues[0]

    return float(max(0.0, energy - passive_energy))


# ---------------------------------------------------------------------------
# Gymnasium environment: original DickeBatteryOneControlDetuning physics
# ---------------------------------------------------------------------------

class DickeBatteryOneControlDetuning(gym.Env):
    """
    Control u(t) is the detuning of the TLS frequency.

    Observation:
        [real(psi), imag(psi), last_control, elapsed_time]

    Action:
        one continuous control in [min_u, max_u].
    """

    metadata = {"render_modes": []}

    def __init__(self, env_params: dict[str, Any], reward_schedule: dict[str, Any] | None = None):
        super().__init__()
        p = dict(env_params)

        self.wc = float(p["wc"])
        self.nc = int(p["nc"])
        self.nc_i = int(p["nc_i"])
        self.wq = float(p["wq"])
        self.nq = int(p["nq"])
        self.g = float(p["g"])
        self.min_u = float(p["min_u"])
        self.max_u = float(p["max_u"])
        self.dt = float(p["dt"])
        self.tau = float(p["tau"])
        self.reward_coeff = float(p.get("reward_coeff", 1.0))
        self.quantity = p.get("quantity", "ergotropy")

        if self.quantity not in ("energy", "ergotropy"):
            raise ValueError("quantity must be 'energy' or 'ergotropy'")
        if self.nc_i > self.nc:
            raise ValueError("nc_i must be <= nc")

        # Number of complete control intervals required to reach/exceed tau.
        self.max_episode_steps = int(math.ceil(self.tau / self.dt))
        self.reward_schedule = reward_schedule or {}
        self.training_step = 0
        self.step_count = 0

        self._init_hamiltonian()

        psi_size = (self.nc + 1) * (self.nq + 1)
        obs_dim = 2 * psi_size + 2

        low = np.full(obs_dim, -1.0, dtype=np.float32)
        high = np.full(obs_dim, 1.0, dtype=np.float32)
        low[-2], high[-2] = self.min_u, self.max_u
        low[-1], high[-1] = 0.0, self.tau

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([self.min_u], dtype=np.float32),
            high=np.array([self.max_u], dtype=np.float32),
            dtype=np.float32,
        )

        self.psi: qt.Qobj | None = None
        self.u = self.min_u
        self.t = 0.0
        self.reset()

    def _init_hamiltonian(self) -> None:
        a = qt.tensor(qt.destroy(self.nc + 1), qt.qeye(self.nq + 1))
        jz_b = qt.tensor(qt.qeye(self.nc + 1), qt.jmat(self.nq / 2, "z"))

        self.hq_bare = jz_b + self.nq / 2
        self.hq = self.wq * self.hq_bare
        self.hc = self.wc * a.dag() * a
        self.hc_hint = self.hc + self.g * 2.0 * qt.tensor(
            qt.destroy(self.nc + 1).dag() + qt.destroy(self.nc + 1),
            qt.jmat(self.nq / 2, "x"),
        )

    def hamiltonian(self, u: float) -> qt.Qobj:
        return self.hc_hint + (self.wq + u) * self.hq_bare

    def current_energy(self) -> float:
        return float(np.real(qt.expect(self.hq, self.psi)))

    def current_ergotropy(self) -> float:
        return ergotropy_1tls(self.psi.ptrace(1), self.wq)

    def _observation(self) -> np.ndarray:
        psi_np = self.psi.full().reshape(-1)
        return np.concatenate(
            [psi_np.real, psi_np.imag, np.array([self.u, self.t])]
        ).astype(np.float32)

    def set_training_step(self, step: int) -> None:
        """Called by the SB3 callback; controls the scheduled scalarization."""
        self.training_step = int(step)

    def _objective_weights(self) -> np.ndarray:
        """Fermi interpolation from C_START to C_END."""
        c_start = np.asarray(self.reward_schedule.get("C_START", [1 / self.nq, 0.0]))
        c_end = np.asarray(self.reward_schedule.get("C_END", [0.0, 1.0]))
        mean = float(self.reward_schedule.get("C_MEAN", 60_000))
        width = max(float(self.reward_schedule.get("C_WIDTH", 20_000)), 1e-8)

        # Stable logistic: fraction rises from 0 to 1 around C_MEAN.
        x = np.clip((self.training_step - mean) / width, -60.0, 60.0)
        fraction = 1.0 / (1.0 + np.exp(-x))
        return (1.0 - fraction) * c_start + fraction * c_end

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        cavity = qt.basis(self.nc + 1, self.nc_i)
        qubits = qt.basis(self.nq + 1, self.nq)
        self.psi = qt.tensor(cavity, qubits)
        self.u = self.min_u
        self.t = 0.0
        self.step_count = 0

        return self._observation(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        u = float(action[0])

        energy_i = self.current_energy()
        ergotropy_i = self.current_ergotropy()

        result = qt.mesolve(self.hamiltonian(u), self.psi, [0.0, self.dt])
        self.psi = result.states[-1]
        self.u = u
        self.step_count += 1
        self.t = min(self.step_count * self.dt, self.tau)

        energy_f = self.current_energy()
        ergotropy_f = self.current_ergotropy()

        delta_energy = self.reward_coeff * (energy_f - energy_i)
        delta_ergotropy = self.reward_coeff * (ergotropy_f - ergotropy_i)
        objectives = np.array([delta_energy, delta_ergotropy], dtype=np.float32)

        weights = self._objective_weights()
        reward = float(np.dot(weights, objectives))

        terminated = self.step_count >= self.max_episode_steps
        truncated = False
        info = {
            "multi_obj": objectives,
            "energy": energy_f,
            "ergotropy": ergotropy_f,
            "control": u,
            "objective_weights": weights.astype(np.float32),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        print({"time": self.t, "control": self.u})

