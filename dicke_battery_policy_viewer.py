"""
Visualize a trained Stable-Baselines3 SAC policy on the Dicke quantum battery.

Install:
    pip install "stable-baselines3[extra]" gymnasium qutip numpy matplotlib

Examples:
    python dicke_battery_policy_viewer.py
    python dicke_battery_policy_viewer.py --model dicke_sac_logs/models/dicke_battery_sac_final.zip
    python dicke_battery_policy_viewer.py --model dicke_sac_logs/models/dicke_sac_20000_steps.zip --episodes 3
    python dicke_battery_policy_viewer.py --stochastic

IMPORTANT:
    Set NQ and the environment parameters below to exactly match the model's
    training observation/action dimensions and physical setup.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from gymnasium import spaces
from stable_baselines3 import SAC


# ---------------------------------------------------------------------------
# Environment configuration: MUST match the model used for training.
# Defaults below match the recent nq=4 experiment.
# Change NQ to 16 if loading a model trained with nq=16.
# ---------------------------------------------------------------------------

NQ = 4
TAU = 2.844
DT = 0.0889

ENV_PARAMS = {
    "wc": 1.0,
    "nc": NQ * 5,
    "nc_i": NQ,
    "wq": 1.0,
    "nq": NQ,
    "g": 0.3,
    "min_u": -1.0,
    "max_u": 6.0,
    "dt": DT,
    "tau": TAU,
    "reward_coeff": 1.0,
    "quantity": "ergotropy",
}

DEFAULT_MODEL = "dicke_sac_logs/models/dicke_battery_sac_final.zip"


# ---------------------------------------------------------------------------
# Physical quantities
# ---------------------------------------------------------------------------

def rho_b1(rho_b: qt.Qobj) -> np.ndarray:
    """One-TLS reduced density matrix in the [ground, excited] basis."""
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
    """
    Single-TLS ergotropy for H_TLS = omega0 |excited><excited|.

    The [ground, excited] convention means the TLS energy is omega0*rho[1,1].
    """
    rho = rho_b1(rho_b)
    energy = omega0 * float(np.real(rho[1, 1]))
    passive_energy = omega0 * float(np.linalg.eigvalsh(rho)[0])
    return float(max(0.0, energy - passive_energy))


# ---------------------------------------------------------------------------
# Same physical environment as training, with trajectory recording in viewer.
# ---------------------------------------------------------------------------

class DickeBatteryOneControlDetuning(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_params: dict):
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

        if self.nc_i > self.nc:
            raise ValueError("nc_i must be <= nc")

        self.max_episode_steps = int(math.ceil(self.tau / self.dt))
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

        self.psi = None
        self.u = self.min_u
        self.t = 0.0
        self.step_count = 0
        self.reset()

    def _init_hamiltonian(self):
        a = qt.tensor(qt.destroy(self.nc + 1), qt.qeye(self.nq + 1))
        jz = qt.tensor(qt.qeye(self.nc + 1), qt.jmat(self.nq / 2, "z"))

        self.hq_bare = jz + self.nq / 2
        self.hq = self.wq * self.hq_bare
        self.hc = self.wc * a.dag() * a
        self.hc_hint = self.hc + 2.0 * self.g * qt.tensor(
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cavity = qt.basis(self.nc + 1, self.nc_i)
        qubits = qt.basis(self.nq + 1, self.nq)
        self.psi = qt.tensor(cavity, qubits)
        self.u = self.min_u
        self.t = 0.0
        self.step_count = 0
        return self._observation(), {}

    def step(self, action):
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

        reward = self.reward_coeff * (ergotropy_f - ergotropy_i)
        terminated = self.step_count >= self.max_episode_steps
        truncated = False

        info = {
            "energy": energy_f,
            "ergotropy": ergotropy_f,
            "control": u,
            "delta_energy": self.reward_coeff * (energy_f - energy_i),
            "delta_ergotropy": self.reward_coeff * (ergotropy_f - ergotropy_i),
        }
        return self._observation(), float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# Live policy visualization
# ---------------------------------------------------------------------------

def run_episode(model, env, episode_index: int, deterministic: bool, live: bool):
    obs, _ = env.reset(seed=episode_index)
    done = False

    times = [0.0]
    controls = [env.u]
    energies = [env.current_energy()]
    ergotropies = [env.current_ergotropy()]
    rewards = [0.0]

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_u, ax_energy, ax_erg, ax_reward = axes.flat

    fig.suptitle(
        f"Dicke Quantum Battery | SAC policy | Episode {episode_index + 1}",
        fontsize=14,
    )

    line_u, = ax_u.plot(times, controls, lw=2, label="detuning u(t)")
    ax_u.set(title="Policy control", xlabel="Time", ylabel="u(t)")
    ax_u.set_ylim(env.min_u - 0.3, env.max_u + 0.3)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend()

    line_energy, = ax_energy.plot(times, energies, lw=2, label="TLS energy")
    ax_energy.set(title="Stored TLS energy", xlabel="Time", ylabel="Energy")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()

    line_erg, = ax_erg.plot(times, ergotropies, lw=2, color="tab:orange",
                            label="single-TLS ergotropy")
    ax_erg.set(title="Extractable work", xlabel="Time", ylabel="Ergotropy")
    ax_erg.grid(True, alpha=0.3)
    ax_erg.legend()

    line_reward, = ax_reward.plot(times, rewards, lw=2, color="tab:green",
                                  label="cumulative ergotropy reward")
    ax_reward.set(title="Cumulative reward", xlabel="Time", ylabel="Reward")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend()

    status = fig.text(0.5, 0.01, "", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.show()
    plt.pause(0.05)

    cumulative_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward += reward

        times.append(env.t)
        controls.append(float(action[0]))
        energies.append(info["energy"])
        ergotropies.append(info["ergotropy"])
        rewards.append(cumulative_reward)

        status.set_text(
            f"t={env.t:.3f}/{env.tau:.3f} | u={float(action[0]):.4f} | "
            f"E={info['energy']:.5f} | W={info['ergotropy']:.5f}"
        )

        if live:
            line_u.set_data(times, controls)
            line_energy.set_data(times, energies)
            line_erg.set_data(times, ergotropies)
            line_reward.set_data(times, rewards)

            for ax in axes.flat:
                ax.relim()
                ax.autoscale_view(scalex=True, scaley=True)
            ax_u.set_ylim(env.min_u - 0.3, env.max_u + 0.3)
            fig.canvas.draw_idle()
            plt.pause(0.03)

    # Ensure final trajectory is fully rendered, even if live=False.
    line_u.set_data(times, controls)
    line_energy.set_data(times, energies)
    line_erg.set_data(times, ergotropies)
    line_reward.set_data(times, rewards)
    for ax in axes.flat:
        ax.relim()
        ax.autoscale_view()
    ax_u.set_ylim(env.min_u - 0.3, env.max_u + 0.3)
    status.set_text(
        f"Completed | return={cumulative_reward:.6f} | "
        f"final energy={energies[-1]:.6f} | "
        f"final ergotropy={ergotropies[-1]:.6f}"
    )
    fig.canvas.draw_idle()
    plt.ioff()
    plt.show()

    print(
        f"Episode {episode_index + 1}: "
        f"return={cumulative_reward:.6f}, "
        f"final_energy={energies[-1]:.6f}, "
        f"final_ergotropy={ergotropies[-1]:.6f}"
    )

    return {
        "time": np.asarray(times),
        "control": np.asarray(controls),
        "energy": np.asarray(energies),
        "ergotropy": np.asarray(ergotropies),
        "cumulative_reward": np.asarray(rewards),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained SAC policy control the Dicke quantum battery."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to SB3 SAC .zip model or checkpoint.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to visualize.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample stochastic policy actions instead of deterministic actions.")
    parser.add_argument("--no-live", action="store_true",
                        help="Skip per-step animation and show completed episode plots.")
    parser.add_argument("--device", default="auto",
                        help="SB3 device: auto, cpu, cuda, cuda:0, etc.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Pass the path to your final model or a saved checkpoint with --model."
        )

    env = DickeBatteryOneControlDetuning(ENV_PARAMS)

    print(f"Loading model: {model_path}")
    model = SAC.load(str(model_path), env=env, device=args.device)

    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            "Observation shape mismatch: model expects "
            f"{model.observation_space.shape}, environment provides "
            f"{env.observation_space.shape}. Check NQ and ENV_PARAMS."
        )
    if model.action_space.shape != env.action_space.shape:
        raise ValueError(
            "Action shape mismatch: model expects "
            f"{model.action_space.shape}, environment provides "
            f"{env.action_space.shape}."
        )

    print(f"Model loaded. Observation shape: {env.observation_space.shape}")
    print(f"Action range: [{env.min_u}, {env.max_u}]")
    print(f"Policy mode: {'stochastic' if args.stochastic else 'deterministic'}")

    for episode in range(args.episodes):
        run_episode(
            model=model,
            env=env,
            episode_index=episode,
            deterministic=not args.stochastic,
            live=not args.no_live,
        )

    env.close()


if __name__ == "__main__":
    main()
