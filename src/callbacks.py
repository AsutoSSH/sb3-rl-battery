import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from typing import Optional
import csv

from src.battery_env import DickeBatteryOneControlDetuning

pg.setConfigOptions(
    antialias=True,
    useOpenGL=False,
    foreground="k",
    background="w"
)


# ---------------------------------------------------------------------------
# Callback: scheduled target entropy and objective weights
# ---------------------------------------------------------------------------

class QuantumBatteryScheduleCallback(BaseCallback):
    def __init__(
        self,
        env: DickeBatteryOneControlDetuning,
        h_start: float = 0.72,
        h_end: float = -3.0,
        h_decay: float = 200_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.quantum_env = env
        self.h_start = float(h_start)
        self.h_end = float(h_end)
        self.h_decay = float(h_decay)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        self.quantum_env.set_training_step(step)

        # SB3 SAC uses this value in its automatic entropy-temperature loss.
        target_entropy = self.h_end + (self.h_start - self.h_end) * np.exp(
            -step / self.h_decay
        )
        self.model.target_entropy = float(target_entropy)
        return True

def numeric_array(values):
    """Convert logged values to a clean float array."""
    result = []

    for value in values:
        try:
            value = float(value)
            result.append(value if np.isfinite(value) else np.nan)
        except (TypeError, ValueError):
            result.append(np.nan)

    return np.asarray(result, dtype=np.float64)


class PyQtGraphQuantumBatteryCallback(BaseCallback):
    """Live four-panel PyQtGraph dashboard plus periodic CSV/PNG snapshots."""

    def __init__(
        self,
        report_freq: int = 500,
        plot_freq: int = 100,
        save_freq: int = 5_000,
        plot_dir: str = "./dicke_sac_logs/plots",
        max_points: int = 5_000,
        live_plot: bool = True,
        save_plots: bool = True,
        save_csv: bool = True,
        window_title: str = "Dicke Quantum Battery — SAC Training",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.report_freq = max(1, int(report_freq))
        self.plot_freq = max(1, int(plot_freq))
        self.save_freq = max(1, int(save_freq))
        self.plot_dir = Path(plot_dir)
        self.max_points = max(100, int(max_points))
        self.live_plot = bool(live_plot)
        self.save_plots = bool(save_plots)
        self.save_csv = bool(save_csv)
        self.window_title = window_title

        self.last_report = 0
        self.last_plot = 0
        self.last_save = 0
        self.latest_info = {}

        # Environment-step series
        self.steps = []
        self.energy = []
        self.ergotropy = []
        self.delta_energy = []
        self.delta_ergotropy = []
        self.controls = []

        # Completed-episode series
        self.episode_steps = []
        self.episode_returns = []

        # SAC diagnostics sampled at plot refreshes
        self.diag_steps = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_coeffs = []

        self.app = None
        self.window = None
        self.plots = {}
        self.curves = {}
        self.status_label = None
        self._window_closed = False

    def _on_training_start(self) -> None:
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        if self.live_plot:
            try:
                # Creates/reuses QApplication. We intentionally do not call exec().
                self.app = pg.mkQApp(self.window_title)
                self._create_dashboard()
                self.window.show()
                self.app.processEvents()
            except Exception as exc:
                print(
                    f"[PyQtGraph] Dashboard unavailable ({exc}); "
                    "continuing with console logging and optional exports.",
                    flush=True,
                )
                self.live_plot = False

    def _create_dashboard(self) -> None:
        self.window = pg.GraphicsLayoutWidget(show=False, title=self.window_title)
        self.window.resize(1450, 900)
        self.window.setBackground("w")

        axis_pen = pg.mkPen(color=(70, 70, 70), width=1)
        grid_alpha = 0.22

        # Top-left: episode return
        p = self.window.addPlot(row=0, col=0, title="Episode return")
        p.setLabel("bottom", "Environment steps")
        p.setLabel("left", "Monitor episode return")
        p.showGrid(x=True, y=True, alpha=grid_alpha)
        p.addLegend()
        self.plots["return"] = p
        self.curves["return"] = p.plot(
            [], [], pen=pg.mkPen((31, 119, 180), width=2),
            name="Episode return"
        )

        # Top-right: physical observables
        p = self.window.addPlot(row=0, col=1, title="Physical observables")
        p.setLabel("bottom", "Environment steps")
        p.setLabel("left", "Value")
        p.showGrid(x=True, y=True, alpha=grid_alpha)
        p.addLegend()
        self.plots["physical"] = p
        self.curves["energy"] = p.plot(
            [], [], pen=pg.mkPen((31, 119, 180), width=2), name="TLS energy"
        )
        self.curves["ergotropy"] = p.plot(
            [], [], pen=pg.mkPen((255, 127, 14), width=2),
            name="Single-TLS ergotropy"
        )

        # Bottom-left: per-transition increments
        p = self.window.addPlot(row=1, col=0, title="Per-transition objective increments")
        p.setLabel("bottom", "Environment steps")
        p.setLabel("left", "Increment")
        p.showGrid(x=True, y=True, alpha=grid_alpha)
        p.addLegend()
        self.plots["increments"] = p
        self.curves["delta_energy"] = p.plot(
            [], [], pen=pg.mkPen((31, 119, 180), width=2), name="Δ energy"
        )
        self.curves["delta_ergotropy"] = p.plot(
            [], [], pen=pg.mkPen((255, 127, 14), width=2), name="Δ ergotropy"
        )

        # Bottom-right: optimizer diagnostics
        p = self.window.addPlot(row=1, col=1, title="SAC optimization diagnostics")
        p.setLabel("bottom", "Environment steps")
        p.setLabel("left", "Value")
        p.showGrid(x=True, y=True, alpha=grid_alpha)
        p.addLegend()
        self.plots["diagnostics"] = p
        self.curves["actor"] = p.plot(
            [], [], pen=pg.mkPen((31, 119, 180), width=2), name="Actor loss"
        )
        self.curves["critic"] = p.plot(
            [], [], pen=pg.mkPen((255, 127, 14), width=2), name="Critic loss"
        )
        self.curves["alpha"] = p.plot(
            [], [], pen=pg.mkPen((44, 160, 44), width=2), name="Entropy coefficient"
        )

        self.window.nextRow()
        self.status_label = self.window.addLabel(
            "Waiting for training data…", row=2, col=0, colspan=2
        )

        for plot in self.plots.values():
            plot.getAxis("bottom").setPen(axis_pen)
            plot.getAxis("left").setPen(axis_pen)

    def _logger_value(self, key: str) -> Optional[float]:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return None
        value = logger.name_to_value.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _append_bounded(series: list, value) -> None:
        series.append(value)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info in infos:
            if not info:
                continue

            self.latest_info = info
            step = int(self.num_timesteps)
            self.steps.append(step)
            self.energy.append(float(info.get("energy", np.nan)))
            self.ergotropy.append(float(info.get("ergotropy", np.nan)))
            self.controls.append(float(info.get("control", np.nan)))

            objectives = info.get("multi_obj")
            if objectives is None:
                self.delta_energy.append(np.nan)
                self.delta_ergotropy.append(np.nan)
            else:
                self.delta_energy.append(float(objectives[0]))
                self.delta_ergotropy.append(float(objectives[1]))

            episode = info.get("episode")
            if episode is not None:
                self.episode_steps.append(step)
                self.episode_returns.append(float(episode["r"]))

        if self.num_timesteps - self.last_report >= self.report_freq:
            self.last_report = self.num_timesteps
            self._print_report(dones)

        if self.num_timesteps - self.last_plot >= self.plot_freq:
            self.last_plot = self.num_timesteps
            self._sample_diagnostics()
            self._refresh_dashboard()

        if self.num_timesteps - self.last_save >= self.save_freq:
            self.last_save = self.num_timesteps
            self._save_outputs()

        return True

    def _sample_diagnostics(self) -> None:
        self.diag_steps.append(int(self.num_timesteps))
        self.actor_losses.append(
            self._logger_value("train/actor_loss")
        )
        self.critic_losses.append(
            self._logger_value("train/critic_loss")
        )
        self.entropy_coeffs.append(
            self._logger_value("train/ent_coef")
        )

    def _print_report(self, dones) -> None:
        info = self.latest_info
        n_done = int(np.sum(dones)) if len(dones) else 0

        print("\n" + "=" * 78)
        print(f"[SAC TRAINING] timestep: {self.num_timesteps:,}")
        print(f"  completed episodes in latest vector step: {n_done}")

        if info:
            objectives = info.get("multi_obj")
            weights = info.get("objective_weights")

            if objectives is not None:
                print(
                    f"  latest increments: ΔE={float(objectives[0]):+.6e}, "
                    f"Δergotropy={float(objectives[1]):+.6e}"
                )
            if weights is not None:
                print(
                    f"  reward weights: c_E={float(weights[0]):.5f}, "
                    f"c_ergotropy={float(weights[1]):.5f}"
                )
            for key in ("energy", "ergotropy", "control"):
                if key in info:
                    print(f"  latest {key}: {float(info[key]):.8f}")

            episode = info.get("episode")
            if episode is not None:
                print(
                    f"  completed episode: return={episode.get('r', np.nan):.6f}, "
                    f"length={episode.get('l', -1)}"
                )

        for label, key in (
            ("actor loss", "train/actor_loss"),
            ("critic loss", "train/critic_loss"),
            ("entropy coefficient", "train/ent_coef"),
        ):
            value = self._logger_value(key)
            if value is not None:
                print(f"  {label}: {value:.6g}")

        print("=" * 78, flush=True)

    def _tail(self, values: list):
        return values[-self.max_points:]

    def _refresh_dashboard(self) -> None:
        if not self.live_plot or self.window is None:
            return

        try:
            # Convert all plot data to float64 arrays.
            self.curves["return"].setData(
                numeric_array(self._tail(self.episode_steps)),
                numeric_array(self._tail(self.episode_returns))
            )

            self.curves["energy"].setData(
                numeric_array(self._tail(self.steps)),
                numeric_array(self._tail(self.energy))
            )

            self.curves["ergotropy"].setData(
                numeric_array(self._tail(self.steps)),
                numeric_array(self._tail(self.ergotropy))
            )

            self.curves["delta_energy"].setData(
                numeric_array(self._tail(self.steps)),
                numeric_array(self._tail(self.delta_energy))
            )

            self.curves["delta_ergotropy"].setData(
                numeric_array(self._tail(self.steps)),
                numeric_array(self._tail(self.delta_ergotropy))
            )

            for key, values in (
                ("actor", self.actor_losses),
                ("critic", self.critic_losses),
                ("alpha", self.entropy_coeffs),
            ):
                valid = [
                    (x, y)
                    for x, y in zip(self.diag_steps, values)
                    if y is not None and np.isfinite(y)
                ]

                if valid:
                    x_values, y_values = zip(*valid)

                    self.curves[key].setData(
                        np.asarray(x_values, dtype=np.float64),
                        np.asarray(y_values, dtype=np.float64)
                    )

            info = self.latest_info

            self.status_label.setText(
                f"Step {self.num_timesteps:,} | "
                f"u={float(info.get('control', np.nan)):.4f} | "
                f"E={float(info.get('energy', np.nan)):.5f} | "
                f"W={float(info.get('ergotropy', np.nan)):.5f}"
            )

            self.app.processEvents()

        except Exception as exc:
            print(f"[PyQtGraph] Refresh failed: {exc}", flush=True)

    def _save_outputs(self) -> None:
        if self.save_csv:
            self._save_csv()
        if self.save_plots and self.live_plot and self.window is not None:
            self._save_png()

    def _save_csv(self) -> None:
        path = self.plot_dir / "training_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "environment_step", "energy", "ergotropy",
                "delta_energy", "delta_ergotropy", "control"
            ])
            writer.writerows(zip(
                self.steps, self.energy, self.ergotropy,
                self.delta_energy, self.delta_ergotropy, self.controls
            ))

        episode_path = self.plot_dir / "episode_returns.csv"
        with episode_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["environment_step", "episode_return"])
            writer.writerows(zip(self.episode_steps, self.episode_returns))

    def _save_png(self) -> None:
        try:
            from pyqtgraph.exporters import ImageExporter

            exporter = ImageExporter(self.window.scene())
            exporter.parameters()["width"] = 1800
            exporter.export(str(self.plot_dir / "training_dashboard_latest.png"))
        except Exception as exc:
            print(f"[PyQtGraph] PNG export unavailable: {exc}", flush=True)

    def _on_training_end(self) -> None:
        self._save_outputs()
        if self.app is not None:
            self.app.processEvents()
        print(f"\n[SAC TRAINING] finished at {self.num_timesteps:,} timesteps.")
        print(f"[SAC TRAINING] dashboard outputs: {self.plot_dir}", flush=True)