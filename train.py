import yaml
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.battery_env import DickeBatteryOneControlDetuning
from src.callbacks import QuantumBatteryScheduleCallback, PyQtGraphQuantumBatteryCallback

# Load the YAML configuration file
with open("training_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract variables and dictionaries
nq = config["nq"]
tau = config["tau"]
dt = config["dt"]
env_params = config["env_params"]
training_hyperparams = config["training_hyperparams"]


log_dir = Path("./dicke_sac_logs")
model_dir = log_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# The schedule is passed into the environment. Monitor records episode returns.
quantum_env = DickeBatteryOneControlDetuning(
    env_params,
    reward_schedule={
        "C_START": training_hyperparams["C_START"],
        "C_END": training_hyperparams["C_END"],
        "C_MEAN": training_hyperparams["C_MEAN"],
        "C_WIDTH": training_hyperparams["C_WIDTH"],
    },
)
check_env(quantum_env, warn=True)

env = Monitor(quantum_env)

policy_kwargs = {
    "net_arch": {
        "pi": list(training_hyperparams["HIDDEN_SIZES"]),
        "qf": list(training_hyperparams["HIDDEN_SIZES"]),
    }
}

device = "cuda" if training_hyperparams["USE_CUDA"] else "cpu"

model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=training_hyperparams["LR"],
    buffer_size=training_hyperparams["REPLAY_MEMORY_SIZE"],
    learning_starts=training_hyperparams["INITIAL_RANDOM_STEPS"],
    batch_size=training_hyperparams["BATCH_SIZE"],
    tau=1.0 - training_hyperparams["POLYAK"],  # SB3 tau is 1 - Polyak
    gamma=training_hyperparams["GAMMA"],
    train_freq=(50, "step"),
    gradient_steps=50,  # 50 gradient updates every 50 env steps
    ent_coef="auto",
    target_entropy=training_hyperparams["H_START"],
    policy_kwargs=policy_kwargs,
    tensorboard_log=str(log_dir / "tensorboard"),
    device=device,
    verbose=2,
    # Keep False: SB3 rejects optimize_memory_usage=True together with timeout handling.
    optimize_memory_usage=False,
)


# SB3 uses its main learning_rate for the entropy optimizer too. Adjust it
# explicitly to retain the original ALPHA_LR as closely as possible.
if model.ent_coef_optimizer is not None:
    for group in model.ent_coef_optimizer.param_groups:
        group["lr"] = training_hyperparams["ALPHA_LR"]

schedule_callback = QuantumBatteryScheduleCallback(
    quantum_env,
    h_start=training_hyperparams["H_START"],
    h_end=training_hyperparams["H_END"],
    h_decay=training_hyperparams["H_DECAY"],
)
checkpoint_callback = CheckpointCallback(
    save_freq=training_hyperparams["CHECKPOINT_FREQ"],
    save_path=str(model_dir),
    name_prefix="dicke_sac",
    save_replay_buffer=True,
)
pg_callback = PyQtGraphQuantumBatteryCallback(
    report_freq=training_hyperparams["REPORT_FREQ"],
    plot_freq=training_hyperparams["PLOT_FREQ"],
    save_freq=training_hyperparams["SAVE_FREQ"],
    max_points=training_hyperparams["MAX_POINTS"],
    plot_dir=str(log_dir / "plots"),
    live_plot=training_hyperparams["LIVE_PLOTS"],
    save_plots=training_hyperparams["SAVE_PLOTS"],
    save_csv=training_hyperparams["SAVE_CSV"]
)

model.learn(
    total_timesteps=training_hyperparams["TOTAL_TIMESTEPS"],
    callback=[schedule_callback, checkpoint_callback, pg_callback],
    log_interval=10,
    tb_log_name="dicke_battery_sac",
)

model.save(str(model_dir / "dicke_battery_sac_final"))
model.save_replay_buffer(str(model_dir / "dicke_battery_replay_buffer"))

# ---------------------------------------------------------------------------
# Deterministic evaluation: final physical energy and ergotropy
# ---------------------------------------------------------------------------

def evaluate_policy(model: SAC, env_params: dict, episodes: int = 5):
    eval_env = DickeBatteryOneControlDetuning(env_params)
    final_results = []

    for episode in range(episodes):
        obs, _ = eval_env.reset(seed=episode)
        terminated = truncated = False
        episode_return = 0.0
        controls = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_return += reward
            controls.append(float(action[0]))

        final_results.append({
            "episode": episode,
            "return": episode_return,
            "final_energy": eval_env.current_energy(),
            "final_ergotropy": eval_env.current_ergotropy(),
            "controls": np.asarray(controls),
        })

    return final_results


if __name__ == "__main__":
    results = evaluate_policy(model, env_params, episodes=5)
    for result in results:
        print(
            f"Episode {result['episode']}: "
            f"return={result['return']:.6f}, "
            f"energy={result['final_energy']:.6f}, "
            f"ergotropy={result['final_ergotropy']:.6f}"
        )