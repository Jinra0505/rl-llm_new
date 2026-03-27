from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from mock_recovery_env import ProjectRecoveryEnv

LOGGER = logging.getLogger(__name__)


def _action_category(action: int) -> str:
    if action in {0, 1, 2}:
        return "road"
    if action in {3, 4, 5}:
        return "power"
    if action in {6, 7, 8}:
        return "comm"
    if action in {9, 10, 11}:
        return "mes"
    if action == 12:
        return "feeder"
    if action == 14:
        return "wait"
    return "coordinated"


class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.buf.append((s, a, r, ns, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(ns), np.array(d, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buf)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_revise_functions(module_path: Path) -> tuple[Callable[..., np.ndarray], Callable[..., float]]:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import revise module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    revise_state = getattr(module, "revise_state", None)
    intrinsic_reward = getattr(module, "intrinsic_reward", None)
    if not callable(revise_state) or not callable(intrinsic_reward):
        raise ValueError(f"Module {module_path} must define revise_state and intrinsic_reward")
    return revise_state, intrinsic_reward


def _call_revise(fn: Callable[..., Any], state: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    try:
        arr = np.asarray(fn(state, info), dtype=np.float32)
    except TypeError:
        try:
            arr = np.asarray(fn(state), dtype=np.float32)
        except Exception:  # noqa: BLE001
            arr = np.asarray(state, dtype=np.float32)
    except Exception:  # noqa: BLE001
        arr = np.asarray(state, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _call_intrinsic(fn: Callable[..., Any], s: np.ndarray, a: int, ns: np.ndarray, info: dict[str, Any], rs: np.ndarray) -> float:
    try:
        return float(fn(s, a, ns, info, rs))
    except TypeError:
        try:
            return float(fn(rs))
        except TypeError:
            return float(fn(ns))


def _effective_state(revised_state: np.ndarray, max_revised_dim: int | None) -> np.ndarray:
    """Policy input is revised_state (LLM can affect representation learning)."""
    return revised_state if max_revised_dim is None else revised_state[:max_revised_dim]


def _valid_action_mask(action_dim: int, info: dict[str, Any]) -> np.ndarray:
    mask = np.ones(action_dim, dtype=bool)
    stage = str(info.get("stage", "middle"))
    mes_soc = float(info.get("mes_soc", 1.0))
    material = float(info.get("material_stock", 1.0))
    backbone_comm = float(info.get("backbone_comm_ratio", 1.0))
    mean_comm = float(info.get("communication_recovery_ratio", 1.0))
    road_by_zone = [
        float(info.get("zone_A_road_ratio", 1.0)),
        float(info.get("zone_B_road_ratio", 1.0)),
        float(info.get("zone_C_road_ratio", 1.0)),
    ]
    resource_weak = material < 0.10 or mes_soc < 0.08

    # MES dispatch (9/10/11): invalid if mes_soc < 0.08 or target zone road < 0.25.
    if mes_soc < 0.08:
        mask[9:12] = False
    else:
        for zone_idx, road_ratio in enumerate(road_by_zone):
            if road_ratio < 0.25:
                mask[9 + zone_idx] = False

    # Material shortage: strongly suppress material-intensive actions.
    if material < 0.15:
        mask[0:9] = False
        mask[13] = False
    if material < 0.10:
        mask[9:12] = False
        mask[12] = False
        mask[13] = False

    # feeder (12): invalid when C0/backbone_comm < 0.30; also invalid under material shortage.
    if backbone_comm < 0.30 or material < 0.10:
        mask[12] = False

    # coordinated (13): suppress in low-resource / weak-comm contexts.
    if material < 0.10 or (backbone_comm < 0.30 and mean_comm < 0.35):
        mask[13] = False
    if stage == "late" and (resource_weak or backbone_comm < 0.35):
        mask[13] = False

    if stage == "late":
        mask[0:3] = False
        weak_layer = str(info.get("weakest_layer", "0"))
        if weak_layer == "0":
            mask[6:9] = False
        elif weak_layer == "1":
            mask[3:6] = False
    # Always keep wait_hold action feasible for safe resource preservation.
    if action_dim > 14:
        mask[14] = True
    return mask


def _selection_score(metrics: dict[str, Any], weights_cfg: dict[str, Any]) -> float:
    weights = dict(weights_cfg.get("__global__", {}))
    if not weights:
        weights = {
            "eval_success_rate": 0.40,
            "critical_load_recovery_ratio": 0.25,
            "min_recovery_ratio": 0.20,
            "constraint_violation_rate_eval": -0.35,
            "mean_progress_delta_eval": 0.08,
            "late_stage_targeted_action_rate": 0.08,
        }
    return float(sum(float(w) * float(metrics.get(k, 0.0)) for k, w in weights.items()))


def run_training(
    revise_module_path: Path | None,
    env_name: str,
    train_episodes: int,
    eval_episodes: int,
    max_steps_per_episode: int,
    gamma: float,
    task_mode: str,
    llm_mode: str,
    output_json_path: Path,
    seed: int = 42,
    max_revised_dim: int | None = None,
    task_mode_metric_weights: dict[str, Any] | None = None,
    dqn_cfg: dict[str, Any] | None = None,
    severity: str = "moderate",
) -> dict[str, Any]:
    if str(llm_mode).lower() != "real":
        raise RuntimeError(f"Formal run requires llm_mode=real, got: {llm_mode}")
    if env_name not in {"project_recovery", "mock_recovery"}:
        raise ValueError("Supported env names: project_recovery or mock_recovery")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    revise_module_path = revise_module_path or Path("baseline_noop.py")
    revise_state_fn, intrinsic_reward_fn = load_revise_functions(revise_module_path)

    env = ProjectRecoveryEnv(max_steps=max_steps_per_episode, seed=seed, severity=severity)
    action_dim = int(env.action_space.n)

    dqn_cfg = dqn_cfg or {}
    batch_size = int(dqn_cfg.get("batch_size", 64))
    replay_size = int(dqn_cfg.get("replay_size", 30000))
    min_replay_size = int(dqn_cfg.get("min_replay_size", 500))
    train_freq = int(dqn_cfg.get("train_freq", 1))
    target_update_interval = int(dqn_cfg.get("target_update_interval", 250))
    lr = float(dqn_cfg.get("learning_rate", 8e-4))
    hidden_dim = int(dqn_cfg.get("hidden_dim", 128))
    eps_start = float(dqn_cfg.get("epsilon_start", 1.0))
    eps_end = float(dqn_cfg.get("epsilon_end", 0.05))
    eps_decay_steps = int(dqn_cfg.get("epsilon_decay_steps", 5000))

    s0, info0 = env.reset(seed=seed)
    rs0 = _effective_state(_call_revise(revise_state_fn, s0, info0), max_revised_dim)
    state_dim = int(rs0.shape[0])

    q_net = QNet(state_dim, action_dim, hidden_dim)
    target_net = QNet(state_dim, action_dim, hidden_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(replay_size)

    global_step = 0
    episode_rewards: list[float] = []
    eval_rewards: list[float] = []
    action_usage = {str(i): 0 for i in range(action_dim)}

    successes = 0
    completion_steps: list[int] = []
    comm_scores: list[float] = []
    power_scores: list[float] = []
    road_scores: list[float] = []
    crit_scores: list[float] = []
    violations = 0
    train_total_steps = 0

    for ep in range(train_episodes):
        s, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_violation_count = 0
        mid_stagnation_steps = 0

        for step in range(max_steps_per_episode):
            rs = _effective_state(_call_revise(revise_state_fn, s, info), max_revised_dim)
            valid_mask = _valid_action_mask(action_dim, info)
            valid_actions = np.where(valid_mask)[0]
            if valid_actions.size == 0:
                if action_dim > 14:
                    valid_mask[14] = True
                    valid_actions = np.array([14], dtype=int)
                else:
                    raise RuntimeError("No feasible action under current mask and no wait action is available.")

            eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - global_step / float(max(1, eps_decay_steps)))
            if random.random() < eps:
                a = int(np.random.choice(valid_actions))
            else:
                with torch.no_grad():
                    qvals = q_net(torch.tensor(rs, dtype=torch.float32).unsqueeze(0))
                    qarr = qvals.squeeze(0).cpu().numpy()
                    qarr[~valid_mask] = -1e9
                    a = int(np.argmax(qarr))

            action_usage[str(a)] += 1
            ns, ext_r, terminated, truncated, info = env.step(a)
            ir = _call_intrinsic(intrinsic_reward_fn, s, a, ns, info, rs)
            critical_gain = float(np.mean(ns[9:12] - s[9:12]))
            progress_bonus = float(info.get("progress_delta", 0.0))
            prev_layers = [float(np.mean(s[0:3])), float(np.mean(s[3:6])), float(np.mean(s[6:9])), float(np.mean(s[9:12]))]
            next_layers = [float(np.mean(ns[0:3])), float(np.mean(ns[3:6])), float(np.mean(ns[6:9])), float(np.mean(ns[9:12]))]
            weak_layer_idx = int(np.argmin(prev_layers))
            weak_layer_gain = max(0.0, next_layers[weak_layer_idx] - prev_layers[weak_layer_idx])
            prev_zones = [float(np.mean([s[0], s[3], s[6], s[9]])), float(np.mean([s[1], s[4], s[7], s[10]])), float(np.mean([s[2], s[5], s[8], s[11]]))]
            next_zones = [float(np.mean([ns[0], ns[3], ns[6], ns[9]])), float(np.mean([ns[1], ns[4], ns[7], ns[10]])), float(np.mean([ns[2], ns[5], ns[8], ns[11]]))]
            weak_zone_idx = int(np.argmin(prev_zones))
            weak_zone_gain = max(0.0, next_zones[weak_zone_idx] - prev_zones[weak_zone_idx])
            prev_global = float(np.mean([prev_layers[0], prev_layers[1], prev_layers[2], prev_layers[3]]))
            next_global = float(np.mean([next_layers[0], next_layers[1], next_layers[2], next_layers[3]]))
            milestone_bonus = 0.0
            for th, bonus in ((0.75, 0.12), (0.85, 0.18), (0.90, 0.28)):
                if prev_global < th <= next_global:
                    milestone_bonus += bonus
            stage_prev = str(info.get("stage", "middle"))
            stage_next = "late" if float(ns[22]) >= 0.99 else ("middle" if float(ns[22]) >= 0.49 else "early")
            enter_late_bonus = 0.8 if (stage_prev != "late" and stage_next == "late") else 0.0
            step_ratio = float(step + 1) / float(max_steps_per_episode)
            late_factor = 1.0 + max(0.0, (step_ratio - 0.60) / 0.40) * 1.4
            invalid_penalty = (0.20 + 0.35 * late_factor) if bool(info.get("invalid_action", False)) else 0.0
            constraint_penalty = (0.25 + 0.45 * late_factor) if bool(info.get("constraint_violation", False)) else 0.0
            completion_bonus = 6.5 if bool(terminated) else 0.0
            late_stage = str(info.get("stage", "middle")) == "late"
            coordinated_late_penalty = 0.35 if (late_stage and a == 13) else 0.0
            feeder_late_penalty = 0.22 if (late_stage and a == 12 and (float(info.get("backbone_comm_ratio", 1.0)) < 0.5)) else 0.0
            targeted_late_bonus = 0.12 if (late_stage and a in {3, 4, 5, 6, 7, 8}) else 0.0
            weakest_close_bonus = 0.18 * (weak_layer_gain + weak_zone_gain) if late_stage else 0.0
            low_violation_finish_bonus = 0.8 if (terminated and ep_violation_count <= 1) else 0.0
            material_now = float(info.get("material_stock", ns[20] if len(ns) > 20 else 0.0))
            material_zero_penalty = 0.8 if material_now <= 0.01 else 0.0
            critical_low_material_penalty = 0.35 if (material_now < 0.10 and a != 14) else 0.0
            material_buffer_bonus = 0.06 if (material_now > 0.22 and progress_bonus > 0.0) else 0.0
            wait_misuse_penalty = 0.18 if (a == 14 and material_now > 0.22 and progress_bonus < 0.0008) else 0.0
            if str(info.get("stage", "middle")) == "middle" and progress_bonus < 0.0015:
                mid_stagnation_steps += 1
            else:
                mid_stagnation_steps = 0
            middle_stagnation_penalty = 0.16 if mid_stagnation_steps >= 8 else 0.0
            r = float(
                ext_r
                + ir
                + 0.25 * critical_gain
                + 0.32 * progress_bonus
                + 0.20 * weak_layer_gain
                + 0.18 * weak_zone_gain
                + milestone_bonus
                + enter_late_bonus
                + completion_bonus
                + targeted_late_bonus
                + weakest_close_bonus
                + low_violation_finish_bonus
                + material_buffer_bonus
                - invalid_penalty
                - constraint_penalty
                - coordinated_late_penalty
                - feeder_late_penalty
                - material_zero_penalty
                - critical_low_material_penalty
                - middle_stagnation_penalty
                - wait_misuse_penalty
            )
            done = bool(terminated or truncated)

            nrs = _effective_state(_call_revise(revise_state_fn, ns, info), max_revised_dim)
            replay.add(rs, a, r, nrs, done)

            if len(replay) >= min_replay_size and global_step % train_freq == 0:
                bs, ba, br, bns, bd = replay.sample(batch_size)
                bs_t = torch.tensor(bs, dtype=torch.float32)
                ba_t = torch.tensor(ba, dtype=torch.int64).unsqueeze(1)
                br_t = torch.tensor(br, dtype=torch.float32).unsqueeze(1)
                bns_t = torch.tensor(bns, dtype=torch.float32)
                bd_t = torch.tensor(bd, dtype=torch.float32).unsqueeze(1)

                q = q_net(bs_t).gather(1, ba_t)
                with torch.no_grad():
                    max_next_q = target_net(bns_t).max(dim=1, keepdim=True).values
                    target = br_t + gamma * (1.0 - bd_t) * max_next_q
                loss = nn.functional.smooth_l1_loss(q, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

            if global_step % target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            ep_reward += r
            s = ns
            global_step += 1
            train_total_steps += 1

            if info.get("constraint_violation", False):
                violations += 1
                ep_violation_count += 1
            if done:
                if terminated:
                    successes += 1
                    completion_steps.append(step + 1)
                break

        episode_rewards.append(ep_reward)

    # Evaluation (greedy)
    eval_backbone_comm: list[float] = []
    eval_backbone_power: list[float] = []
    eval_backbone_road: list[float] = []
    eval_invalid_rates: list[float] = []
    eval_violation_rates: list[float] = []
    eval_progress_deltas: list[float] = []
    eval_stage_indicators: list[float] = []
    eval_mes_usage_rates: list[float] = []
    eval_mes_soc_end: list[float] = []
    eval_material_end: list[float] = []
    eval_switching_end: list[float] = []
    eval_crew_power_end: list[float] = []
    eval_crew_comm_end: list[float] = []
    eval_crew_road_end: list[float] = []
    eval_zone_A_power: list[float] = []
    eval_zone_B_power: list[float] = []
    eval_zone_C_power: list[float] = []
    eval_zone_A_comm: list[float] = []
    eval_zone_B_comm: list[float] = []
    eval_zone_C_comm: list[float] = []
    eval_zone_A_road: list[float] = []
    eval_zone_B_road: list[float] = []
    eval_zone_C_road: list[float] = []
    eval_zone_A_load: list[float] = []
    eval_zone_B_load: list[float] = []
    eval_zone_C_load: list[float] = []
    eval_steps_per_episode: list[int] = []
    eval_terminated_count = 0
    eval_truncated_count = 0
    eval_action_usage = {str(i): 0 for i in range(action_dim)}
    eval_stage_counts: Counter[str] = Counter()
    eval_invalid_reason_counts: Counter[str] = Counter()
    eval_weakest_zone_counts: Counter[str] = Counter()
    eval_weakest_layer_counts: Counter[str] = Counter()
    late_stage_steps_total = 0
    late_stage_targeted_steps = 0
    late_stage_coordinated_steps = 0
    representative_eval_trace: list[dict[str, Any]] = []
    representative_eval_summary: dict[str, Any] = {}

    for ep in range(eval_episodes):
        s, info = env.reset(seed=seed + 1000 + ep)
        total = 0.0
        ep_steps = 0
        ep_invalid = 0
        ep_violate = 0
        ep_progress: list[float] = []
        ep_stage: list[float] = []
        ep_mes_moves = 0
        terminated = False
        truncated = False
        episode_trace: list[dict[str, Any]] = []
        for step_idx in range(max_steps_per_episode):
            rs = _effective_state(_call_revise(revise_state_fn, s, info), max_revised_dim)
            valid_mask = _valid_action_mask(action_dim, info)
            valid_actions = np.where(valid_mask)[0]
            if valid_actions.size == 0:
                if action_dim > 14:
                    valid_mask[14] = True
                else:
                    raise RuntimeError("No feasible action during eval and no wait action is available.")
            with torch.no_grad():
                qvals = q_net(torch.tensor(rs, dtype=torch.float32).unsqueeze(0))
                qarr = qvals.squeeze(0).cpu().numpy()
                qarr[~valid_mask] = -1e9
                a = int(np.argmax(qarr))
            eval_action_usage[str(a)] += 1
            ns, ext_r, terminated, truncated, info = env.step(a)
            total += float(ext_r)
            ep_steps += 1
            ep_invalid += int(bool(info.get("invalid_action", False)))
            ep_violate += int(bool(info.get("constraint_violation", False)))
            if info.get("invalid_action", False):
                eval_invalid_reason_counts[str(info.get("invalid_reason", "unknown"))] += 1
            ep_progress.append(float(info.get("progress_delta", 0.0)))
            ep_stage.append(float(info.get("stage_indicator", 0.0)))
            eval_stage_counts[str(info.get("stage", "unknown"))] += 1
            eval_weakest_zone_counts[str(info.get("weakest_zone", "A"))] += 1
            eval_weakest_layer_counts[str(info.get("weakest_layer", "0"))] += 1
            if str(info.get("stage", "middle")) == "late":
                late_stage_steps_total += 1
                if a in {3, 4, 5, 6, 7, 8}:
                    late_stage_targeted_steps += 1
                if a == 13:
                    late_stage_coordinated_steps += 1
            if a in {9, 10, 11}:
                ep_mes_moves += 1
            if ep == 0 and step_idx < 12:
                episode_trace.append(
                    {
                        "step": step_idx,
                        "action": a,
                        "progress_delta": float(info.get("progress_delta", 0.0)),
                        "stage": str(info.get("stage", "unknown")),
                        "invalid_action": bool(info.get("invalid_action", False)),
                        "invalid_reason": str(info.get("invalid_reason", "")),
                        "constraint_violation": bool(info.get("constraint_violation", False)),
                    }
                )
            s = ns
            if terminated or truncated:
                break

        eval_rewards.append(total)
        eval_steps_per_episode.append(ep_steps)
        eval_terminated_count += int(bool(terminated))
        eval_truncated_count += int(bool(truncated))
        comm_scores.append(float(info.get("communication_recovery_ratio", 0.0)))
        power_scores.append(float(info.get("power_recovery_ratio", 0.0)))
        road_scores.append(float(info.get("road_recovery_ratio", 0.0)))
        crit_scores.append(float(info.get("critical_load_recovery_ratio", 0.0)))
        eval_backbone_comm.append(float(info.get("backbone_comm_ratio", info.get("communication_recovery_ratio", 0.0))))
        eval_backbone_power.append(float(info.get("backbone_power_ratio", info.get("power_recovery_ratio", 0.0))))
        eval_backbone_road.append(float(info.get("backbone_road_ratio", info.get("road_recovery_ratio", 0.0))))
        eval_invalid_rates.append(ep_invalid / float(max(1, ep_steps)))
        eval_violation_rates.append(ep_violate / float(max(1, ep_steps)))
        eval_progress_deltas.append(float(np.mean(ep_progress)) if ep_progress else 0.0)
        eval_stage_indicators.append(float(np.mean(ep_stage)) if ep_stage else 0.0)
        eval_mes_usage_rates.append(ep_mes_moves / float(max(1, ep_steps)))
        eval_mes_soc_end.append(float(info.get("mes_soc", 0.0)))
        eval_material_end.append(float(info.get("material_stock", 0.0)))
        eval_switching_end.append(float(info.get("switching_capability", 0.0)))
        eval_crew_power_end.append(float(info.get("crew_power_status", 0.0)))
        eval_crew_comm_end.append(float(info.get("crew_comm_status", 0.0)))
        eval_crew_road_end.append(float(info.get("crew_road_status", 0.0)))
        eval_zone_A_power.append(float(info.get("zone_A_power_ratio", 0.0)))
        eval_zone_B_power.append(float(info.get("zone_B_power_ratio", 0.0)))
        eval_zone_C_power.append(float(info.get("zone_C_power_ratio", 0.0)))
        eval_zone_A_comm.append(float(info.get("zone_A_comm_ratio", 0.0)))
        eval_zone_B_comm.append(float(info.get("zone_B_comm_ratio", 0.0)))
        eval_zone_C_comm.append(float(info.get("zone_C_comm_ratio", 0.0)))
        eval_zone_A_road.append(float(info.get("zone_A_road_ratio", 0.0)))
        eval_zone_B_road.append(float(info.get("zone_B_road_ratio", 0.0)))
        eval_zone_C_road.append(float(info.get("zone_C_road_ratio", 0.0)))
        eval_zone_A_load.append(float(info.get("zone_A_critical_load_ratio", 0.0)))
        eval_zone_B_load.append(float(info.get("zone_B_critical_load_ratio", 0.0)))
        eval_zone_C_load.append(float(info.get("zone_C_critical_load_ratio", 0.0)))
        if ep == 0:
            representative_eval_trace = episode_trace
            representative_eval_summary = {
                "steps": ep_steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_stage": str(info.get("stage", "unknown")),
                "final_progress_delta": float(info.get("progress_delta", 0.0)),
                "final_critical_load_shortfall": float(info.get("critical_load_shortfall", 1.0)),
            }

    total_actions = max(1, sum(action_usage.values()))
    action_usage_norm = {k: v / total_actions for k, v in action_usage.items()}
    action_category_usage = {"road": 0.0, "power": 0.0, "comm": 0.0, "mes": 0.0, "feeder": 0.0, "coordinated": 0.0, "wait": 0.0}
    for a_str, frac in action_usage_norm.items():
        action_category_usage[_action_category(int(a_str))] += float(frac)
    dominant_action_category = max(action_category_usage.items(), key=lambda kv: kv[1])[0]

    eval_success_rate = eval_terminated_count / float(max(1, eval_episodes))
    min_recovery_ratio = min(
        float(np.mean(power_scores)) if power_scores else 0.0,
        float(np.mean(comm_scores)) if comm_scores else 0.0,
        float(np.mean(road_scores)) if road_scores else 0.0,
        float(np.mean(crit_scores)) if crit_scores else 0.0,
    )

    result = {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "train_success_rate": successes / float(max(1, train_episodes)),
        "eval_success_rate": eval_success_rate,
        "success_rate": eval_success_rate,
        "communication_recovery_ratio": float(np.mean(comm_scores)) if comm_scores else 0.0,
        "power_recovery_ratio": float(np.mean(power_scores)) if power_scores else 0.0,
        "road_recovery_ratio": float(np.mean(road_scores)) if road_scores else 0.0,
        "critical_load_recovery_ratio": float(np.mean(crit_scores)) if crit_scores else 0.0,
        "min_recovery_ratio": min_recovery_ratio,
        "critical_load_shortfall": float(max(0.0, 1.0 - (float(np.mean(crit_scores)) if crit_scores else 0.0))),
        "recovery_completion_time": float(np.mean(completion_steps)) if completion_steps else float(max_steps_per_episode),
        "cumulative_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "constraint_violation_count": int(violations),
        "constraint_violation_rate": float(np.mean(eval_violation_rates)) if eval_violation_rates else 0.0,
        "constraint_violation_rate_eval": float(np.mean(eval_violation_rates)) if eval_violation_rates else 0.0,
        "train_constraint_violation_rate": float(violations) / float(max(1, train_total_steps)),
        "invalid_action_rate": float(np.mean(eval_invalid_rates)) if eval_invalid_rates else 0.0,
        "invalid_action_rate_eval": float(np.mean(eval_invalid_rates)) if eval_invalid_rates else 0.0,
        "mean_progress_delta_eval": float(np.mean(eval_progress_deltas)) if eval_progress_deltas else 0.0,
        "mean_progress_delta": float(np.mean(eval_progress_deltas)) if eval_progress_deltas else 0.0,
        "mean_stage_indicator_eval": float(np.mean(eval_stage_indicators)) if eval_stage_indicators else 0.0,
        "backbone_comm_ratio": float(np.mean(eval_backbone_comm)) if eval_backbone_comm else 0.0,
        "backbone_power_ratio": float(np.mean(eval_backbone_power)) if eval_backbone_power else 0.0,
        "backbone_road_ratio": float(np.mean(eval_backbone_road)) if eval_backbone_road else 0.0,
        "mes_usage_rate": float(np.mean(eval_mes_usage_rates)) if eval_mes_usage_rates else 0.0,
        "mes_usage_rate_eval": float(np.mean(eval_mes_usage_rates)) if eval_mes_usage_rates else 0.0,
        "mes_soc_mean_end": float(np.mean(eval_mes_soc_end)) if eval_mes_soc_end else 0.0,
        "mes_soc_end_mean": float(np.mean(eval_mes_soc_end)) if eval_mes_soc_end else 0.0,
        "material_stock_mean_end": float(np.mean(eval_material_end)) if eval_material_end else 0.0,
        "material_stock_end_mean": float(np.mean(eval_material_end)) if eval_material_end else 0.0,
        "switching_capability_mean_end": float(np.mean(eval_switching_end)) if eval_switching_end else 0.0,
        "switching_capability_end_mean": float(np.mean(eval_switching_end)) if eval_switching_end else 0.0,
        "crew_power_status_mean_end": float(np.mean(eval_crew_power_end)) if eval_crew_power_end else 0.0,
        "crew_power_status_end_mean": float(np.mean(eval_crew_power_end)) if eval_crew_power_end else 0.0,
        "crew_comm_status_mean_end": float(np.mean(eval_crew_comm_end)) if eval_crew_comm_end else 0.0,
        "crew_comm_status_end_mean": float(np.mean(eval_crew_comm_end)) if eval_crew_comm_end else 0.0,
        "crew_road_status_mean_end": float(np.mean(eval_crew_road_end)) if eval_crew_road_end else 0.0,
        "crew_road_status_end_mean": float(np.mean(eval_crew_road_end)) if eval_crew_road_end else 0.0,
        "zone_A_power_ratio": float(np.mean(eval_zone_A_power)) if eval_zone_A_power else 0.0,
        "zone_B_power_ratio": float(np.mean(eval_zone_B_power)) if eval_zone_B_power else 0.0,
        "zone_C_power_ratio": float(np.mean(eval_zone_C_power)) if eval_zone_C_power else 0.0,
        "zone_A_comm_ratio": float(np.mean(eval_zone_A_comm)) if eval_zone_A_comm else 0.0,
        "zone_B_comm_ratio": float(np.mean(eval_zone_B_comm)) if eval_zone_B_comm else 0.0,
        "zone_C_comm_ratio": float(np.mean(eval_zone_C_comm)) if eval_zone_C_comm else 0.0,
        "zone_A_road_ratio": float(np.mean(eval_zone_A_road)) if eval_zone_A_road else 0.0,
        "zone_B_road_ratio": float(np.mean(eval_zone_B_road)) if eval_zone_B_road else 0.0,
        "zone_C_road_ratio": float(np.mean(eval_zone_C_road)) if eval_zone_C_road else 0.0,
        "zone_A_critical_load_ratio": float(np.mean(eval_zone_A_load)) if eval_zone_A_load else 0.0,
        "zone_B_critical_load_ratio": float(np.mean(eval_zone_B_load)) if eval_zone_B_load else 0.0,
        "zone_C_critical_load_ratio": float(np.mean(eval_zone_C_load)) if eval_zone_C_load else 0.0,
        "task_mode_used": task_mode,
        "llm_mode_used": "real",
        "revise_module_path": str(revise_module_path),
        "policy_feature_dim_used": state_dim,
        "env_name": env_name,
        "severity": severity,
        "action_usage": action_usage_norm,
        "action_category_usage": action_category_usage,
        "dominant_action_category": dominant_action_category,
        "late_stage_targeted_action_rate": float(late_stage_targeted_steps) / float(max(1, late_stage_steps_total)),
        "late_stage_coordinated_action_rate": float(late_stage_coordinated_steps) / float(max(1, late_stage_steps_total)),
        "weakest_zone": max(eval_weakest_zone_counts.items(), key=lambda kv: kv[1])[0] if eval_weakest_zone_counts else "A",
        "weakest_layer": max(eval_weakest_layer_counts.items(), key=lambda kv: kv[1])[0] if eval_weakest_layer_counts else "0",
        "weakest_zone_frequency": dict(eval_weakest_zone_counts),
        "stage_distribution_eval": {
            k: v / float(max(1, sum(eval_stage_counts.values()))) for k, v in eval_stage_counts.items()
        },
        "stage_distribution": {
            k: v / float(max(1, sum(eval_stage_counts.values()))) for k, v in eval_stage_counts.items()
        },
        "invalid_reason_counts_eval": dict(eval_invalid_reason_counts),
        "representative_eval_trace": representative_eval_trace,
        "representative_eval_summary": representative_eval_summary,
        "eval_trajectory_summary": {
            "mean_steps": float(np.mean(eval_steps_per_episode)) if eval_steps_per_episode else 0.0,
            "terminated_rate": eval_terminated_count / float(max(1, eval_episodes)),
            "truncated_rate": eval_truncated_count / float(max(1, eval_episodes)),
            "mean_invalid_action_rate": float(np.mean(eval_invalid_rates)) if eval_invalid_rates else 0.0,
            "mean_constraint_violation_rate": float(np.mean(eval_violation_rates)) if eval_violation_rates else 0.0,
            "mean_progress_delta": float(np.mean(eval_progress_deltas)) if eval_progress_deltas else 0.0,
            "eval_action_usage": {
                k: v / float(max(1, sum(eval_action_usage.values()))) for k, v in eval_action_usage.items()
            },
        },
    }

    weights_cfg = task_mode_metric_weights or {}
    result["selection_score"] = _selection_score(result, weights_cfg=weights_cfg)
    result["selection_metric_used"] = "global_objective_score"

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("Saved RL results to %s", output_json_path)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Lightweight DQN trainer for project recovery env.")
    parser.add_argument("--env", default="project_recovery")
    parser.add_argument("--revise-module", default="")
    parser.add_argument("--task-mode", default="global_efficiency_priority")
    parser.add_argument("--llm-mode", default="real")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="outputs/rl_result.json")
    args = parser.parse_args()
    if str(args.llm_mode).lower() != "real":
        raise RuntimeError(f"Formal run requires llm_mode=real, got: {args.llm_mode}")

    cfg = load_yaml(Path(args.config))
    tr = cfg["training"]
    max_dim = cfg.get("state_representation", {}).get("max_revised_dim", None)
    weights = cfg.get("selection", {}).get("task_mode_metric_weights", {})
    revise_module_path = Path(args.revise_module) if args.revise_module else None

    run_training(
        revise_module_path=revise_module_path,
        env_name=args.env,
        train_episodes=int(tr["train_episodes"]),
        eval_episodes=int(tr["eval_episodes"]),
        max_steps_per_episode=int(cfg["env"].get("max_steps", 60)),
        gamma=float(tr["gamma"]),
        task_mode=args.task_mode,
        llm_mode=args.llm_mode,
        output_json_path=Path(args.output),
        max_revised_dim=int(max_dim) if max_dim is not None else None,
        task_mode_metric_weights=weights,
        dqn_cfg=tr,
        severity=str(cfg.get("scenario", {}).get("severity", "moderate")),
    )


if __name__ == "__main__":
    main()
