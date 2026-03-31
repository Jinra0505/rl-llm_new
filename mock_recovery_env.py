from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ProjectRecoveryEnv(gym.Env):
    """Project-grade reduced-order zonal tri-layer coupled recovery environment.

    Zones: A, B, C
    Layers per zone: power P_*, communication C_*, road R_*, critical-load L_*
    Backbone/global: P0, C0, R0

    Observation index map (continuous, [0,1] except mes_location in [0,2]):
      0-2   : P_A, P_B, P_C
      3-5   : C_A, C_B, C_C
      6-8   : R_A, R_B, R_C
      9-11  : L_A, L_B, L_C
      12-14 : P0, C0, R0
      15    : crew_power_status
      16    : crew_comm_status
      17    : crew_road_status
      18    : mes_location (0=A,1=B,2=C; normalized in state to /2)
      19    : mes_soc
      20    : material_stock
      21    : switching_capability
      22    : stage_indicator (0 early, 0.5 middle, 1 late)
      23    : constraint_flag

    Actions (Discrete 15):
      0-2 road_A/B/C, 3-5 power_A/B/C, 6-8 comm_A/B/C,
      9-11 mes_to_A/B/C, 12 feeder_reconfigure, 13 coordinated_balanced, 14 wait_hold
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 60,
        seed: int | None = None,
        severity: str = "moderate",
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.severity = severity

        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(24,), dtype=np.float32)

        self.w = reward_weights or {
            "delta_power": 0.20,
            "delta_comm": 0.20,
            "delta_road": 0.15,
            "delta_critical_load": 0.20,
            "synergy_bonus": 0.10,
            "constraint_penalty": 0.08,
            "action_switch_penalty": 0.04,
            "mes_overuse_penalty": 0.03,
        }

        self.state = np.zeros(24, dtype=np.float32)
        self.step_count = 0
        self.constraint_violation_count = 0
        self.prev_action = 13
        self.prev_action_category = "coordinated"

    def _severity_profile(self) -> tuple[tuple[float, float], float]:
        if self.severity == "mild":
            return (0.40, 0.70), 1.00
        if self.severity == "severe":
            return (0.05, 0.35), 0.75
        return (0.20, 0.55), 0.88  # moderate

    def _progress(self, s: np.ndarray) -> float:
        power = np.mean(np.concatenate([s[0:3], [s[12]]]))
        comm = np.mean(np.concatenate([s[3:6], [s[13]]]))
        road = np.mean(np.concatenate([s[6:9], [s[14]]]))
        return float(0.35 * power + 0.35 * comm + 0.30 * road)

    def _stage(self, s: np.ndarray) -> tuple[float, str]:
        p = self._progress(s)
        if p < 0.35:
            return 0.0, "early"
        if p < 0.75:
            return 0.5, "middle"
        return 1.0, "late"

    def _clip01(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, 0.0, 1.0)

    def _action_category(self, action: int) -> str:
        if 0 <= action <= 2:
            return "road"
        if 3 <= action <= 5:
            return "power"
        if 6 <= action <= 8:
            return "comm"
        if 9 <= action <= 11:
            return "mes"
        if action == 12:
            return "feeder"
        if action == 14:
            return "wait"
        return "coordinated"

    def _apply_preset(self, s: np.ndarray, preset_name: str, jitter: float = 0.0) -> np.ndarray:
        presets: dict[str, dict[str, Any]] = {
            "critical_load_dominant": {
                "power": [0.48, 0.52, 0.49],
                "comm": [0.58, 0.60, 0.57],
                "road": [0.57, 0.58, 0.56],
                "critical": [0.45, 0.42, 0.43],
                "backbone": [0.55, 0.60, 0.58],
                "mes_soc": 0.58,
                "material": 0.38,
                "switching": 0.62,
            },
            "capability_bottleneck_dominant": {
                "power": [0.55, 0.58, 0.56],
                "comm": [0.50, 0.52, 0.51],
                "road": [0.40, 0.42, 0.41],
                "critical": [0.66, 0.64, 0.63],
                "backbone": [0.47, 0.45, 0.44],
                "mes_soc": 0.52,
                "material": 0.18,
                "switching": 0.48,
            },
            "global_finishing_dominant": {
                "power": [0.72, 0.73, 0.71],
                "comm": [0.70, 0.72, 0.71],
                "road": [0.69, 0.70, 0.68],
                "critical": [0.85, 0.87, 0.86],
                "backbone": [0.74, 0.73, 0.72],
                "mes_soc": 0.60,
                "material": 0.33,
                "switching": 0.72,
            },
            "uncertain_boundary_case_u1": {
                "power": [0.53, 0.54, 0.52],
                "comm": [0.45, 0.47, 0.46],
                "road": [0.39, 0.41, 0.40],
                "critical": [0.62, 0.60, 0.58],
                "backbone": [0.43, 0.44, 0.42],
                "mes_soc": 0.50,
                "material": 0.15,
                "switching": 0.46,
            },
            "uncertain_boundary_case_u2": {
                "power": [0.50, 0.51, 0.49],
                "comm": [0.56, 0.57, 0.55],
                "road": [0.55, 0.56, 0.54],
                "critical": [0.54, 0.52, 0.51],
                "backbone": [0.57, 0.58, 0.56],
                "mes_soc": 0.62,
                "material": 0.34,
                "switching": 0.63,
            },
            "definition_shift_case_d1": {
                "power": [0.69, 0.70, 0.68],
                "comm": [0.67, 0.69, 0.68],
                "road": [0.65, 0.67, 0.66],
                "critical": [0.83, 0.84, 0.82],
                "backbone": [0.70, 0.69, 0.68],
                "mes_soc": 0.59,
                "material": 0.31,
                "switching": 0.70,
            },
            "definition_shift_case_d2": {
                "power": [0.57, 0.58, 0.56],
                "comm": [0.30, 0.33, 0.31],
                "road": [0.28, 0.30, 0.29],
                "critical": [0.70, 0.71, 0.69],
                "backbone": [0.32, 0.34, 0.31],
                "mes_soc": 0.55,
                "material": 0.24,
                "switching": 0.40,
            },
        }
        if preset_name not in presets:
            return s
        p = presets[preset_name]
        s[0:3] = np.asarray(p["power"], dtype=np.float32)
        s[3:6] = np.asarray(p["comm"], dtype=np.float32)
        s[6:9] = np.asarray(p["road"], dtype=np.float32)
        s[9:12] = np.asarray(p["critical"], dtype=np.float32)
        s[12:15] = np.asarray(p["backbone"], dtype=np.float32)
        s[19] = float(p["mes_soc"])
        s[20] = float(p["material"])
        s[21] = float(p["switching"])
        if jitter > 0.0:
            noise = self.rng.normal(0.0, jitter, size=15).astype(np.float32)
            s[0:15] = np.clip(s[0:15] + noise, 0.0, 1.0)
            s[19] = float(np.clip(s[19] + self.rng.normal(0.0, jitter * 0.5), 0.0, 1.0))
            s[20] = float(np.clip(s[20] + self.rng.normal(0.0, jitter * 0.5), 0.0, 1.0))
            s[21] = float(np.clip(s[21] + self.rng.normal(0.0, jitter * 0.5), 0.0, 1.0))
        return s

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.constraint_violation_count = 0
        self.prev_action = 13
        self.prev_action_category = "coordinated"

        rng_range, difficulty = self._severity_profile()
        low, high = rng_range

        s = np.zeros(24, dtype=np.float32)
        s[0:3] = self.rng.uniform(low, high, size=3)  # P_A/B/C
        s[3:6] = self.rng.uniform(low, high, size=3)  # C_A/B/C
        s[6:9] = self.rng.uniform(low, high, size=3)  # R_A/B/C
        s[9:12] = self.rng.uniform(low, high, size=3)  # L_A/B/C
        s[12] = self.rng.uniform(low, high)  # P0
        s[13] = self.rng.uniform(low, high)  # C0
        s[14] = self.rng.uniform(low, high)  # R0
        s[15:18] = self.rng.uniform(0.6, 1.0, size=3) * difficulty  # crew statuses
        s[18] = self.rng.integers(0, 3) / 2.0  # mes_location normalized
        s[19] = self.rng.uniform(0.5, 0.9) * difficulty  # mes_soc
        s[20] = self.rng.uniform(0.5, 0.95) * difficulty  # material_stock
        s[21] = self.rng.uniform(0.5, 0.9) * difficulty  # switching capability

        preset_name = str(options.get("preset_name", "")).strip()
        preset_jitter = float(options.get("preset_jitter", 0.0))
        if preset_name:
            s = self._apply_preset(s, preset_name, jitter=preset_jitter)
        s[22], _ = self._stage(s)
        s[23] = 0.0

        self.state = self._clip01(s)
        info = self._build_info(progress_delta=0.0, invalid_action=False, invalid_reason="", mes_used=False)
        info["preset_name"] = preset_name
        return self.state.copy(), info

    def step(self, action: int):
        action = int(action)
        s_prev = self.state.copy()
        s = self.state.copy()

        P = s[0:3]
        C = s[3:6]
        R = s[6:9]
        L = s[9:12]
        P0, C0, R0 = float(s[12]), float(s[13]), float(s[14])
        crew_p, crew_c, crew_r = float(s[15]), float(s[16]), float(s[17])
        mes_loc = int(round(float(s[18]) * 2))
        mes_soc = float(s[19])
        material = float(s[20])
        switch_cap = float(s[21])
        stage_val, stage_name = self._stage(s)
        action_category = self._action_category(action)
        repeated_category = action_category == self.prev_action_category

        invalid_action = False
        invalid_reason = ""
        mes_used = False

        def zone_eff(z: int) -> float:
            # road improves same-zone repair efficiency + trunk road backbone support.
            return float(0.45 + 0.40 * R[z] + 0.15 * R0)

        if 0 <= action <= 2:  # road A/B/C
            z = action
            repeat_factor = 0.93 if repeated_category else 1.0
            gain = 0.06 * crew_r * repeat_factor * (1.0 if stage_name != "late" else 0.75)
            R[z] += gain + self.rng.normal(0.0, 0.004)
            R0 += 0.02 * gain
            material -= 0.03

        elif 3 <= action <= 5:  # power A/B/C
            z = action - 3
            eff = zone_eff(z)
            repeat_factor = 0.93 if repeated_category else 1.0
            gain = 0.055 * crew_p * eff * repeat_factor
            if stage_name == "early":
                gain *= 0.95
            P[z] += gain + self.rng.normal(0.0, 0.004)
            L[z] += 0.045 * gain
            P0 += 0.012 * gain
            material -= 0.035

        elif 6 <= action <= 8:  # comm A/B/C
            z = action - 6
            eff = zone_eff(z)
            # comm depends on local power or MES support in-zone.
            power_support = 1.0 if (P[z] >= 0.35 or (mes_loc == z and mes_soc > 0.15)) else 0.6
            repeat_factor = 0.93 if repeated_category else 1.0
            gain = 0.055 * crew_c * eff * power_support * repeat_factor
            C[z] += gain + self.rng.normal(0.0, 0.004)
            C0 += 0.01 * gain
            material -= 0.03

        elif 9 <= action <= 11:  # MES dispatch A/B/C
            z = action - 9
            mes_used = True
            if R[z] < 0.25 or mes_soc < 0.08:
                invalid_action = True
                invalid_reason = "mes_unreachable_or_low_soc"
                material -= 0.01
            else:
                mes_loc = z
                mes_soc -= 0.06
                # MES supports local power + critical load
                P[z] += 0.04
                L[z] += 0.045

        elif action == 12:  # feeder reconfigure / load transfer
            # if C0 low, reconfiguration coordination degrades.
            coord_eff = 0.5 + 0.5 * C0
            if C0 < 0.30:
                invalid_action = True
                invalid_reason = "low_backbone_comm_for_feeder"
            repeat_factor = 0.90 if repeated_category else 1.0
            transfer = 0.04 * switch_cap * coord_eff * repeat_factor
            weakest = int(np.argmin(L))
            donor = int(np.argmax(P))
            L[weakest] += transfer
            P[donor] -= 0.01 * transfer
            material -= 0.025

        elif action == 13:  # coordinated balanced restoration
            # high C0 and recovered communication improve balanced coordination.
            coord = (0.55 + 0.30 * C0 + 0.15 * float(np.mean(C)))
            if stage_name == "middle":
                coord *= 1.08
            if stage_name == "late":
                coord *= 1.05
            if repeated_category:
                coord *= 0.95
            P += 0.018 * coord * np.array([zone_eff(0), zone_eff(1), zone_eff(2)])
            C += 0.018 * coord * np.array([zone_eff(0), zone_eff(1), zone_eff(2)])
            R += 0.014 * coord
            L += 0.016 * coord
            P0 += 0.01 * coord
            C0 += 0.01 * coord
            R0 += 0.008 * coord
            material -= 0.04
        else:  # 14 wait_hold (resource preservation / regroup)
            # Hold action: preserve resources and recover teams for late-stage finishing.
            crew_p += 0.020
            crew_c += 0.020
            crew_r += 0.020
            switch_cap += 0.015
            material += 0.020
            mes_soc += 0.015

        # lightweight resource dynamics
        if 3 <= action <= 5:
            crew_p -= 0.016 + (0.010 if repeated_category else 0.0)
        else:
            crew_p += 0.006
        if 6 <= action <= 8:
            crew_c -= 0.016 + (0.010 if repeated_category else 0.0)
        else:
            crew_c += 0.006
        if 0 <= action <= 2:
            crew_r -= 0.016 + (0.010 if repeated_category else 0.0)
        else:
            crew_r += 0.006

        if action == 12:
            switch_cap -= 0.030 + (0.010 if repeated_category else 0.0)
        else:
            switch_cap += 0.006

        if action == 13 and repeated_category:
            # mild coordinated overuse fatigue across multiple resources
            crew_p -= 0.006
            crew_c -= 0.006
            crew_r -= 0.006
            switch_cap -= 0.008

        crew_p = float(np.clip(crew_p, 0.20, 1.0))
        crew_c = float(np.clip(crew_c, 0.20, 1.0))
        crew_r = float(np.clip(crew_r, 0.20, 1.0))
        switch_cap = float(np.clip(switch_cap, 0.20, 1.0))

        # resource replenishment
        material = float(np.clip(material + 0.008, 0.0, 1.0))
        mes_soc = float(np.clip(mes_soc + 0.006 if not mes_used else mes_soc, 0.0, 1.0))
        resource_shortage = material < 0.10
        if resource_shortage:
            invalid_action = True
            invalid_reason = "material_shortage"

        # late stage stabilization bonus/penalty logic (simple)
        if stage_name == "late" and action in (0, 1, 2):
            invalid_action = True
            invalid_reason = "late_stage_road_action_penalty"

        # write back + clamp
        s[0:3], s[3:6], s[6:9], s[9:12] = P, C, R, L
        s[12], s[13], s[14] = P0, C0, R0
        s[15], s[16], s[17] = crew_p, crew_c, crew_r
        s[18], s[19], s[20] = mes_loc / 2.0, mes_soc, material
        s[21] = switch_cap

        s = self._clip01(s)
        s[22], _ = self._stage(s)
        s[23] = 1.0 if invalid_action else 0.0
        if invalid_action:
            self.constraint_violation_count += 1

        # reward terms
        dp = float(np.mean(s[0:3] - s_prev[0:3]))
        dc = float(np.mean(s[3:6] - s_prev[3:6]))
        dr = float(np.mean(s[6:9] - s_prev[6:9]))
        dl = float(np.mean(s[9:12] - s_prev[9:12]))

        # synergy: zones with simultaneously high P,C,R
        synergy = float(np.mean(np.minimum(np.minimum(s[0:3], s[3:6]), s[6:9])))
        constraint_pen = 1.0 if invalid_action else 0.0
        switch_pen = 1.0 if action != self.prev_action else 0.0
        mes_overuse_pen = 1.0 if (mes_used and mes_soc < 0.12) else 0.0

        reward = (
            self.w["delta_power"] * dp
            + self.w["delta_comm"] * dc
            + self.w["delta_road"] * dr
            + self.w["delta_critical_load"] * dl
            + self.w["synergy_bonus"] * synergy
            - self.w["constraint_penalty"] * constraint_pen
            - self.w["action_switch_penalty"] * switch_pen
            - self.w["mes_overuse_penalty"] * mes_overuse_pen
        )

        self.state = s
        self.prev_action = action
        self.prev_action_category = action_category
        self.step_count += 1

        progress_prev = self._progress(s_prev)
        progress_now = self._progress(s)
        progress_delta = progress_now - progress_prev

        terminated = bool(np.mean(s[9:12]) >= 0.82 and progress_now >= 0.78)
        truncated = self.step_count >= self.max_steps

        info = self._build_info(progress_delta=progress_delta, invalid_action=invalid_action, invalid_reason=invalid_reason, mes_used=mes_used)
        return self.state.copy(), float(reward), terminated, truncated, info

    def _build_info(self, progress_delta: float, invalid_action: bool, invalid_reason: str, mes_used: bool) -> dict[str, Any]:
        s = self.state
        stage_val, stage_name = self._stage(s)
        weakest_zone = int(np.argmin((s[0:3] + s[3:6] + s[6:9]) / 3.0))
        zone_map = {0: "A", 1: "B", 2: "C"}

        return {
            "progress_delta": float(progress_delta),
            "invalid_action": bool(invalid_action),
            "invalid_reason": str(invalid_reason),
            "mes_used": bool(mes_used),
            "stage": stage_name,
            "stage_indicator": float(stage_val),
            "constraint_violation": bool(s[23] > 0.5),
            "constraint_violation_count": int(self.constraint_violation_count),
            "communication_recovery_ratio": float(np.mean(s[3:6])),
            "power_recovery_ratio": float(np.mean(s[0:3])),
            "road_recovery_ratio": float(np.mean(s[6:9])),
            "critical_load_recovery_ratio": float(np.mean(s[9:12])),
            "backbone_power_ratio": float(s[12]),
            "backbone_comm_ratio": float(s[13]),
            "backbone_road_ratio": float(s[14]),
            "crew_power_status": float(s[15]),
            "crew_comm_status": float(s[16]),
            "crew_road_status": float(s[17]),
            "mes_soc": float(s[19]),
            "material_stock": float(s[20]),
            "switching_capability": float(s[21]),
            "zone_A_power_ratio": float(s[0]),
            "zone_B_power_ratio": float(s[1]),
            "zone_C_power_ratio": float(s[2]),
            "zone_A_comm_ratio": float(s[3]),
            "zone_B_comm_ratio": float(s[4]),
            "zone_C_comm_ratio": float(s[5]),
            "zone_A_road_ratio": float(s[6]),
            "zone_B_road_ratio": float(s[7]),
            "zone_C_road_ratio": float(s[8]),
            "zone_A_critical_load_ratio": float(s[9]),
            "zone_B_critical_load_ratio": float(s[10]),
            "zone_C_critical_load_ratio": float(s[11]),
            "weakest_zone": zone_map[weakest_zone],
            "weakest_layer": str(np.argmin([np.mean(s[0:3]), np.mean(s[3:6]), np.mean(s[6:9])])),
            "critical_load_shortfall": float(1.0 - np.mean(s[9:12])),
        }


# Backward name alias to avoid breaking existing scripts that still import MockRecoveryEnv.
MockRecoveryEnv = ProjectRecoveryEnv
