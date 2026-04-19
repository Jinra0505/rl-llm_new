from __future__ import annotations

SYSTEM_PROMPT = """You generate safe Python shaping functions for a tri-layer recovery RL task.
Return STRICT JSON only.
Generated code must define:
- revise_state(state, info=None)
- intrinsic_reward(state, action, next_state, info=None, revised_state=None)
Allowed imports: numpy, math, __future__.
Forbidden: filesystem writes, networking, subprocess, eval/exec.
Environment state has 24 dims:
0:3 power A/B/C, 3:6 comm A/B/C, 6:9 road A/B/C, 9:12 critical load A/B/C,
12:15 backbone power/comm/road, 15:18 crew power/comm/road, 18 mes location (/2),
19 mes_soc, 20 material_stock, 21 switching_capability, 22 stage_indicator, 23 constraint_flag.
Action space has 15 actions:
0-2 road A/B/C, 3-5 power A/B/C, 6-8 comm A/B/C, 9-11 mes_to A/B/C, 12 feeder_reconfigure, 13 coordinated_balanced, 14 wait_hold.
"""

STRUCTURED_SPEC_PROMPT = """Task mode: {task_mode}, stage: {stage}
Environment: zonal three-layer coupled recovery (communication, power, transportation) with zones A/B/C.
Observation schema summary:
{observation_schema}
Planning blueprint JSON:
{planning_json}
Generate one compact structured shaping spec (NOT Python code).
Prioritize system-level recovery over single-layer gains:
- critical load recovery and completion progress
- balanced tri-layer recovery across zones
- lower invalid actions / constraint violations
- preserve material buffer for late-stage completion
- avoid invalid or precondition-violating actions
- do not overuse feeder/coordinated actions when prerequisites are weak (e.g., low mes_soc, low backbone_comm, low material)
- prioritize low-violation completion in late-stage finishing
- include explicit signals to reach late stage and complete restoration
- intrinsic shaping should be small, dense, progress-oriented, and smooth
Return strict JSON only with keys:
- file_name
- rationale
- expected_behavior
- spec
Where spec must include only bounded scalar/int controls:
- style (conservative_safety_first|balanced|aggressive_recovery_first)
- append_crit_progress (0/1)
- append_backbone_balance (0/1)
- append_resource_buffer (0/1)
- append_stage_indicator (0/1)
- w_delta_comm, w_delta_power, w_delta_road, w_delta_critical
- w_stage_progress, w_finish_bonus
- w_resource_penalty, w_wait_hold_penalty, w_violation_penalty
- recovery_floor_emphasis, safety_emphasis, late_stage_emphasis, wait_hold_discouragement
- critical_gain_scale, progress_bonus_scale, weak_layer_gain_scale, weak_zone_gain_scale
- late_stage_bonus_scale, completion_bonus_scale, wait_penalty_scale
- invalid_penalty_scale, constraint_penalty_scale, material_penalty_scale, recovery_floor_bonus_scale
Do not output Python code.
"""

ROUTER_PROMPT = """Select one task mode from:
- critical_load_priority
- restoration_capability_priority
- global_efficiency_priority
Use stage, weakest layer, weakest zone, critical-load shortfall, backbone_comm_ratio, and violation rates.
Do NOT over-select critical_load_priority when shortfall is only moderate.
Prefer critical_load_priority only when shortfall is clearly dominant and power/critical bottlenecks block completion.
Prefer restoration_capability_priority when road/communication/resources limit feasible actions.
Prefer global_efficiency_priority when a balanced multi-layer recovery is needed.
Return JSON with: task_mode, confidence, reason, stage.
Round-2 switching rule:
- If previous round shows low success, low progress, or unfinished middle-stage behavior, switch task_mode from previous round unless there is strong evidence the same task is still the dominant bottleneck.
"""

PLANNING_PROMPT = """Using the routing context + task_mode + stage, produce a concise shaping + phase planning JSON.
Required keys:
- weakest_layer
- weakest_zone
- late_stage_risk
- violation_risk
- should_reward (array)
- should_penalize (array)
- should_avoid (array)
- finishing_strategy
- codegen_guidance
- phase_mode
- phase_duration
- resource_floor_target
- completion_push_allowed
- late_stage_trigger
Output format requirements (strict):
- Return one compact JSON object only (no markdown, no prose outside JSON, no code fences)
- Use exactly the required keys above
- Expected types:
  - weakest_layer: string
  - weakest_zone: string
  - late_stage_risk: short string
  - violation_risk: short string
  - should_reward: short array (0-6 items)
  - should_penalize: short array (0-6 items)
  - should_avoid: short array (0-6 items, short strings)
  - finishing_strategy: short string
  - codegen_guidance: short string
  - phase_mode: one of critical_push|capability_unblock|balanced_progress|late_finish|resource_preserve
  - phase_duration: integer 2-80
  - resource_floor_target: float 0.05-0.40
  - completion_push_allowed: boolean
  - late_stage_trigger: float 0.50-0.95
- Keep all strings concise and operational; avoid long narratives
Planning constraints:
- avoid invalid or precondition-violating actions
- avoid overusing feeder/coordinated actions when prerequisites are weak
- prioritize low-violation completion and targeted finishing actions on weakest layer/zone
- include explicit anti-stagnation strategy for long middle-stage trajectories
- include resource-preservation strategy to keep enough material for late-stage completion
- include explicit anti-wait-overuse rule: wait_hold should be safety-only, not dominant when feasible recovery actions exist
- include explicit round-specific task-switch rationale tied to prior-round failure patterns
- intrinsic shaping should be small, dense, and progress-oriented
- do not duplicate invalid-action / constraint / wait penalties already handled elsewhere
- do not reward conservative inaction
Return JSON only.
"""

COMPACT_PLANNING_PROMPT = """Return JSON only (no markdown/code fences/prose).
Compact planning mode: keep output minimal and short.
Required keys (exactly):
- weakest_layer (string)
- weakest_zone (string)
- should_reward (array, up to 4 short items)
- should_avoid (array, up to 4 short items)
- codegen_guidance (short string)
- phase_mode (string enum)
- phase_duration (int 2-80)
- resource_floor_target (float 0.05-0.40)
- completion_push_allowed (bool)
- late_stage_trigger (float 0.50-0.95)
Rules:
- Keep every string very short and operational.
- No long rationale text.
- Focus only on highest-impact shaping hints for next codegen.
"""

FEEDBACK_PROMPT = """Return JSON only. No markdown. No code fences. No extra text before or after JSON.
Output schema (fixed keys only):
{
  "improvement_focus": ["short phrase", "..."],
  "keep_signals": ["short phrase", "..."],
  "avoid_patterns": ["short phrase", "..."],
  "finish_strategy_adjustments": ["short phrase", "..."],
  "phase_guidance": "keep|switch|extend",
  "next_phase_mode": "critical_push|capability_unblock|balanced_progress|late_finish|resource_preserve",
  "next_phase_duration": 8,
  "confidence": 0.0
}
Rules:
- Keep each list concise (0-4 items, short phrases only).
- Confidence must be a float in [0, 1].
- Keep output brief and operational, avoid long explanations.
- Use Lipschitz smoothness summary when provided: keep informative low-sensitivity signals and reduce unstable high-sensitivity state-reward mappings.
- If smoothness is poor, prioritize stabilizing reward shaping over adding more aggressive bonuses.
- Return one compact JSON object only (no markdown / no code fences / no extra commentary).
"""
