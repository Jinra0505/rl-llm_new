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

CODEGEN_PROMPT = """Task mode: {task_mode}, stage: {stage}
Environment: zonal three-layer coupled recovery (communication, power, transportation) with zones A/B/C.
Observation schema summary:
{observation_schema}
Planning blueprint JSON:
{planning_json}
Generate one module that improves policy state representation and intrinsic shaping for this 24D state.
Prioritize system-level recovery over single-layer gains:
- critical load recovery and completion progress
- balanced tri-layer recovery across zones
- lower invalid actions / constraint violations
- preserve material buffer for late-stage completion
- avoid invalid or precondition-violating actions
- do not overuse feeder/coordinated actions when prerequisites are weak (e.g., low mes_soc, low backbone_comm, low material)
- prioritize low-violation completion in late-stage finishing
- include explicit signals to reach late stage and complete restoration
- intrinsic reward should be small, dense, and progress-oriented
- do not duplicate invalid-action / constraint / wait penalties already handled elsewhere
- do not reward conservative inaction
Only output code with revise_state and intrinsic_reward (no extra dependencies/modules).
Keep code short and robust (target <= 45 lines).
Return JSON keys: file_name, rationale, code, expected_behavior.
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

PLANNING_PROMPT = """Using the routing context + task_mode + stage, produce a concise shaping planning JSON.
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

FEEDBACK_PROMPT = """Return JSON only. No markdown. No code fences. No extra text before or after JSON.
Output schema (fixed keys only):
{
  "improvement_focus": ["short phrase", "..."],
  "keep_signals": ["short phrase", "..."],
  "avoid_patterns": ["short phrase", "..."],
  "finish_strategy_adjustments": ["short phrase", "..."],
  "confidence": 0.0
}
Rules:
- Keep each list concise (0-4 items, short phrases only).
- Confidence must be a float in [0, 1].
- Keep output brief and operational, avoid long explanations.
"""
