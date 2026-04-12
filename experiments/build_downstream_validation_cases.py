from __future__ import annotations

from typing import Any


def build_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_name": "uncertain_boundary_u1",
            "case_type": "uncertain_like",
            "preset_name": "uncertain_boundary_case_u1",
            "oracle_task": "restoration_capability_priority",
            "routing_context": {
                "env_summary": {
                    "communication_recovery_ratio": 0.54,
                    "power_recovery_ratio": 0.53,
                    "road_recovery_ratio": 0.52,
                    "critical_load_shortfall": 0.40,
                    "material_stock": 0.18,
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0028,
                    "constraint_violation_rate": 0.14,
                    "action_category_distribution": {"wait": 0.24},
                },
                "semantic_cue": "backbone and mobility bottleneck dominates despite visible critical gap",
            },
        },
        {
            "case_name": "uncertain_boundary_u2",
            "case_type": "uncertain_like",
            "preset_name": "uncertain_boundary_case_u2",
            "oracle_task": "critical_load_priority",
            "routing_context": {
                "env_summary": {
                    "communication_recovery_ratio": 0.55,
                    "power_recovery_ratio": 0.54,
                    "road_recovery_ratio": 0.53,
                    "critical_load_shortfall": 0.43,
                    "material_stock": 0.24,
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0032,
                    "constraint_violation_rate": 0.11,
                    "action_category_distribution": {"wait": 0.22},
                },
                "semantic_cue": "critical service obligations remain primary",
            },
        },
        {
            "case_name": "definition_shift_d1",
            "case_type": "definition_shift_like",
            "preset_name": "definition_shift_case_d1",
            "oracle_task": "global_efficiency_priority",
            "routing_context": {
                "env_summary": {
                    "communication_recovery_ratio": 0.66,
                    "power_recovery_ratio": 0.67,
                    "road_recovery_ratio": 0.65,
                    "critical_load_shortfall": 0.24,
                    "material_stock": 0.28,
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0020,
                    "constraint_violation_rate": 0.07,
                    "action_category_distribution": {"wait": 0.40},
                },
                "definition_profile": "shifted_finish_coordination",
                "semantic_cue": "cross-layer finishing coordination and closeout quality are explicit goals",
            },
        },
        {
            "case_name": "definition_shift_d2",
            "case_type": "definition_shift_like",
            "preset_name": "definition_shift_case_d2",
            "oracle_task": "restoration_capability_priority",
            "routing_context": {
                "env_summary": {
                    "communication_recovery_ratio": 0.58,
                    "power_recovery_ratio": 0.60,
                    "road_recovery_ratio": 0.54,
                    "critical_load_shortfall": 0.26,
                    "material_stock": 0.13,
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0022,
                    "constraint_violation_rate": 0.16,
                    "action_category_distribution": {"wait": 0.30},
                },
                "definition_profile": "shifted_finish_coordination",
                "semantic_cue": "backbone mobility/material bottleneck must be removed before finish",
            },
        },
        {
            "case_name": "clear_control_c1",
            "case_type": "clear_control",
            "preset_name": "critical_load_dominant",
            "oracle_task": "critical_load_priority",
            "routing_context": {
                "env_summary": {
                    "communication_recovery_ratio": 0.58,
                    "power_recovery_ratio": 0.46,
                    "road_recovery_ratio": 0.57,
                    "critical_load_shortfall": 0.55,
                    "material_stock": 0.30,
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0040,
                    "constraint_violation_rate": 0.08,
                    "action_category_distribution": {"wait": 0.18},
                },
                "semantic_cue": "critical backlog dominates",
            },
        },
        {
            "case_name": "clear_control_c2",
            "case_type": "clear_control",
            "preset_name": "global_finishing_dominant",
            "oracle_task": "global_efficiency_priority",
            "routing_context": {
                "env_summary": {
                    "communication_recovery_ratio": 0.70,
                    "power_recovery_ratio": 0.71,
                    "road_recovery_ratio": 0.69,
                    "critical_load_shortfall": 0.20,
                    "material_stock": 0.30,
                },
                "trajectory_summary": {
                    "mean_progress_delta": 0.0019,
                    "constraint_violation_rate": 0.06,
                    "action_category_distribution": {"wait": 0.43},
                },
                "semantic_cue": "finishing coordination is primary",
            },
        },
    ]
