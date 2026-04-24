from __future__ import annotations

from typing import Any

ACTION_ID_TO_NAME = {
    0: "road_A",
    1: "road_B",
    2: "road_C",
    3: "power_A",
    4: "power_B",
    5: "power_C",
    6: "comm_A",
    7: "comm_B",
    8: "comm_C",
    9: "mes_to_A",
    10: "mes_to_B",
    11: "mes_to_C",
    12: "feeder_reconfigure",
    13: "coordinated_balanced",
    14: "wait_hold",
}

ACTION_NAME_TO_CATEGORY = {
    "road_A": "road",
    "road_B": "road",
    "road_C": "road",
    "power_A": "power",
    "power_B": "power",
    "power_C": "power",
    "comm_A": "communication",
    "comm_B": "communication",
    "comm_C": "communication",
    "mes_to_A": "mobile_energy",
    "mes_to_B": "mobile_energy",
    "mes_to_C": "mobile_energy",
    "feeder_reconfigure": "feeder",
    "coordinated_balanced": "coordinated",
    "wait_hold": "wait",
}

ACTION_NAME_TO_LABEL = {
    "road_A": "Road repair A",
    "road_B": "Road repair B",
    "road_C": "Road repair C",
    "power_A": "Power repair A",
    "power_B": "Power repair B",
    "power_C": "Power repair C",
    "comm_A": "Communication repair A",
    "comm_B": "Communication repair B",
    "comm_C": "Communication repair C",
    "mes_to_A": "Mobile energy to A",
    "mes_to_B": "Mobile energy to B",
    "mes_to_C": "Mobile energy to C",
    "feeder_reconfigure": "Feeder reconfiguration",
    "coordinated_balanced": "Coordinated restoration",
    "wait_hold": "Wait / hold",
}


def normalize_action_name(action: Any) -> str | None:
    if action is None:
        return None
    if isinstance(action, str):
        action = action.strip()
        if action == "":
            return None
        if action.isdigit():
            return ACTION_ID_TO_NAME.get(int(action))
        return action if action in ACTION_NAME_TO_CATEGORY else None
    try:
        aid = int(action)
    except Exception:
        return None
    return ACTION_ID_TO_NAME.get(aid)


def action_fields(action: Any) -> dict[str, Any]:
    name = normalize_action_name(action)
    if name is None:
        return {"action": action, "action_name": None, "action_label": None, "action_category": None}
    return {
        "action": action,
        "action_name": name,
        "action_label": ACTION_NAME_TO_LABEL.get(name),
        "action_category": ACTION_NAME_TO_CATEGORY.get(name),
    }
