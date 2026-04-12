from pathlib import Path

TASK_TO_MODULE = {
    "critical_load_priority": Path("generated/static_task_modules/critical_load_priority_module.py"),
    "restoration_capability_priority": Path("generated/static_task_modules/restoration_capability_priority_module.py"),
    "global_efficiency_priority": Path("generated/static_task_modules/global_efficiency_priority_module.py"),
}


def get_module_path(task_mode: str) -> Path:
    if task_mode not in TASK_TO_MODULE:
        raise ValueError(f"Unknown task_mode: {task_mode}")
    return TASK_TO_MODULE[task_mode]
