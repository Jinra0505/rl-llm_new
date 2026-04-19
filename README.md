# Three-Task Recovery Recognition Framework

This repository is a clean project centered on **three-task situation recognition** for a coupled recovery scenario.

## Main purpose (primary)
Recognize each scenario into exactly one task:
- `critical_load_priority`
- `restoration_capability_priority`
- `global_efficiency_priority`

## Optional downstream (secondary)
After recognition, you may optionally run:
1. LLM function generation (`revise_state` + `intrinsic_reward`)
2. DQN training/evaluation in the recovery environment

## Important design rule
**Task override logic is removed.**
The recognized task comes directly from task recognition output (rule mode or LLM classification output) and is not post-corrected by forced engineering overrides.

## Installation
```bash
python -m pip install -r requirements.txt
```

## Project structure
- `task_recognition_prompt.py`: builds recognition prompt from scenario summary.
- `task_recognizer.py`: three-task recognizer (rule + LLM).
- `run_task_recognition.py`: standalone recognition-only entrypoint.
- `mock_recovery_env.py`: coupled recovery environment.
- `run_outer_loop.py`: optional downstream full pipeline.
- `train_rl.py`: optional downstream DQN training module.
- `llm_client.py`: strict real-LLM client.
- `prompts.py`, `router.py`, `baseline_noop.py`, `config.yaml`.

## Run: task recognition only
### Rule-based recognition
```bash
python run_task_recognition.py --mode rule
```

### LLM-based recognition
```bash
python run_task_recognition.py --mode llm --llm-mode real
```

(Optionally pass `--input-json path/to/context.json` where JSON includes `env_summary` and `trajectory_summary`.)

## Run: optional downstream pipeline
```bash
python run_outer_loop.py --env project_recovery --llm-mode real --router-mode llm --reroute-each-round --config config.yaml
```

## Run: iterative diagnosis protocol (6 conditions × 3 seeds)
Use this helper to execute a full diagnosis pass over:
- `baseline_rl` / `single_shot_llm` / `full_outer_loop`
- each under `clean` and `engineered` rewards
- default seeds `42 43 44`

It writes per-iteration artifacts under `outputs/diagnosis_protocol/` and a consolidated `protocol_report.json`.

```bash
python experiments/run_diagnosis_protocol.py \
  --config config_outer_loop_real_small.yaml \
  --split-name benchmark_resource_constrained_presets \
  --seeds 42 43 44 \
  --max-iterations 2 \
  --resume
```

## Notes
- Repository is intentionally minimal and recognition-first.
- Old run artifacts are removed from versioned files to keep the repo clean.
