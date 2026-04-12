本补实验采用 offline LLM decision proxy for downstream validation，不调用真实 API。  
现有 recognizer_metrics_latest.json 已表明：LLM 在 uncertain、definition_shift 场景优于 rule。  
本次进一步将 rule/llm_proxy 识别结果映射到固定 task-oriented module，再进入同配置 RL。  
上一版问题根因为：case 信息主要停留在 routing_context，RL 环境 reset 未稳定接入 case preset，导致同 task 下跨 case 指标过度同质。  
本次修复：train_rl.run_training 新增 env_reset_options，并在 train/eval 每次 reset 统一注入 preset；同时输出 case_env_snapshot / training_env_check / eval_env_check 证明生效。  
在 uncertain_like 与 definition_shift_like 中，rule 与 llm_proxy 任务选择出现差异。  
这些差异会传递到下游：selection_score 与 success_rate 在多例中发生可观变化。  
在 clear control case 中，两者任务一致或差异较小，性能接近。  
结果说明 LLM 贡献不止于“分类更准”，也体现在“为 RL 提供更有效任务导向”。  
该证据 supports/validates LLM+RL 协同恢复主线，但不等价于证明真实在线 LLM 永远更优。  
本结果可作为 EI 论文补充实验支撑材料。
