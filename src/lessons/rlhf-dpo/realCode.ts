// authored-by: Claude Opus 5.5
import type { AlignmentLabConfig } from './dpo';

/**
 * The Hugging Face TRL config for an alignment-lab setting. In TRL, `beta` is
 * the KL coefficient for both DPO and the reward-model RL trainers (RLOO is
 * used here as the RLHF example); larger β keeps the policy closer to the reference.
 */
export function trlConfigLine(cfg: AlignmentLabConfig, allowDpo: boolean): string {
  const method = allowDpo ? cfg.method : 'rlhf-proxy';
  if (method === 'sft') return `SFTConfig(max_steps=${cfg.steps})  # imitate the chosen replies only`;
  if (method === 'dpo') return `DPOConfig(beta=${cfg.beta}, max_steps=${cfg.steps})`;
  return `RLOOConfig(beta=${cfg.beta})  # KL leash against the reference; reward from the proxy reward model`;
}
