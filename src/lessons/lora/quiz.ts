// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'lora.zero-init',
    kind: 'choice',
    concepts: ['lora'],
    prompt: 'In LoRA (ΔW = (α/r) · B · A), why is B initialised to all zeros while A is initialised with random Gaussian numbers?',
    options: [
      {
        text: 'So that B·A = 0 at step 0 (leaving the pretrained model completely untouched), while A ≠ 0 gives B an immediately non-zero gradient on step 1 to break symmetry.',
        correct: true,
        why: 'If both A and B were zero, both gradients would be identically zero forever. If both were random, B·A ≠ 0 would corrupt the pretrained model before learning even starts.',
      },
      {
        text: 'Because A is frozen and only B is trained.',
        why: 'Both A and B are trained! On step 1, B receives a non-zero gradient and moves away from zero; from step 2 onward, A also receives non-zero gradients.',
      },
      {
        text: 'Because setting B = 0 cuts the number of trainable parameters in half.',
        why: 'Every entry of B is still a trainable parameter; zeroing B only controls the starting point at step 0.',
      },
    ],
  },
  {
    id: 'lora.param-count',
    kind: 'numeric',
    concepts: ['lora'],
    tolerance: 1,
    make: (r) => {
      const d = [1024, 2048, 4096, 8192][r.int(4)];
      const rank = [4, 8, 16, 32][r.int(4)];
      const ans = rank * (d + d);
      return {
        prompt: `You attach a rank-${rank} LoRA adapter (B ∈ ℝ^{${d}×${rank}}, A ∈ ℝ^{${rank}×${d}}) to a single ${d} × ${d} square weight matrix. How many trainable parameters does this adapter have?`,
        answer: ans,
        explain: `A has ${rank} × ${d} = ${(rank * d).toLocaleString('en-GB')} weights and B has ${d} × ${rank} = ${(rank * d).toLocaleString('en-GB')} weights, for a total of ${rank} × (${d} + ${d}) = ${ans.toLocaleString('en-GB')} parameters (vs ${(d * d).toLocaleString('en-GB')} in the full matrix).`,
      };
    },
  },
  {
    id: 'lora.memory-why',
    kind: 'choice',
    concepts: ['lora', 'qlora'],
    prompt: 'Why does LoRA shrink fine-tuning VRAM by ~8× on Llama-3 8B (from 120 GiB down to 15.6 GiB) even though the frozen base weights still sit in BF16?',
    options: [
      {
        text: 'In full mixed-precision AdamW, master weights + gradients + Adam’s m and v states cost 14 extra bytes per trainable parameter (105 GiB on 8B!). Freezing 99.5% of weights eliminates ~99.5% of those optimizer and gradient states.',
        correct: true,
        why: 'Adam states alone take 59.8 GiB in full FT on Llama-3 8B vs just 0.31 GiB with r=16 LoRA — a ~193× drop! QLoRA then quantises the remaining frozen base weights to 4-bit, dropping total VRAM to 4.8 GiB.',
      },
      {
        text: 'Because LoRA discards the base model weights during training.',
        why: 'The frozen base weights must stay in VRAM for every forward pass (and backward input-gradient pass); LoRA eliminates their optimizer states and weight gradients.',
      },
      {
        text: 'Because LoRA skips backpropagation altogether.',
        why: 'LoRA runs full backpropagation through the frozen base weights so gradients reach A and B — it simply skips storing weight gradients and Adam moments for W_0.',
      },
    ],
  },
  {
    id: 'lora.merge-inference',
    kind: 'choice',
    concepts: ['lora'],
    prompt: 'Once a LoRA adapter (A, B) has finished training, how much extra inference latency does it add when serving a single fine-tuned model?',
    options: [
      {
        text: 'Zero — you add (α/r)·B·A directly into W_0 offline (W_merged = W_0 + (α/r)BA), producing a standard weight matrix with identical inference cost.',
        correct: true,
        why: 'Because matrix addition is linear, merging the adapter offline costs zero extra FLOPs and zero extra memory at inference time. You only keep A and B separate if you dynamically hot-swap many customer adapters on one shared base model.',
      },
      {
        text: 'It doubles inference latency because every token runs through both W_0 and BA.',
        why: 'You only pay a small separate BA pass if you choose NOT to merge (for multi-tenant adapter serving). For a single deployment, merging is exact and free.',
      },
      {
        text: 'It cuts inference latency in half because W_0 becomes low-rank.',
        why: 'W_0 itself remains full-rank; only the learned delta ΔW = (α/r)BA is low-rank.',
      },
    ],
  },
];
