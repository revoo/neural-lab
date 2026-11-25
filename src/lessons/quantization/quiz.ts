// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'quant.bf16-vs-fp16',
    kind: 'choice',
    concepts: ['bf16'],
    prompt: 'Why did BF16 replace FP16 as the default 16-bit format for training large neural networks?',
    options: [
      {
        text: 'BF16 keeps FP32’s full 8-bit exponent range (up to 3.4 × 10³⁸), preventing FP16’s frequent 65,504 overflows and gradient underflows at the small price of 3 fewer mantissa bits.',
        correct: true,
        why: 'Neural nets care far more about dynamic range (not turning gradients into Inf or 0) than about 3 vs 4 decimal digits of precision.',
      },
      {
        text: 'BF16 uses half as much memory as FP16.',
        why: 'Both FP16 and BF16 are 16 bits (2 bytes) per number — they only divide those 16 bits differently between exponent and mantissa.',
      },
      {
        text: 'BF16 has more decimal precision than FP16.',
        why: 'The opposite: FP16 has 10 mantissa bits (~3.3 digits), while BF16 has 7 mantissa bits (~2.4 digits). BF16 wins on range, not precision.',
      },
    ],
  },
  {
    id: 'quant.outlier',
    kind: 'choice',
    concepts: ['quantization'],
    prompt: 'In symmetric per-tensor INT4 quantisation, why does a single weight of 6.5 ruin 63 other weights sitting between −0.9 and +0.9?',
    options: [
      {
        text: 'The shared scale factor s = max(|w|) / 7 stretches to 0.93, so almost every normal weight falls within ±0.46 of zero and gets rounded to exact 0.',
        correct: true,
        why: 'In our widget, 52 of the 63 normal weights (83%) collapsed to exact zero! Group-wise quantisation isolates the outlier to its own tiny group so the remaining groups keep fine scales.',
      },
      {
        text: 'Because INT4 cannot represent negative numbers.',
        why: 'Signed INT4 covers −7..+7 (or −8..+7 in two’s complement). The problem is that only 15 discrete steps have to span the entire range up to the outlier.',
      },
      {
        text: 'Because 6.5 causes integer overflow in 32-bit accumulators.',
        why: '6.5 simply sets the quantisation scale; the damage is loss of resolution on all the smaller weights sharing that scale.',
      },
    ],
  },
  {
    id: 'quant.group-overhead',
    kind: 'numeric',
    concepts: ['quantization'],
    tolerance: 0.01,
    make: (r) => {
      const g = [16, 32, 64, 128][r.int(4)];
      const eff = 4 + 16 / g;
      return {
        prompt: `You quantise weights to INT4 (4 bits each) with one FP16 (16-bit) scale factor per group of ${g} weights. What is the effective bits per weight?`,
        answer: eff,
        explain: `4 bits for each weight plus 16 / ${g} = ${(16 / g).toFixed(3)} bits of scale overhead = ${eff.toFixed(3)} effective bits per weight.`,
      };
    },
  },
  {
    id: 'quant.mixed-precision',
    kind: 'choice',
    concepts: ['quantization'],
    prompt: 'Why do practical formats like GGUF Q4_K_M use mixed precision instead of putting every single tensor in plain 4-bit?',
    options: [
      {
        text: 'A handful of sensitive tensors (outlier-heavy attention/embedding blocks) cause most of the quantisation error; keeping them in higher precision while compressing smooth FFN layers to INT4 gets far lower error for almost the same memory.',
        correct: true,
        why: 'In our 4-block lab, uniform INT4-g16 (5.0 b) got 0.0646 RMSE, while putting just Attn QKV in INT8 at 5.625 avg bits cut RMSE almost in half to 0.0334.',
      },
      {
        text: 'Because GPUs cannot run INT4 on attention layers.',
        why: 'GPUs can dequantise any layer from INT4; the choice is driven by accuracy sensitivity, not hardware inability.',
      },
      {
        text: 'Because mixed precision uses strictly fewer bits than uniform INT4.',
        why: 'Keeping a few blocks in INT8/INT6 raises average bits slightly (e.g. from 4.25 to ~4.8–5.6), buying a huge accuracy jump for that modest memory increase.',
      },
    ],
  },
];
