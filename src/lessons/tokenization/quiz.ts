// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'tokenization.tradeoff',
    kind: 'choice',
    concepts: ['tokenization'],
    prompt: 'Why do modern language models use subword tokens instead of single characters or whole words?',
    options: [
      {
        text: 'Single characters make sequences very long (multiplying n² attention cost), while whole words explode the vocabulary and break on unseen words; subwords compromise cleanly between the two.',
        correct: true,
        why: 'On our corpus, 28 BPE merges halve the sequence length (quartering n² attention work) while keeping a compact vocabulary — and unseen words like "unwatched" still break into known pieces rather than vanishing as <UNK>.',
      },
      {
        text: 'Neural networks cannot embed single characters into vectors.',
        why: 'Character-level transformers work fine mathematically — a 256-row embedding table is tiny. The trouble is sequence length: four times as many tokens means sixteen times as many attention pairs.',
      },
      {
        text: 'Subword tokens guarantee that every English word maps to its exact dictionary root.',
        why: 'BPE knows nothing about linguistics. It merges frequent pairs, which often look like suffixes (-est, -ing) but also produce arbitrary chunks like "berr" inside "strawberry".',
      },
    ],
  },
  {
    id: 'tokenization.strawberry',
    kind: 'choice',
    concepts: ['tokenization', 'bpe'],
    prompt: 'Why do large language models often stumble on tasks like counting the "r"s in "strawberry" or reversing a word?',
    options: [
      {
        text: 'The model never receives characters — only integer token IDs for whole chunks, so spelling has to be recalled indirectly from training rather than read off letter by letter.',
        correct: true,
        why: 'In our toy BPE, "strawberry" arrives as seven opaque IDs for st|r|a|w|berr|y|_. Two of the "r"s are hidden inside one token ID. Real models face the same blindness on a much larger vocabulary.',
      },
      {
        text: 'Attention layers cannot compare tokens that contain the same letter.',
        why: 'Attention compares any two token vectors; the bottleneck happens earlier, when text is chopped into integer IDs before the transformer ever sees it.',
      },
      {
        text: 'The causal mask prevents the model from seeing the end of the word.',
        why: 'When answering a prompt about "strawberry", the whole word is already in the prompt, so every token in the answer can attend to all of it.',
      },
    ],
  },
  {
    id: 'tokenization.byte-level',
    kind: 'choice',
    concepts: ['bpe'],
    prompt: 'Why do tokenisers like GPT-2’s and Llama 3’s start BPE from the 256 raw UTF-8 bytes instead of Unicode characters?',
    options: [
      {
        text: 'Any string in any language, code or emoji is already a sequence of those 256 bytes, so the tokeniser never needs an <UNK> token and never wastes thousands of vocabulary slots on rare characters.',
        correct: true,
        why: 'Unicode has over 150,000 characters, most of them rare. Starting from 256 bytes guarantees complete coverage with a tiny base alphabet — though scripts that take 2–4 UTF-8 bytes per character start at a length disadvantage.',
      },
      {
        text: 'Because UTF-8 uses exactly one byte for every character in every language.',
        why: 'Only basic ASCII characters take 1 byte in UTF-8. Greek takes 2 bytes, Japanese and Hindi take 3, and emoji take 4 — which is part of the non-English token tax.',
      },
      {
        text: 'Because BPE cannot merge strings longer than two bytes.',
        why: 'BPE merges pairs of tokens recursively, so tokens quickly grow to whole words and phrases of many bytes.',
      },
    ],
  },
  {
    id: 'tokenization.attn-savings',
    kind: 'numeric',
    concepts: ['tokenization', 'attention'],
    tolerance: 0.05,
    make: (r) => {
      const factor = [2, 3, 4][r.int(3)];
      const sq = factor * factor;
      return {
        prompt: `Suppose merging subwords makes a document ${factor}× shorter in tokens than character-level encoding. By what factor does the number of (n × n) attention pairs shrink?`,
        answer: sq,
        explain: `Attention compares every token with every other token, so pairs scale as n². Dividing n by ${factor} divides n² by ${factor}² = ${sq}×.`,
      };
    },
  },
  {
    id: 'tokenization.mdl',
    kind: 'choice',
    concepts: ['tokenization', 'bpe'],
    prompt: 'In the compression lab, why did total bits (corpus IDs + merge table) stop falling after 28 merges and start rising again?',
    options: [
      {
        text: 'Once the remaining pairs occur only 2 or 3 times, recording each new merge rule (and widening every token ID slightly) costs more bits than the handful of tokens it removes from the corpus.',
        correct: true,
        why: 'Merge 28 (n+g, ×4) saved 14.7 corpus bits at a cost of 12.9 table bits; merge 29 (s+at_, ×3) saved only 9.5 corpus bits against 12.9 table bits. Real models face the analogous trade-off between shorter sequences and a larger V × d embedding table.',
      },
      {
        text: 'Because after 28 merges, BPE begins splitting tokens back into characters.',
        why: 'BPE only ever merges pairs; corpus token count drops on every single merge all the way to step 66.',
      },
      {
        text: 'Because corpus-only bits (tokens × log₂ V) started rising after 28 merges.',
        why: 'Corpus-only bits fell monotonically all the way from 2,368 at 0 merges to 1,143 at 66. The sweet spot appears only when you also pay for the vocabulary / merge table.',
      },
    ],
  },
];
