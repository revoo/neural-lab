// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'rag.why',
    kind: 'choice',
    concepts: ['rag'],
    prompt: 'Why do most teams answer questions about their own documents with RAG rather than by fine-tuning the model on them?',
    options: [
      {
        text: 'Retrieval puts the exact current text in front of the model at question time, so updating a document updates the answers immediately, and the model reads rather than remembers.',
        correct: true,
        why: 'Fine-tuning is slow, has to be redone when pages change, and still doesn’t guarantee the model recalls a specific fact.',
      },
      {
        text: 'Fine-tuning can’t teach a model new facts at all.',
        why: 'It can teach some, but unreliably and expensively; RAG is simply the cheaper, fresher, more checkable option for looking things up.',
      },
      {
        text: 'RAG makes the model itself smarter.',
        why: 'The model is unchanged. RAG changes what it is given to read.',
      },
    ],
  },
  {
    id: 'rag.lexsem',
    kind: 'choice',
    concepts: ['embedding', 'rag'],
    prompt: 'Lexical search finds “What does E2 mean?” but misses “How soon do I get my money back?”. Semantic search does the opposite. Why?',
    options: [
      {
        text: 'Lexical search needs shared words and loves rare exact ones like “E2”; semantic search matches meanings across different words but has no useful meaning for an identifier it never learned.',
        correct: true,
        why: 'Their failures are complementary, which is why a hybrid found 97.1% where each alone found about 80%.',
      },
      {
        text: 'Semantic search is simply worse; it should be replaced by lexical search.',
        why: 'Semantic search found the reworded questions that lexical search scored zero. Neither dominates.',
      },
      {
        text: 'The questions were mislabelled.',
        why: 'The labels are right. The failures come from how each method turns text into vectors.',
      },
    ],
  },
  {
    id: 'rag.cosine',
    kind: 'numeric',
    concepts: ['cosine-similarity'],
    tolerance: 0.005,
    make: (r) => {
      const q = [[1, 0, 1], [2, 1, 0], [1, 1, 1]][r.int(3)];
      const d = [[1, 1, 0], [0, 2, 2], [3, 0, 1]][r.int(3)];
      const dot = q.reduce((s, v, i) => s + v * d[i], 0);
      const c = dot / (Math.hypot(...q) * Math.hypot(...d));
      return {
        prompt: `A question vector is (${q.join(', ')}) and a passage vector is (${d.join(', ')}). What is their cosine similarity? (Three decimal places.)`,
        answer: Number(c.toFixed(3)),
        explain: `Dot product ${dot}, lengths ${Math.hypot(...q).toFixed(3)} and ${Math.hypot(...d).toFixed(3)}, so cosine = ${dot} / (${Math.hypot(...q).toFixed(3)} × ${Math.hypot(...d).toFixed(3)}) = ${c.toFixed(3)}.`,
      };
    },
  },
  {
    id: 'rag.chunks',
    kind: 'choice',
    concepts: ['rag'],
    prompt: 'Making chunks bigger raises recall. Why not just use very big chunks?',
    options: [
      {
        text: 'Every retrieved chunk is pasted into the prompt on every request, so bigger chunks cost more money and latency and bury the answer among more irrelevant text.',
        correct: true,
        why: 'In the lab, finding every answer cost 82.3 words per question; accepting 90% recall cut that to 33.',
      },
      {
        text: 'Bigger chunks always lower recall.',
        why: 'They usually raise it: the answer only has to be somewhere inside a retrieved chunk.',
      },
      {
        text: 'Embedding models can’t handle more than one sentence.',
        why: 'They handle hundreds of tokens. The limit that matters is the cost of the prompt.',
      },
    ],
  },
  {
    id: 'rag.hallucinate',
    kind: 'choice',
    concepts: ['rag'],
    prompt: 'A RAG support bot gives a confident, wrong answer. What should you check first?',
    options: [
      {
        text: 'Whether the passage containing the right answer was retrieved at all, by looking at what went into the prompt.',
        correct: true,
        why: 'Most RAG failures are retrieval failures. If the answer wasn’t in the prompt, no amount of prompt wording will fix it.',
      },
      {
        text: 'Whether the model is large enough.',
        why: 'A bigger model reads better, but can’t read a passage that isn’t there.',
      },
      {
        text: 'Nothing: RAG prevents hallucination, so the documents must be wrong.',
        why: 'RAG only helps when retrieval succeeds. When it misses, the model is back to guessing, possibly from an irrelevant passage.',
      },
    ],
  },
  {
    id: 'rag.overfit',
    kind: 'choice',
    concepts: ['rag'],
    prompt: 'You tried 125 retriever settings on 34 labelled questions and picked the best. Why is its score probably optimistic?',
    options: [
      {
        text: 'The setting was chosen because it did well on exactly these questions, partly thanks to their quirks. Its score on new questions will usually be lower; measure that on a separate set you didn’t tune on.',
        correct: true,
        why: 'This is overfitting again, with settings instead of weights. Hold out a test set.',
      },
      {
        text: 'It isn’t: the evaluation was exact.',
        why: 'The measurement is exact for these questions. The question is how well it predicts the next ones.',
      },
      {
        text: 'Because 34 is an odd number.',
        why: 'The problem is small samples and selection, not parity.',
      },
    ],
  },
];
