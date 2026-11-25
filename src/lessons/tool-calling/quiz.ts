// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'agents.who-executes',
    kind: 'choice',
    concepts: ['tool-calling'],
    prompt: 'When an LLM “calls a tool” like issue_refund(order_id="K-1042", amount_eur=599), what actually executes the refund?',
    options: [
      {
        text: 'Your host application parses the structured tool-call text the model emitted, validates it, runs the refund function, and pastes the JSON result back into the prompt.',
        correct: true,
        why: 'A language model only takes tokens in and outputs tokens. Tool calling is a protocol between the model and your host loop.',
      },
      {
        text: 'The transformer’s attention layers open a network socket to your payment API during the forward pass.',
        why: 'The forward pass is pure matrix multiplication on weights and activations; it has no network or database access.',
      },
      {
        text: 'The JSON schema executes the function automatically inside the tokeniser.',
        why: 'The schema only describes (or constrains) the format of the emitted tokens; your code still has to run the function.',
      },
    ],
  },
  {
    id: 'agents.constrained-decoding',
    kind: 'choice',
    concepts: ['structured-outputs', 'tool-calling'],
    prompt: 'You enable strict JSON-Schema constrained decoding for issue_refund. Which bug can STILL happen if you add no other checks?',
    options: [
      {
        text: 'The model emits a syntactically valid, schema-valid call that refunds €599 on a used machine (forgetting the €25 fee) or refunds an order that was already refunded.',
        correct: true,
        why: 'Structured outputs guarantee shape, types and enums — never business invariants or database state. Those must be enforced in your backend.',
      },
      {
        text: 'The model emits broken JSON with a trailing comma.',
        why: 'Syntax errors like trailing commas are exactly what grammar-constrained decoding prevents by masking illegal tokens.',
      },
      {
        text: 'The model passes amount_eur as the string "599" instead of a number.',
        why: 'Strict JSON Schema constrained decoding enforces field types as well as JSON syntax.',
      },
    ],
  },
  {
    id: 'agents.compound-num',
    kind: 'numeric',
    concepts: ['agent-loop'],
    tolerance: 0.5,
    make: (r) => {
      const pPct = [85, 90, 92, 95][r.int(4)];
      const n = [5, 8, 10, 12][r.int(4)];
      const p = pPct / 100;
      const ans = 100 * p ** n;
      return {
        prompt: `An agent runs ${n} independent sequential steps with no retries. Each step succeeds ${pPct}% of the time. What percentage of whole tasks succeed? (Round to one decimal place.)`,
        answer: Number(ans.toFixed(1)),
        explain: `All ${n} steps must succeed: (${p.toFixed(2)})^${n} = ${(p ** n).toFixed(4)}, or ${ans.toFixed(1)}%.`,
      };
    },
  },
  {
    id: 'agents.weak-verifier',
    kind: 'choice',
    concepts: ['agent-loop'],
    prompt: 'A step succeeds 90% of the time, and a fuzzy verifier catches only 50% of its errors. You raise max retries from 3 to 50. What happens to single-step reliability?',
    options: [
      {
        text: 'It stalls at about 94.7% and never approaches 100%, because half of all failures escape undetected on the very first time they occur.',
        correct: true,
        why: 'Exact ceiling is p / (1 − (1−p)d) = 0.90 / (1 − 0.05) = 94.74%. Retries only help on errors your check actually catches.',
      },
      {
        text: 'It climbs smoothly to 99.99%, because 50 retries eventually overcome any failure rate.',
        why: 'That only holds when d = 100% (every failure is caught). Silent failures never trigger a retry at all.',
      },
      {
        text: 'It drops below 90%, because retrying introduces new errors.',
        why: 'Retries only happen after a caught failure, so they can never make single-step reliability worse than 90%.',
      },
    ],
  },
  {
    id: 'agents.budget',
    kind: 'choice',
    concepts: ['agent-loop', 'tool-calling'],
    prompt: 'In the 4-stage pipeline lab, upgrading every stage to the Strong model (no checks) cost 12 credits and reached only 80.5% success — while a mixed pipeline beat 90% for 7.69 credits. Why?',
    options: [
      {
        text: 'Cheap deterministic checks + retries lift verifiable stages (schema lookup, arithmetic) above 99% on the Fast model, freeing budget to put the Strong model where checks are fuzzy.',
        correct: true,
        why: 'Replacing fuzzy LLM accuracy with deterministic verification is far cheaper than buying another 10% from a bigger model.',
      },
      {
        text: 'The Fast model has higher single-step accuracy than the Strong model.',
        why: 'The Fast model is less accurate at every single stage; it wins on verifiable stages only because cheap exact checks let it retry.',
      },
      {
        text: 'Retries eliminate all errors even without a verifier.',
        why: 'Without a check that detects failure, you never know when to retry.',
      },
    ],
  },
  {
    id: 'agents.idempotency',
    kind: 'choice',
    concepts: ['tool-calling'],
    prompt: 'Your agent’s issue_refund tool call times out after 10 seconds, so the loop automatically retries it. What makes that retry safe?',
    options: [
      {
        text: 'Checking state or passing an idempotency key (like the order ID) so the backend executes the refund at most once even if the call arrives twice.',
        correct: true,
        why: 'Read-only tools like lookup_order are naturally safe to retry; side-effecting tools like refunds need idempotency guards in your backend.',
      },
      {
        text: 'Asking the model in the system prompt “please do not issue duplicate refunds”.',
        why: 'Prompt instructions reduce mistakes, but after a network timeout neither the model nor the prompt knows whether the first request succeeded on the server.',
      },
      {
        text: 'Using temperature 0 so the model emits the exact same JSON.',
        why: 'Sending the exact same JSON twice without an idempotency guard still charges or refunds twice!',
      },
    ],
  },
];
