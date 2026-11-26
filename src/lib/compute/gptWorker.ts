// authored-by: Claude Opus 5.5
/**
 * Web Worker for the tiny GPT's CPU tier (V1 capstone, A2): a multi-minute training run must not
 * freeze the page. Messages: { id, op, args } → { id, result } or { id, error }.
 * Same `Gpt` class and float32 maths as everywhere else, so results are bit-identical to Node.
 */
import { Gpt, type AdamWOpts, type GptConfig } from './gpt';

type Msg =
  | { id: number; op: 'init'; cfg: GptConfig }
  | { id: number; op: 'setParams'; p: Float32Array }
  | { id: number; op: 'train'; batches: Int32Array; K: number; B: number; T: number; lr: Float32Array; o: AdamWOpts }
  | { id: number; op: 'evalLoss'; batch: Int32Array; B: number; T: number }
  | { id: number; op: 'readParams' };

const scope = self as unknown as { onmessage: ((e: MessageEvent<Msg>) => void) | null; postMessage(m: unknown, transfer?: Transferable[]): void };
let model: Gpt | null = null;

scope.onmessage = (e) => {
  const msg = e.data;
  try {
    if (msg.op === 'init') {
      model = new Gpt(msg.cfg);
      scope.postMessage({ id: msg.id, result: null });
      return;
    }
    if (!model) throw new Error('init first');
    switch (msg.op) {
      case 'setParams':
        model.load(msg.p);
        model.resetOptimiser();
        scope.postMessage({ id: msg.id, result: null });
        break;
      case 'train': {
        const { batches, K, B, T, lr, o } = msg;
        const row = B * (T + 1);
        const losses = new Float32Array(K);
        for (let k = 0; k < K; k++) losses[k] = model.trainStep(batches.subarray(k * row, (k + 1) * row), B, T, lr[k], o);
        scope.postMessage({ id: msg.id, result: losses }, [losses.buffer]);
        break;
      }
      case 'evalLoss':
        scope.postMessage({ id: msg.id, result: model.evalLoss(msg.batch, msg.B, msg.T) });
        break;
      case 'readParams': {
        const p = model.snapshot();
        scope.postMessage({ id: msg.id, result: p }, [p.buffer]);
        break;
      }
    }
  } catch (err) {
    scope.postMessage({ id: msg.id, error: String(err) });
  }
};
