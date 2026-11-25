// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { CUSTOMER_TICKETS, GATE_PRESETS, initialShopState, runTicketTrace, validateToolCall } from './shop';
import {
  LAB_BRIEFS,
  STAGE_OPTIONS,
  evaluatePipeline,
  labExhaustive,
  simulatePipeline,
  stepMetrics,
  uniformPipeline,
  type PipelineConfig,
} from './reliability';

const p1 = (v: number) => (100 * v).toFixed(1);
const p2 = (v: number) => (100 * v).toFixed(2);

describe('Kestrel shop environment & 3-stage gatekeeper', () => {
  it('all 4 customer tickets run to clean completion with real state mutations', () => {
    const unused = runTicketTrace('unused');
    expect(unused).toHaveLength(3);
    expect(unused.filter((s) => s.kind === 'tool_call').map((s) => s.call!.tool)).toEqual(['lookup_order', 'issue_refund']);
    expect(unused[2].stateAfter.orders.find((o) => o.order_id === 'K-1042')!.status).toBe('refunded');
    expect(unused[2].stateAfter.refunds.find((r) => r.order_id === 'K-1042')!.amount_eur).toBe(599);

    const used = runTicketTrace('used');
    expect(used).toHaveLength(3);
    expect(used[2].stateAfter.refunds.find((r) => r.order_id === 'K-1088')!.amount_eur).toBe(574);

    const warranty = runTicketTrace('warranty');
    expect(warranty).toHaveLength(4);
    expect(warranty.filter((s) => s.kind === 'tool_call').map((s) => s.call!.tool)).toEqual([
      'lookup_order',
      'check_stock',
      'create_repair_ticket',
    ]);
    expect(warranty[3].stateAfter.orders.find((o) => o.order_id === 'K-1103')!.status).toBe('repair_open');
    expect(warranty[3].stateAfter.inventory.find((i) => i.sku === 'K2-PUMP')!.in_stock).toBe(3);

    const missing = runTicketTrace('missing');
    expect(missing).toHaveLength(2);
    expect(missing[0].toolOutput!.error).toBe('ORDER_NOT_FOUND');
    expect(missing[1].kind).toBe('final_answer');
    expect(CUSTOMER_TICKETS).toHaveLength(4);
  });

  it('3-stage gatekeeper separates syntax, schema, and business-rule failures', () => {
    const st = initialShopState();
    const byId = Object.fromEntries(GATE_PRESETS.map((p) => [p.id, validateToolCall(p.raw, st)]));

    expect([byId.valid.syntax.ok, byId.valid.schema.ok, byId.valid.business.ok]).toEqual([true, true, true]);
    expect([byId.syntax.syntax.ok, byId.syntax.schema.ok, byId.syntax.business.ok]).toEqual([false, false, false]);
    expect([byId.schema.syntax.ok, byId.schema.schema.ok, byId.schema.business.ok]).toEqual([true, false, false]);
    expect([byId.overrefund.syntax.ok, byId.overrefund.schema.ok, byId.overrefund.business.ok]).toEqual([true, true, false]);
    expect([byId.duplicate.syntax.ok, byId.duplicate.schema.ok, byId.duplicate.business.ok]).toEqual([true, true, false]);
  });
});

describe('compounding reliability & verifier maths', () => {
  it('single-step probabilities sum to 1 for all (p, d, retries)', () => {
    for (const p of [0.7, 0.9, 0.97])
      for (const d of [0, 0.5, 0.95, 1])
        for (const r of [0, 1, 2, 5]) {
          const s = stepMetrics(p, d, r);
          expect(s.pOk + s.pSilent + s.pAbort).toBeCloseTo(1, 12);
          if (d === 0) {
            expect(s.pOk).toBeCloseTo(p, 12);
            expect(s.pAbort).toBe(0);
            expect(s.expectedAttempts).toBe(1);
          }
          if (d === 1) {
            expect(s.pSilent).toBeCloseTo(0, 12);
            expect(s.pOk).toBeCloseTo(1 - (1 - p) ** (r + 1), 12);
          }
        }
  });

  it('pins the numbers quoted in the compounding section', () => {
    const raw10 = uniformPipeline(10, 0.9, 0, 0);
    expect(p1(raw10.pTask)).toBe('34.9');

    const raw20 = uniformPipeline(20, 0.9, 0, 0);
    expect(p1(raw20.pTask)).toBe('12.2');

    const det1 = uniformPipeline(10, 0.9, 1, 1);
    expect(p1(det1.stepOk)).toBe('99.0');
    expect(p1(det1.pTask)).toBe('90.4');
    expect(det1.callsMultiplier.toFixed(2)).toBe('1.05');

    const det2 = uniformPipeline(10, 0.9, 1, 2);
    expect(p1(det2.pTask)).toBe('99.0');
    expect(det2.callsMultiplier.toFixed(2)).toBe('1.11');

    const weak3 = uniformPipeline(10, 0.9, 0.5, 3);
    expect(p1(weak3.stepOk)).toBe('94.7');
    expect(p1(weak3.pTask)).toBe('58.2');
    expect(p1(weak3.stepCeiling)).toBe('94.7');
  });

  it('closed-form pipeline evaluation matches 100,000-trial Monte Carlo simulation', () => {
    const cfg: PipelineConfig = [
      { model: 'fast', check: true, retries: 1 },
      { model: 'strong', check: true, retries: 1 },
      { model: 'fast', check: true, retries: 2 },
      { model: 'fast', check: true, retries: 1 },
    ];
    const exact = evaluatePipeline(cfg);
    const mc = simulatePipeline(cfg, 100_000, 123);
    expect(Math.abs(exact.pTask - mc.pTask)).toBeLessThan(0.004);
    expect(Math.abs(exact.pSilentAny - mc.pSilentAny)).toBeLessThan(0.003);
    expect(Math.abs(exact.pAbortAny - mc.pAbortAny)).toBeLessThan(0.003);
    expect(Math.abs(exact.pBadAction - mc.pBadAction)).toBeLessThan(0.002);
    expect(Math.abs(exact.expectedCost - mc.expectedCost)).toBeLessThan(0.02);
  });
});

describe('Lab: Harden the Agent Pipeline (exhaustive over 4,096 configs)', () => {
  it('pins unguarded baselines: all-Fast (52.0%, 4.00) and all-Strong (80.5%, 12.00) both fail all briefs', () => {
    const allFast = evaluatePipeline([
      { model: 'fast', check: false, retries: 0 },
      { model: 'fast', check: false, retries: 0 },
      { model: 'fast', check: false, retries: 0 },
      { model: 'fast', check: false, retries: 0 },
    ]);
    const allStrong = evaluatePipeline([
      { model: 'strong', check: false, retries: 0 },
      { model: 'strong', check: false, retries: 0 },
      { model: 'strong', check: false, retries: 0 },
      { model: 'strong', check: false, retries: 0 },
    ]);
    expect(p1(allFast.pTask)).toBe('52.0');
    expect(allFast.expectedCost.toFixed(2)).toBe('4.00');
    expect(p1(allStrong.pTask)).toBe('80.5');
    expect(allStrong.expectedCost.toFixed(2)).toBe('12.00');
  });

  it('pins exhaustive counts and unique cheapest optimums for all 3 briefs', () => {
    const [b1, b2, b3] = LAB_BRIEFS.map(labExhaustive);

    expect([b1.total, b1.passing, b1.bestCost]).toEqual([4096, 314, 7.69]);
    expect(b1.best).toHaveLength(1);
    expect(b1.best[0].cfg).toEqual([
      { model: 'fast', check: true, retries: 1 },
      { model: 'strong', check: true, retries: 0 },
      { model: 'fast', check: true, retries: 2 },
      { model: 'fast', check: true, retries: 1 },
    ]);
    expect(p1(b1.best[0].eval.pTask)).toBe('90.6');
    expect(p1(b1.best[0].eval.pSilentAny)).toBe('2.0');

    expect([b2.total, b2.passing, b2.bestCost]).toEqual([4096, 572, 7.72]);
    expect(b2.best).toHaveLength(1);
    expect(b2.best[0].cfg).toEqual([
      { model: 'fast', check: true, retries: 1 },
      { model: 'fast', check: true, retries: 1 },
      { model: 'fast', check: true, retries: 1 },
      { model: 'strong', check: true, retries: 0 },
    ]);
    expect(p1(b2.best[0].eval.pTask)).toBe('85.4');
    expect(p2(b2.best[0].eval.pBadAction)).toBe('0.14');

    expect([b3.total, b3.passing, b3.bestCost]).toEqual([4096, 54, 9.90]);
    expect(b3.best).toHaveLength(1);
    expect(b3.best[0].cfg).toEqual([
      { model: 'fast', check: true, retries: 1 },
      { model: 'strong', check: true, retries: 1 },
      { model: 'fast', check: true, retries: 2 },
      { model: 'strong', check: true, retries: 1 },
    ]);
    expect(p1(b3.best[0].eval.pTask)).toBe('95.8');
    expect(p1(b3.best[0].eval.pSilentAny)).toBe('1.7');

    // No all-Fast config passes Brief 1 or Brief 3
    let fastB1 = 0;
    let fastB3 = 0;
    for (const a of STAGE_OPTIONS.slice(0, 4))
      for (const b of STAGE_OPTIONS.slice(0, 4))
        for (const c of STAGE_OPTIONS.slice(0, 4))
          for (const d of STAGE_OPTIONS.slice(0, 4)) {
            const ev = evaluatePipeline([a, b, c, d]);
            if (ev.pTask >= 0.9 && ev.pSilentAny <= 0.05) fastB1++;
            if (ev.pTask >= 0.95 && ev.pSilentAny <= 0.02) fastB3++;
          }
    expect(fastB1).toBe(0);
    expect(fastB3).toBe(0);
  });
});
