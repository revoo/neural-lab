// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  CUSTOMER_TICKETS,
  GATE_PRESETS,
  POLICY_RULES,
  SYSTEM_PROMPT,
  initialShopState,
  runTicketTrace,
  validateToolCall,
  type CustomerTicket,
} from './shop';
import {
  LAB_BRIEFS,
  LAB_STAGES,
  briefPasses,
  evaluatePipeline,
  formatStageConfig,
  labExhaustive,
  uniformPipeline,
  type LabBrief,
  type PipelineConfig,
  type StageConfig,
} from './reliability';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;
const pct2 = (v: number) => `${(100 * v).toFixed(2)}%`;

// ------------------------------------------------------------------ 1. Inside the Agent Loop

export function AgentLoopWidget() {
  const [ticketId, setTicketId] = useState<CustomerTicket['id']>('unused');
  const [turn, setTurn] = useState(0); // 0 = before first step, 1..steps.length = after step k
  const [showPrompt, setShowPrompt] = useState(false);
  const [showRules, setShowRules] = useState(false);

  const trace = useMemo(() => runTicketTrace(ticketId), [ticketId]);
  const ticket = CUSTOMER_TICKETS.find((t) => t.id === ticketId)!;
  const clampedTurn = Math.min(turn, trace.length);
  const currentStep = clampedTurn > 0 ? trace[clampedTurn - 1] : null;
  const state = currentStep ? currentStep.stateAfter : initialShopState();

  const selectTicket = (id: CustomerTicket['id']) => {
    setTicketId(id);
    setTurn(0);
  };

  return (
    <Widget
      title="Inside the agent loop"
      wide
      caption={
        <>
          Pick a customer ticket and step through the loop one turn at a time. <strong>No language model runs here</strong>: a small, transparent rule-based policy (click “Show policy rules” to read it) stands in for the model’s decisions. Everything else — the JSON tool calls, schema validation, database state and exact prompt transcript — is real and computed live.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        {CUSTOMER_TICKETS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`btn small${t.id === ticketId ? ' primary' : ''}`}
            onClick={() => selectTicket(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="btn-row" style={{ marginBottom: 10, alignItems: 'center' }}>
        <button
          type="button"
          className="btn primary"
          disabled={clampedTurn >= trace.length}
          onClick={() => setTurn((n) => Math.min(trace.length, n + 1))}
        >
          {clampedTurn === 0 ? 'Step 1 (start loop)' : clampedTurn >= trace.length ? 'Loop finished' : `Step ${clampedTurn + 1} of ${trace.length}`}
        </button>
        <button type="button" className="btn" disabled={clampedTurn >= trace.length} onClick={() => setTurn(trace.length)}>
          Run all
        </button>
        <button type="button" className="btn" disabled={clampedTurn === 0} onClick={() => setTurn(0)}>
          Reset
        </button>
        <Toggle label="Show exact prompt" checked={showPrompt} onChange={setShowPrompt} />
        <Toggle label="Show policy rules" checked={showRules} onChange={setShowRules} />
      </div>

      {showRules && (
        <div className="readout" style={{ marginBottom: 10, whiteSpace: 'pre-wrap', fontSize: 12.5, lineHeight: 1.55 }}>
          <span className="k">Deterministic stand-in policy (standing in for an LLM):{'\n'}</span>
          {POLICY_RULES.join('\n')}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        {/* Left: Conversation & tool trace */}
        <div style={{ display: 'grid', gap: 8 }}>
          <div className="field-label">Agent loop trace ({clampedTurn}/{trace.length} turns)</div>
          <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 13.5 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>USER MESSAGE</div>
            {ticket.message}
          </div>

          {trace.slice(0, clampedTurn).map((st) => (
            <div
              key={st.step}
              style={{
                padding: '8px 10px',
                borderRadius: 8,
                background: 'var(--bg-sunken)',
                border: `1px solid ${st.step === clampedTurn ? 'var(--accent)' : 'var(--border)'}`,
                fontSize: 13,
                display: 'grid',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                <span>TURN {st.step} · {st.kind === 'tool_call' ? 'MODEL EMITS TOOL CALL' : 'FINAL ANSWER'}</span>
                {st.kind === 'tool_call' && (
                  <span style={{ color: st.validation?.overallOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {st.validation?.overallOk ? '✓ schema + guards ok' : '✗ blocked'}
                  </span>
                )}
              </div>

              {st.kind === 'tool_call' ? (
                <>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent)', wordBreak: 'break-word' }}>
                    → {st.callJson}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: st.toolOutput?.error ? 'var(--c-bad)' : 'var(--c-c)', wordBreak: 'break-word' }}>
                    ← host returns: {JSON.stringify(st.toolOutput)}
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 13.5, lineHeight: 1.5, color: 'var(--text)' }}>{st.finalAnswer}</div>
              )}
            </div>
          ))}
        </div>

        {/* Right: Live shop database */}
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="field-label">Live Kestrel Shop Database (Host State)</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="num-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Order</th>
                  <th>Days</th>
                  <th>Used</th>
                  <th>Price</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {state.orders.map((o) => {
                  const hi = currentStep?.touchedOrder === o.order_id;
                  const bg = hi ? (currentStep?.mutated ? 'color-mix(in srgb, var(--c-good) 16%, transparent)' : 'color-mix(in srgb, var(--c-a) 16%, transparent)') : undefined;
                  return (
                    <tr key={o.order_id} style={{ background: bg }}>
                      <td style={{ textAlign: 'left', fontWeight: hi ? 700 : 400 }}>{o.order_id}</td>
                      <td>{o.days_since_delivery}d</td>
                      <td>{o.used ? 'yes' : 'no'}</td>
                      <td>€{o.price_eur}</td>
                      <td style={{ color: o.status === 'delivered' ? 'var(--text-muted)' : 'var(--c-good)', fontWeight: o.status === 'delivered' ? 400 : 600 }}>
                        {o.status}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="num-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Part SKU</th>
                  <th style={{ textAlign: 'left' }}>Description</th>
                  <th>In stock</th>
                </tr>
              </thead>
              <tbody>
                {state.inventory.map((i) => {
                  const hi = currentStep?.touchedSku === i.sku;
                  const bg = hi ? (currentStep?.mutated ? 'color-mix(in srgb, var(--c-good) 16%, transparent)' : 'color-mix(in srgb, var(--c-a) 16%, transparent)') : undefined;
                  return (
                    <tr key={i.sku} style={{ background: bg }}>
                      <td style={{ textAlign: 'left', fontWeight: hi ? 700 : 400 }}>{i.sku}</td>
                      <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)', fontSize: 12.5 }}>{i.name}</td>
                      <td>{i.in_stock}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="readout" style={{ fontSize: 12 }}>
            <span className="k">refunds ledger ({state.refunds.length}): </span>
            {state.refunds.map((r) => `${r.refund_id} ${r.order_id} €${r.amount_eur}`).join(' · ')}
            {'\n'}
            <span className="k">repair tickets ({state.tickets.length}): </span>
            {state.tickets.length === 0 ? '(none yet)' : state.tickets.map((t) => `${t.ticket_id} ${t.order_id} ${t.part_sku}`).join(' · ')}
          </div>
        </div>
      </div>

      {showPrompt && (
        <div className="readout" style={{ marginTop: 12, whiteSpace: 'pre-wrap', fontSize: 12, lineHeight: 1.5, maxHeight: 260, overflowY: 'auto' }}>
          <span className="k">[SYSTEM PROMPT + TOOL SCHEMAS]{'\n'}</span>
          {SYSTEM_PROMPT}
          {'\n\n'}
          {(currentStep ? currentStep.transcriptSoFar : [{ role: 'user' as const, text: ticket.message }]).map((t, i) => (
            <span key={i}>
              <span className="k">[{t.role.toUpperCase()}] </span>
              {t.text}
              {'\n'}
            </span>
          ))}
        </div>
      )}

      <div className="live-note" style={{ marginTop: 10 }}>
        {currentStep
          ? currentStep.note
          : 'Before Turn 1: the host sends the system prompt (with the 4 tool definitions) plus the customer’s message. Press “Step 1” to see what the policy emits.'}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. The Schema Gatekeeper

export function SchemaGatekeeper() {
  const [presetId, setPresetId] = useState(GATE_PRESETS[0].id);
  const [raw, setRaw] = useState(GATE_PRESETS[0].raw);
  const state = useMemo(() => initialShopState(), []);
  const report = useMemo(() => validateToolCall(raw, state), [raw, state]);

  const choose = (id: string) => {
    const p = GATE_PRESETS.find((x) => x.id === id)!;
    setPresetId(id);
    setRaw(p.raw);
  };

  const stages = [
    { num: 1, title: 'JSON syntax', subtitle: 'Preventable by constrained decoding', res: report.syntax },
    { num: 2, title: 'Tool schema & types', subtitle: 'Preventable by structured outputs', res: report.schema },
    { num: 3, title: 'Shop state & business rules', subtitle: 'Only your backend code can check this', res: report.business },
  ];

  return (
    <Widget
      title="The three-stage tool gatekeeper"
      wide
      caption={
        <>
          Pick a realistic raw model output — or edit the JSON yourself — and watch it pass through all three gates against the live Kestrel database (where K-1042 is unused €599, K-1088 is used €599, and K-1015 is already refunded). Try changing K-1088’s <code>amount_eur</code> between 574 and 599.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        {GATE_PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`btn small${p.id === presetId ? ' primary' : ''}`}
            onClick={() => choose(p.id)}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        <div>
          <div className="field-label">Raw text emitted by model (editable)</div>
          <textarea
            value={raw}
            onChange={(e) => {
              setPresetId('custom');
              setRaw(e.target.value);
            }}
            rows={9}
            spellCheck={false}
            aria-label="Raw model JSON output"
            style={{
              width: '100%',
              fontFamily: 'var(--font-mono)',
              fontSize: 12.5,
              lineHeight: 1.5,
              padding: '8px 10px',
              borderRadius: 8,
              border: '1px solid var(--border)',
              background: 'var(--bg-sunken)',
              color: 'var(--text)',
              resize: 'vertical',
            }}
          />
        </div>

        <div style={{ display: 'grid', gap: 8 }}>
          <div className="field-label">Validation pipeline</div>
          {stages.map((s) => (
            <div
              key={s.num}
              style={{
                padding: '8px 10px',
                borderRadius: 8,
                background: 'var(--bg-sunken)',
                border: `1px solid ${s.res.ok ? 'var(--c-good)' : 'var(--c-bad)'}`,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 13, fontWeight: 650 }}>
                <span>
                  <span style={{ color: s.res.ok ? 'var(--c-good)' : 'var(--c-bad)', marginRight: 6 }}>{s.res.ok ? '✓' : '✗'}</span>
                  Gate {s.num}: {s.title}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>{s.subtitle}</span>
              </div>
              <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 4 }}>{s.res.detail}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {report.overallOk
          ? 'All three gates passed: safe to execute the tool and append its output to the conversation.'
          : !report.syntax.ok
            ? 'Caught at Gate 1 (syntax). Grammar-constrained decoding eliminates malformed JSON by masking illegal tokens before softmax — but notice what it still leaves for Gate 3.'
            : !report.schema.ok
              ? 'Caught at Gate 2 (schema). Strict JSON Schema enforcement blocks wrong types, missing keys and hallucinated enums before your tool function is ever called.'
              : 'Passed Gates 1 and 2, caught at Gate 3 (business logic)! Structured outputs guarantee the shape of a call, never whether the refund amount is lawful or whether the order was already refunded.'}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. The Compounding Cliff

export function CompoundingCliff() {
  const [p, setP] = useState(0.9);
  const [n, setN] = useState(10);
  const [d, setD] = useState(0);
  const [retries, setRetries] = useState(0);

  const raw = useMemo(() => uniformPipeline(n, p, 0, 0), [n, p]);
  const hardened = useMemo(() => uniformPipeline(n, p, d, retries), [n, p, d, retries]);

  const rawPts = useMemo(() => Array.from({ length: 20 }, (_, i) => ({ k: i + 1, v: uniformPipeline(i + 1, p, 0, 0).pTask })), [p]);
  const hardPts = useMemo(
    () => Array.from({ length: 20 }, (_, i) => ({ k: i + 1, v: uniformPipeline(i + 1, p, d, retries).pTask })),
    [p, d, retries],
  );
  const n99Pts = useMemo(() => Array.from({ length: 20 }, (_, i) => ({ k: i + 1, v: 0.99 ** (i + 1) })), []);

  const hasGuard = d > 0;

  return (
    <Widget
      title="The compounding cliff (and what retries can and cannot fix)"
      wide
      caption={
        <>
          Set single-step reliability <strong>p</strong> and step count <strong>n</strong>, then turn on a verifier (detection rate <strong>d</strong>) and retries. Try <code>p = 0.90, n = 10</code> with no verifier (34.9%), then deterministic check <code>d = 100%</code> + 1 retry (90.4% for 1.05× calls), then a weak 50% verifier with 3 retries (stalls at 58.2% because half of errors slip past undetected).
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 10 }}>
          <Slider label="single-step reliability p" value={p} min={0.7} max={0.99} step={0.01} onChange={setP} format={pct1} color="c-a" />
          <Slider label="number of sequential steps n" value={n} min={1} max={20} step={1} onChange={setN} format={(v) => `${v} steps`} color="c-b" />
          <Segmented
            label="Verifier catch rate"
            value={d}
            onChange={(v) => {
              setD(v);
              if (v === 0) setRetries(0);
              else if (retries === 0) setRetries(1);
            }}
            options={[
              { value: 0, label: 'None (0%)' },
              { value: 0.5, label: 'Weak (50%)' },
              { value: 0.8, label: 'Judge (80%)' },
              { value: 1, label: 'Exact (100%)' },
            ]}
          />
          <Segmented
            label="Max retries per step"
            value={d === 0 ? 0 : retries}
            onChange={setRetries}
            options={
              d === 0
                ? [{ value: 0, label: '0 (needs verifier)' }]
                : [0, 1, 2, 3].map((r) => ({ value: r, label: String(r) }))
            }
          />
          <div className="readout">
            <span className="k">raw agent (pⁿ) at {n} steps </span>
            <span style={{ color: 'var(--c-a)', fontWeight: 650 }}>{pct1(raw.pTask)}</span>
            {'\n'}
            <span className="k">with verifier + {d === 0 ? 0 : retries} retries </span>
            <span style={{ color: 'var(--c-c)', fontWeight: 650 }}>{pct1(hardened.pTask)}</span>
            {'\n'}
            <span className="k">single-step success (ceiling {pct1(hardened.stepCeiling)}) </span>
            {pct1(hardened.stepOk)}
            {'\n'}
            <span className="k">silent failure / clean abort </span>
            {pct1(hardened.pSilentAny)} / {pct1(hardened.pAbortAny)}
            {'\n'}
            <span className="k">expected model calls </span>
            {hardened.expectedCalls.toFixed(2)} ({hardened.callsMultiplier.toFixed(2)}×)
          </div>
        </div>

        <div>
          <Plot width={440} height={270} x={[1, 20]} y={[0, 1]} pad={34} xLabel="steps (n)" yLabel="P(task succeeds)" ariaLabel="Whole-task success rate vs number of steps">
            {({ x, y }) => {
              const pathOf = (pts: { k: number; v: number }[]) =>
                pts.map((pt, i) => `${i === 0 ? 'M' : 'L'}${x(pt.k).toFixed(1)},${y(pt.v).toFixed(1)}`).join(' ');
              return (
                <>
                  {/* Reference 99% per-step curve */}
                  <path d={pathOf(n99Pts)} fill="none" stroke="var(--text-muted)" strokeWidth={1.2} strokeDasharray="4 4" opacity={0.55} />
                  {/* Raw p^n */}
                  <path d={pathOf(rawPts)} fill="none" stroke="var(--c-a)" strokeWidth={2.5} />
                  {/* Hardened */}
                  {hasGuard && <path d={pathOf(hardPts)} fill="none" stroke="var(--c-c)" strokeWidth={2.5} />}
                  {/* Vertical marker at n */}
                  <line x1={x(n)} x2={x(n)} y1={y(0)} y2={y(1)} stroke="var(--c-b)" strokeWidth={1.2} strokeDasharray="3 3" />
                  <circle cx={x(n)} cy={y(raw.pTask)} r={4.5} fill="var(--c-a)" />
                  {hasGuard && <circle cx={x(n)} cy={y(hardened.pTask)} r={4.5} fill="var(--c-c)" />}
                  {/* Compact inline legend */}
                  <text x={x(19.6)} y={y(n99Pts[19].v) - 6} textAnchor="end" fontSize={10.5} fill="var(--text-muted)">
                    99%/step ref
                  </text>
                </>
              );
            }}
          </Plot>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            <span>
              <span style={{ color: 'var(--c-a)', fontWeight: 700 }}>━</span> Raw pⁿ ({pct1(raw.pTask)})
            </span>
            {hasGuard && (
              <span>
                <span style={{ color: 'var(--c-c)', fontWeight: 700 }}>━</span> Verified + retries ({pct1(hardened.pTask)})
              </span>
            )}
            <span>
              <span style={{ color: 'var(--c-b)', fontWeight: 700 }}>┆</span> n = {n} steps
            </span>
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {!hasGuard
          ? `Without verification, errors multiply silently: ${pct1(p)} per step drops to ${pct1(raw.pTask)} over ${n} steps. Turn on a verifier to see how retries change the curve.`
          : d === 1
            ? `With a 100% deterministic check (like a schema validator, compiler or test suite) and ${retries} ${retries === 1 ? 'retry' : 'retries'}, every step rises from ${pct1(p)} to ${pct1(hardened.stepOk)}, lifting the ${n}-step task from ${pct1(raw.pTask)} to ${pct1(hardened.pTask)} for only ${hardened.callsMultiplier.toFixed(2)}× as many calls.`
            : `Because this verifier catches only ${pct1(d)} of failures, ${(100 * (1 - d)).toFixed(0)}% slip past silently. Even with infinite retries, each step hits a hard ceiling of ${pct1(hardened.stepCeiling)}, leaving ${pct1(hardened.pSilentAny)} silent failures across ${n} steps.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Harden the Agent Pipeline

const LAB_KEY = 'nl-agent-lab-v1';

type LabEntry = {
  cfg: PipelineConfig;
  solved: boolean;
  bestCost: number | null;
};

type LabState = Record<LabBrief['id'], LabEntry>;

const DEFAULT_CFG: PipelineConfig = [
  { model: 'fast', check: false, retries: 0 },
  { model: 'fast', check: false, retries: 0 },
  { model: 'fast', check: false, retries: 0 },
  { model: 'fast', check: false, retries: 0 },
];

function cloneCfg(c: PipelineConfig): PipelineConfig {
  return [
    { ...c[0] },
    { ...c[1] },
    { ...c[2] },
    { ...c[3] },
  ];
}

function isValidStage(x: unknown): x is StageConfig {
  if (!x || typeof x !== 'object') return false;
  const r = x as Record<string, unknown>;
  return (r.model === 'fast' || r.model === 'strong') && typeof r.check === 'boolean' && (r.retries === 0 || r.retries === 1 || r.retries === 2);
}

function loadLab(): LabState {
  const fresh: LabState = {
    reliable: { cfg: cloneCfg(DEFAULT_CFG), solved: false, bestCost: null },
    'safe-refund': { cfg: cloneCfg(DEFAULT_CFG), solved: false, bestCost: null },
    strict: { cfg: cloneCfg(DEFAULT_CFG), solved: false, bestCost: null },
  };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return fresh;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of LAB_BRIEFS) {
      const e = parsed[b.id];
      if (!e || !Array.isArray(e.cfg) || e.cfg.length !== 4 || !e.cfg.every(isValidStage)) continue;
      fresh[b.id] = {
        cfg: cloneCfg(e.cfg as PipelineConfig),
        solved: !!e.solved,
        bestCost: typeof e.bestCost === 'number' ? e.bestCost : null,
      };
    }
  } catch {
    // ignore storage errors
  }
  return fresh;
}

export function AgentPipelineLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [showOpt, setShowOpt] = useState(false);

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const ev = useMemo(() => evaluatePipeline(entry.cfg), [entry.cfg]);
  const pass = briefPasses(brief, ev);
  const costRounded = Math.round(ev.expectedCost * 100) / 100;
  const ex = useMemo(() => (showOpt ? labExhaustive(brief) : null), [showOpt, brief]);

  const saved = useRef(lab);
  saved.current = lab;
  const persist = (next: LabState) => {
    setLab(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };

  const updateStage = (idx: number, patch: Partial<StageConfig>) => {
    const nextCfg = cloneCfg(entry.cfg);
    nextCfg[idx] = { ...nextCfg[idx], ...patch };
    if (!nextCfg[idx].check) nextCfg[idx].retries = 0;
    persist({ ...lab, [brief.id]: { ...entry, cfg: nextCfg } });
  };

  const applyPreset = (kind: 'allFast' | 'allStrong') => {
    const nextCfg: PipelineConfig = [0, 1, 2, 3].map(() => ({
      model: kind === 'allStrong' ? 'strong' : 'fast',
      check: false,
      retries: 0,
    })) as PipelineConfig;
    persist({ ...lab, [brief.id]: { ...entry, cfg: nextCfg } });
  };

  useEffect(() => {
    if (!pass) return;
    const cur = saved.current[brief.id];
    if (cur.solved && cur.bestCost !== null && cur.bestCost <= costRounded) return;
    persist({
      ...saved.current,
      [brief.id]: {
        ...cur,
        solved: true,
        bestCost: cur.bestCost === null ? costRounded : Math.min(cur.bestCost, costRounded),
      },
    });
  }, [pass, brief.id, costRounded]); // eslint-disable-line react-hooks/exhaustive-deps

  const okSuccess = ev.pTask >= brief.minSuccess - 1e-9;
  const okSilent = ev.pSilentAny <= brief.maxSilent + 1e-9;
  const okBadAct = brief.maxBadAction === undefined || ev.pBadAction <= brief.maxBadAction + 1e-9;

  return (
    <Widget
      title="Lab: Harden the Agent Pipeline"
      wide
      caption={
        <>
          Configure all 4 stages of the Kestrel support agent: choose Fast (1.0 cr) or Strong (3.0 cr), whether to run that stage’s check, and how many retries to allow when the check catches an error. Every probability and expected cost is computed exactly. Meet the brief’s reliability targets; optionally, minimise expected credits per ticket. Your work is saved in this browser.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'space-between', alignItems: 'center' }}>
        <Segmented
          value={briefIdx}
          onChange={(i) => {
            setBriefIdx(i);
            setShowOpt(false);
          }}
          options={LAB_BRIEFS.map((b, i) => ({
            value: i,
            label: `${lab[b.id].solved ? '✓ ' : ''}${b.title.split(' · ')[0]}`,
          }))}
        />
        <div className="btn-row">
          <button type="button" className="btn small" onClick={() => applyPreset('allFast')}>
            Preset: All Fast (4 cr)
          </button>
          <button type="button" className="btn small" onClick={() => applyPreset('allStrong')}>
            Preset: All Strong (12 cr)
          </button>
        </div>
      </div>

      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(360px, 100%), 1fr))', gap: 10 }}>
        {LAB_STAGES.map((st, idx) => {
          const c = entry.cfg[idx];
          const stat = ev.stageStats[idx];
          return (
            <div
              key={st.id}
              style={{
                padding: '10px 12px',
                borderRadius: 10,
                background: 'var(--bg-sunken)',
                border: '1px solid var(--border)',
                display: 'grid',
                gap: 8,
              }}
            >
              <div>
                <div style={{ fontWeight: 650, fontSize: 13.5 }}>{st.name}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{st.desc}</div>
              </div>

              <Segmented
                label="Model"
                value={c.model}
                onChange={(model) => updateStage(idx, { model })}
                options={[
                  { value: 'fast', label: `Fast (${(100 * st.pFast).toFixed(0)}%, 1cr)` },
                  { value: 'strong', label: `Strong (${(100 * st.pStrong).toFixed(0)}%, 3cr)` },
                ]}
              />

              <Segmented
                label={`${st.checkLabel} (+${st.checkCost}cr)`}
                value={c.check ? 1 : 0}
                onChange={(v) => updateStage(idx, { check: v === 1 })}
                options={[
                  { value: 0, label: 'Off' },
                  { value: 1, label: 'On' },
                ]}
              />

              <Segmented
                label="Retries on catch"
                value={c.check ? c.retries : 0}
                onChange={(retries) => updateStage(idx, { retries: retries as 0 | 1 | 2 })}
                options={
                  c.check
                    ? [
                        { value: 0, label: '0 (abort)' },
                        { value: 1, label: '1' },
                        { value: 2, label: '2' },
                      ]
                    : [{ value: 0, label: '0 (check off)' }]
                }
              />

              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                <span>step ok: {pct1(stat.pOk)}</span>
                <span>silent: {pct2(stat.pSilent)}</span>
                <span>{stat.expectedCost.toFixed(2)} cr</span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
        <div className="btn-row">
          <span className={`status ${pass ? 'good' : 'bad'}`}>{pass ? 'Meets the brief' : 'Not yet'}</span>
          {entry.solved && (
            <span className="status good">Solved{entry.bestCost !== null ? ` · cheapest so far: ${entry.bestCost.toFixed(2)} cr` : ''}</span>
          )}
        </div>

        <div className="readout">
          <span style={{ color: okSuccess ? 'var(--c-good)' : 'var(--c-bad)' }}>{okSuccess ? '✓' : '✗'}</span>{' '}
          <span className="k">end-to-end success </span>
          {pct1(ev.pTask)} (need ≥ {pct1(brief.minSuccess)})
          {'\n'}
          <span style={{ color: okSilent ? 'var(--c-good)' : 'var(--c-bad)' }}>{okSilent ? '✓' : '✗'}</span>{' '}
          <span className="k">silent failure (whole pipeline) </span>
          {pct1(ev.pSilentAny)} (need ≤ {pct1(brief.maxSilent)})
          {brief.maxBadAction !== undefined && (
            <>
              {'\n'}
              <span style={{ color: okBadAct ? 'var(--c-good)' : 'var(--c-bad)' }}>{okBadAct ? '✓' : '✗'}</span>{' '}
              <span className="k">Stage 4 silent bad action </span>
              {pct2(ev.pBadAction)} (need ≤ {pct2(brief.maxBadAction)})
            </>
          )}
          {'\n'}
          <span className="k">clean aborts (escalated safely) </span>
          {pct1(ev.pAbortAny)}
          {'\n'}
          <span className="k">expected cost per ticket </span>
          <strong>{costRounded.toFixed(2)} credits</strong>
        </div>

        <div className="live-note">
          {pass
            ? `Your pipeline passes at ${costRounded.toFixed(2)} credits per ticket. Optional: can you replace any Strong model with Fast + a cheap deterministic check, or trim a retry without breaking the brief?`
            : !okBadAct
              ? 'Stage 4 is still committing too many silent errors. Notice that a 95% dry-run guard lets 5% of Stage 4 mistakes through — lowering the base error rate there (Strong + guard) cuts silent bad actions dramatically.'
              : !okSilent
                ? 'Too many errors are slipping through silently. Any stage with its check turned off sends 100% of its mistakes downstream unnoticed.'
                : 'End-to-end success is still below target. Find the stage with the lowest “step ok” above: cheap deterministic checks + retries fix Lookup and Math; fuzzy Policy judgement needs a stronger base model.'}
        </div>

        {entry.solved && (
          <div>
            <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>
              {showOpt ? 'Hide' : 'Show'} the exhaustive search (all 4,096 pipelines)
            </button>
            {showOpt && ex && (
              <div className="live-note" style={{ marginTop: 10 }}>
                All {ex.total.toLocaleString()} combinations were evaluated exactly: <strong>{ex.passing}</strong> meet this brief. The cheapest costs{' '}
                <strong>{ex.bestCost.toFixed(2)} credits</strong>:{' '}
                {ex.best
                  .map((w) => w.cfg.map((c, i) => `${LAB_STAGES[i].shortName} ${formatStageConfig(c)}`).join(' · '))
                  .join('; or ')}{' '}
                ({pct1(ex.best[0].eval.pTask)} success, {pct1(ex.best[0].eval.pSilentAny)} silent).
              </div>
            )}
          </div>
        )}
      </div>
    </Widget>
  );
}
