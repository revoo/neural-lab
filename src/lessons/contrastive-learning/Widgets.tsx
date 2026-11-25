// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
import {
  BATCH_OPTIONS,
  CLIP_SAMPLE_COS,
  CLIP_SAMPLE_LABELS,
  CONTRAST_LAB_BRIEFS,
  LAB_TAUS_B1,
  LAB_TAUS_B2,
  contrastLabExhaustive,
  contrastLabHint,
  dot,
  evalContrastBrief1,
  evalContrastBrief2,
  evalContrastBrief3,
  hardNegativeFocus,
  trainCircleContrastive,
  type BatchOption,
} from './contrastive';
import { contrastiveRealCode } from './realCode';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

const PAIR_COLORS = [
  'var(--c-a)',
  'var(--c-b)',
  'var(--c-c)',
  'var(--accent)',
  'var(--c-good)',
  'var(--c-bad)',
  'var(--text)',
  'var(--text-muted)',
];

// ------------------------------------------------------------------ 1. Temperature & Hard-Negative Focus

export function TemperatureFocusWidget() {
  const [tau, setTau] = useState<number>(0.15);
  const res = useMemo(() => hardNegativeFocus(tau), [tau]);

  return (
    <Widget
      title="Inside InfoNCE: temperature τ controls hard-negative mining"
      wide
      caption={
        <>
          Query image: <strong>cat</strong>. True caption cosine = <code>0.85</code>; hard negative (<strong>dog</strong>) = <code>0.70</code>; easy negatives (<strong>car, espresso</strong>) = <code>0.10, −0.05</code>. Slide <strong>Temperature τ</strong> down from <code>1.00</code> to <code>0.05</code>: the hard negative’s share of repulsive gradient surges from <strong>49.5% → 100.0%</strong>!
        </>
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Slider label="Temperature τ" value={tau} min={0.04} max={1.0} step={0.02} onChange={setTau} format={(v) => `τ = ${v.toFixed(2)} (scale 1/τ = ${(1 / v).toFixed(1)})`} />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
        {CLIP_SAMPLE_LABELS.map((label, i) => {
          const p = res.probs[i];
          const isPos = i === 0;
          const isHard = i === 1;
          return (
            <div key={label} style={{ display: 'grid', gridTemplateColumns: 'minmax(150px, 210px) 1fr 68px', gap: 10, alignItems: 'center', fontSize: 13 }}>
              <div>
                <span style={{ fontWeight: isPos || isHard ? 600 : 400 }}>{label}</span>{' '}
                <span style={{ color: 'var(--text-muted)', fontSize: 11.5 }}>({CLIP_SAMPLE_COS[i].toFixed(2)})</span>
              </div>
              <div style={{ height: 14, borderRadius: 7, background: 'var(--bg-sunken)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${(100 * p).toFixed(1)}%`,
                    height: '100%',
                    background: isPos ? 'var(--c-good)' : isHard ? 'var(--c-bad)' : 'var(--text-muted)',
                    transition: 'width 60ms linear',
                  }}
                />
              </div>
              <div style={{ fontFamily: 'var(--mono)', textAlign: 'right', fontWeight: 600 }}>{pct1(p)}</div>
            </div>
          );
        })}
      </div>

      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
        <div className="readout">
          <div className="readout-label">True-match prob</div>
          <div className="readout-value" style={{ color: 'var(--c-good)' }}>{pct1(res.posProb)}</div>
        </div>
        <div className="readout">
          <div className="readout-label">Hard-neg share of push</div>
          <div className="readout-value" style={{ color: 'var(--c-b)' }}>{pct1(res.hardShareOfNegs)}</div>
        </div>
        <div className="readout">
          <div className="readout-label">InfoNCE loss</div>
          <div className="readout-value">{res.loss.toFixed(3)}</div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Alignment & Uniformity on the Unit Circle

export function CircleUniformityWidget() {
  const [regime, setRegime] = useState<'collapse' | 'hot' | 'sweet'>('sweet');
  const [steps, setSteps] = useState<number>(90);

  const negatives = regime !== 'collapse';
  const tau = regime === 'hot' ? 2.0 : 0.15;
  const run = useMemo(() => trainCircleContrastive(negatives, tau, steps, 61), [negatives, tau, steps]);

  return (
    <Widget
      title="Alignment & Uniformity on the unit circle: collapse vs regular octagon"
      wide
      caption={
        <>
          Eight image (filled circle) and text (hollow ring) pairs start crammed inside an 80° arc. Drag <strong>Steps</strong> from <code>0 → 90</code>. Without negatives (<strong>Alignment only</strong>), all 8 pairs collapse onto a single dot (<code>mean cos = +1.000</code>)! With <strong>InfoNCE (τ = 0.15)</strong>, every positive pair snaps together while the 8 classes fan evenly around 360° (<code>mean cos = −0.136</code>, near the exact regular-octagon floor <code>−1/7 = −0.143</code>)!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Training objective"
          value={regime}
          onChange={setRegime}
          options={[
            { value: 'collapse', label: 'Alignment only (collapse!)' },
            { value: 'hot', label: 'InfoNCE τ = 2.0 (weak)' },
            { value: 'sweet', label: 'InfoNCE τ = 0.15 (uniform)' },
          ]}
        />
        <div style={{ flex: '1 1 180px' }}>
          <Slider label="SGD steps" value={steps} min={0} max={90} step={10} onChange={setSteps} format={(v) => `${v} steps`} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 14, alignItems: 'center' }}>
        <div style={{ width: '100%', maxWidth: 280, margin: '0 auto' }}>
          <Plot x={[-1.35, 1.35]} y={[-1.35, 1.35]} width={280} height={280} xLabel="z₁" yLabel="z₂">
            {({ x: sx, y: sy }) => (
              <>
                <ellipse
                  cx={sx(0)}
                  cy={sy(0)}
                  rx={Math.abs(sx(1) - sx(0))}
                  ry={Math.abs(sy(0) - sy(1))}
                  fill="none"
                  stroke="var(--border)"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
                {run.A.map((a, i) => {
                  const b = run.B[i];
                  const col = PAIR_COLORS[i % PAIR_COLORS.length];
                  return (
                    <g key={i}>
                      <line x1={sx(a[0])} y1={sy(a[1])} x2={sx(b[0])} y2={sy(b[1])} stroke={col} strokeWidth={1.5} opacity={0.6} />
                      <circle cx={sx(a[0])} cy={sy(a[1])} r={5.5} fill={col} />
                      <circle cx={sx(b[0])} cy={sy(b[1])} r={7} fill="none" stroke={col} strokeWidth={2} />
                    </g>
                  );
                })}
              </>
            )}
          </Plot>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Alignment (↓ 0)</div>
              <div className="readout-value" style={{ color: run.alignment <= 0.05 ? 'var(--c-good)' : 'var(--c-b)' }}>{run.alignment.toFixed(4)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Uniformity (↓)</div>
              <div className="readout-value" style={{ color: run.uniformity < -2.0 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {Math.abs(run.uniformity) < 5e-4 ? '0.000' : run.uniformity.toFixed(3)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Mean pair cos</div>
              <div className="readout-value" style={{ color: run.meanPairCos <= 0.05 ? 'var(--c-good)' : 'var(--c-bad)' }}>{run.meanPairCos.toFixed(3)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Zero-shot Top-1</div>
              <div className="readout-value" style={{ color: run.top1Acc >= 0.99 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(run.top1Acc)}</div>
            </div>
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {regime === 'collapse'
              ? 'Total collapse: alignment looks “perfect” (0.0000), yet every single concept shares the same coordinates (mean cos = +1.000)!'
              : regime === 'hot'
                ? 'High temperature weakens repulsion: points stay bunched on one hemisphere (uniformity −1.52, mean cos +0.061).'
                : 'Balanced alignment + uniformity: image circles sit dead-centre inside their matching text rings, fanned cleanly around the entire circle.'}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Lab: Tune Contrastive Training

type CLabState = {
  b1Negs: boolean;
  b1Tau: number;
  b2Tau: number;
  b3Opt: BatchOption;
};

const CLAB_KEY = 'nl-contrast-lab-v1';
const CLAB_PROG_KEY = 'nl-contrast-lab-progress-v2';
const CLAB_DEFAULT: CLabState = {
  b1Negs: false,
  b1Tau: 0.15,
  b2Tau: 1.0,
  b3Opt: 'simclr-64',
};

function loadLegacyCSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(CLAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved ?? {};
  } catch {
    return {};
  }
}

function loadCLab(): CLabState {
  try {
    const raw = localStorage.getItem(CLAB_KEY);
    if (!raw) return CLAB_DEFAULT;
    const p = JSON.parse(raw);
    return {
      b1Negs: typeof p.b1Negs === 'boolean' ? p.b1Negs : CLAB_DEFAULT.b1Negs,
      b1Tau: (LAB_TAUS_B1 as readonly number[]).includes(p.b1Tau) ? p.b1Tau : CLAB_DEFAULT.b1Tau,
      b2Tau: (LAB_TAUS_B2 as readonly number[]).includes(p.b2Tau) ? p.b2Tau : CLAB_DEFAULT.b2Tau,
      b3Opt: BATCH_OPTIONS.some((o) => o.value === p.b3Opt) ? p.b3Opt : CLAB_DEFAULT.b3Opt,
    };
  } catch {
    return CLAB_DEFAULT;
  }
}

function cosCellBg(v: number): string {
  const t = Math.min(1, Math.abs(v));
  if (t < 0.05) return 'var(--bg)';
  const pct = Math.round(14 + 66 * t);
  const col = v > 0 ? 'var(--c-a)' : 'var(--c-b)';
  return `color-mix(in srgb, ${col} ${pct}%, var(--bg))`;
}

export function ContrastiveLab() {
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const briefId = CONTRAST_LAB_BRIEFS[activeIdx].id;
  const [st, setSt] = useState<CLabState>(loadCLab);
  const progress = useLabProgress(CLAB_PROG_KEY, () => ({ solved: loadLegacyCSolved() }));
  const log = useMachineLog(18);

  useEffect(() => {
    try {
      localStorage.setItem(CLAB_KEY, JSON.stringify({ ...st, solved: progress.solved }));
    } catch {
      /* ignore */
    }
  }, [st, progress.solved]);

  const r1 = useMemo(() => evalContrastBrief1(st.b1Negs, st.b1Tau), [st.b1Negs, st.b1Tau]);
  const r2 = useMemo(() => evalContrastBrief2(st.b2Tau), [st.b2Tau]);
  const r3 = useMemo(() => evalContrastBrief3(st.b3Opt), [st.b3Opt]);
  const exhaust = useMemo(() => contrastLabExhaustive(briefId), [briefId]);
  const hint = useMemo(() => contrastLabHint(briefId, st), [briefId, st]);
  const realCode = useMemo(() => contrastiveRealCode(briefId, st), [briefId, st]);

  const allB1 = useMemo(() => {
    const out: ReturnType<typeof evalContrastBrief1>[] = [];
    for (const negs of [false, true]) for (const tau of LAB_TAUS_B1) out.push(evalContrastBrief1(negs, tau));
    return out;
  }, []);
  const allB2 = useMemo(() => LAB_TAUS_B2.map((t) => evalContrastBrief2(t)), []);
  const allB3 = useMemo(() => BATCH_OPTIONS.map((o) => evalContrastBrief3(o.value)), []);

  const curPass = briefId === 'circle' ? r1.pass : briefId === 'temp' ? r2.pass : r3.pass;
  const curOptimal =
    briefId === 'circle'
      ? r1.pass && Math.abs(st.b1Tau - 0.15) < 1e-6
      : briefId === 'temp'
        ? r2.pass && r2.invTau === 20
        : r3.pass && r3.vramGiB < 0.7;

  const tweenedProbs2 = useTweenedArray(r2.probs, 240);
  const tweenedB3 = useTweenedArray([r3.prob, r3.vramGiB], 240);

  const updateB1 = (patch: Partial<Pick<CLabState, 'b1Negs' | 'b1Tau'>>) => {
    const negs = patch.b1Negs ?? st.b1Negs;
    const tau = patch.b1Tau ?? st.b1Tau;
    const ev = evalContrastBrief1(negs, tau);
    setSt((p) => ({ ...p, b1Negs: negs, b1Tau: tau }));
    log.push(
      `B1: ${negs ? `InfoNCE τ=${tau}` : 'Alignment only'} → Top-1 ${pct1(ev.top1Acc)}, meanCos ${ev.meanPairCos.toFixed(3)}, unif ${ev.uniformity.toFixed(2)} (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : 'bad',
    );
  };

  const updateB2 = (b2Tau: number) => {
    const ev = evalContrastBrief2(b2Tau);
    setSt((p) => ({ ...p, b2Tau }));
    log.push(
      `B2: τ=${b2Tau} (1/τ=${ev.invTau.toFixed(0)}) → hard-neg share ${pct1(ev.hardShareOfNegs)} (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const updateB3 = (b3Opt: BatchOption) => {
    const ev = evalContrastBrief3(b3Opt);
    setSt((p) => ({ ...p, b3Opt }));
    log.push(
      `B3: ${ev.label} → ${pct1(ev.prob)} coverage, ${ev.vramGiB.toFixed(2)} GiB VRAM (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const resetBrief = () => {
    if (briefId === 'circle') {
      setSt((p) => ({ ...p, b1Negs: CLAB_DEFAULT.b1Negs, b1Tau: CLAB_DEFAULT.b1Tau }));
      log.push('B1: reset to Alignment only');
    } else if (briefId === 'temp') {
      setSt((p) => ({ ...p, b2Tau: CLAB_DEFAULT.b2Tau }));
      log.push('B2: reset to τ=1.0');
    } else {
      setSt((p) => ({ ...p, b3Opt: CLAB_DEFAULT.b3Opt }));
      log.push('B3: reset to SimCLR B=64');
    }
  };

  const statusText = useMemo(() => {
    if (briefId === 'circle') {
      return r1.pass
        ? `Top-1 ${pct1(r1.top1Acc)} · align ${r1.alignment.toFixed(4)} · mean cos ${r1.meanPairCos.toFixed(3)} · uniformity ${r1.uniformity.toFixed(3)}.`
        : `Top-1 ${pct1(r1.top1Acc)} (= 100%) · align ${r1.alignment.toFixed(4)} (≤ 0.05) · mean cos ${r1.meanPairCos.toFixed(3)} (≤ 0.05).`;
    }
    if (briefId === 'temp') {
      return r2.pass
        ? `Hard-negative share ${pct1(r2.hardShareOfNegs)} (≥ 80%) · logit scale 1/τ = ${r2.invTau.toFixed(0)} (≤ 25).`
        : `Hard-negative share ${pct1(r2.hardShareOfNegs)} (need ≥ 80%) · logit scale 1/τ = ${r2.invTau.toFixed(0)} (limit 25).`;
    }
    return r3.pass
      ? `Coverage ${pct1(r3.prob)} (≥ 95%) · activation VRAM ${r3.vramGiB.toFixed(2)} GiB (≤ 8.0 GiB).`
      : `Coverage ${pct1(r3.prob)} (need ≥ 95%) · activation VRAM ${r3.vramGiB.toFixed(2)} GiB (limit 8.0 GiB).`;
  }, [briefId, r1, r2, r3]);

  return (
    <LabShell
      title="Lab: Tune Contrastive Training"
      caption={
        <>
          Watch 8 image–text pairs collapse or spread across the unit circle, focus repulsion onto hard negatives safely, and bypass the batch-size VRAM wall with a MoCo queue.
        </>
      }
      briefs={CONTRAST_LAB_BRIEFS.map((b, i) => ({
        id: b.id,
        tab: `Brief ${i + 1}`,
        title: b.title,
        text: b.story,
      }))}
      active={activeIdx}
      onSelect={setActiveIdx}
      progress={progress}
      passing={curPass}
      optimal={curOptimal}
      status={statusText}
      hint={hint}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
      log={{
        title: 'Contrastive bench log',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Toggle negatives, temperature, or batch/queue setup to log runs.',
      }}
      reveal={{
        label: `exhaustive search (${exhaust.total} configs)`,
        content: (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 13 }}>
              <strong>
                {exhaust.passing}/{exhaust.total}
              </strong>{' '}
              pass · {exhaust.summary}
            </div>

            {briefId === 'circle' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Objective</th>
                    <th>Top-1</th>
                    <th>Mean cos</th>
                    <th>Uniformity</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {allB1.map((row) => (
                    <tr
                      key={`${row.negatives}-${row.tau}`}
                      style={{ fontWeight: row.negatives === st.b1Negs && row.tau === st.b1Tau ? 600 : 400 }}
                    >
                      <td>{row.negatives ? `InfoNCE τ=${row.tau}` : `Align-only (τ=${row.tau})`}</td>
                      <td>{pct1(row.top1Acc)}</td>
                      <td>{row.meanPairCos.toFixed(3)}</td>
                      <td>{row.uniformity.toFixed(3)}</td>
                      <td style={{ color: row.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{row.pass ? 'PASS' : 'FAIL'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {briefId === 'temp' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>τ</th>
                    <th>1/τ</th>
                    <th>Hard-neg share</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {allB2.map((row) => (
                    <tr key={row.tau} style={{ fontWeight: row.tau === st.b2Tau ? 600 : 400 }}>
                      <td>{row.tau}</td>
                      <td>{row.invTau.toFixed(0)}</td>
                      <td>{pct1(row.hardShareOfNegs)}</td>
                      <td style={{ color: row.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{row.pass ? 'PASS' : 'FAIL'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {briefId === 'batch' && (
              <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Negatives</th>
                    <th>Coverage</th>
                    <th>VRAM</th>
                    <th>Pass</th>
                  </tr>
                </thead>
                <tbody>
                  {allB3.map((row) => (
                    <tr key={row.value} style={{ fontWeight: row.value === st.b3Opt ? 600 : 400 }}>
                      <td>{row.label}</td>
                      <td>{row.totalNegs.toLocaleString()}</td>
                      <td>{pct1(row.prob)}</td>
                      <td>{row.vramGiB.toFixed(2)} GiB</td>
                      <td style={{ color: row.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{row.pass ? 'PASS' : 'FAIL'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ),
      }}
    >
      {briefId === 'circle' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="In-batch negatives"
              value={st.b1Negs ? 'yes' : 'no'}
              onChange={(v) => updateB1({ b1Negs: v === 'yes' })}
              options={[
                { value: 'no', label: 'Off (Alignment only)' },
                { value: 'yes', label: 'On (InfoNCE)' },
              ]}
            />
            <Segmented
              label="Temperature"
              value={st.b1Tau}
              onChange={(v) => updateB1({ b1Tau: v })}
              options={LAB_TAUS_B1.map((t) => ({ value: t, label: `τ = ${t}` }))}
            />
          </div>

          {/* Live unit-circle wheel + 8x8 cosine similarity matrix */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center', padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
            <div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
                Unit circle after 90 steps (● image, ○ caption)
              </div>
              <svg viewBox="0 0 180 180" style={{ width: 170, height: 170, display: 'block', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg)' }}>
                <circle cx={90} cy={90} r={68} fill="none" stroke="var(--border)" strokeWidth={1.2} strokeDasharray="3 3" />
                {r1.A.map((a, i) => {
                  const b = r1.B[i];
                  const ax = 90 + a[0] * 68;
                  const ay = 90 - a[1] * 68;
                  const bx = 90 + b[0] * 68;
                  const by = 90 - b[1] * 68;
                  const col = PAIR_COLORS[i % PAIR_COLORS.length];
                  return (
                    <g key={i}>
                      <line x1={ax} y1={ay} x2={bx} y2={by} stroke={col} strokeOpacity={0.5} strokeWidth={1.4} />
                      <circle cx={bx} cy={by} r={6.2} fill="none" stroke={col} strokeWidth={1.8} />
                      <circle cx={ax} cy={ay} r={3.6} fill={col} />
                    </g>
                  );
                })}
              </svg>
            </div>

            <div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
                8×8 cosine matrix A · Bᵀ (healthy = bright diagonal)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 20px)', gap: 2, padding: 3, borderRadius: 6, background: 'var(--border)', width: 'fit-content' }}>
                {r1.A.map((a, i) =>
                  r1.B.map((b, j) => {
                    const sim = dot(a, b);
                    return (
                      <div
                        key={`${i}-${j}`}
                        title={`cos(img_${i}, txt_${j}) = ${sim.toFixed(2)}`}
                        style={{
                          width: 20,
                          height: 20,
                          background: cosCellBg(sim),
                          borderRadius: 2,
                          outline: i === j ? '1px solid var(--text)' : undefined,
                          outlineOffset: -1,
                        }}
                      />
                    );
                  }),
                )}
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 6 }}>
                Top-1: <strong style={{ color: r1.top1Acc >= 0.99 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r1.top1Acc)}</strong> · mean cos{' '}
                <strong style={{ color: r1.meanPairCos <= 0.05 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.meanPairCos.toFixed(3)}</strong> · unif{' '}
                <strong>{Math.abs(r1.uniformity) < 5e-4 ? '0.000' : r1.uniformity.toFixed(3)}</strong>
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'temp' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <Segmented
            label="Temperature"
            value={st.b2Tau}
            onChange={updateB2}
            options={LAB_TAUS_B2.map((t) => ({ value: t, label: `${t}` }))}
          />

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12, padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                4-way softmax weights at τ = {st.b2Tau} (logit = cos / τ)
              </div>
              {CLIP_SAMPLE_LABELS.map((label, i) => {
                const p = tweenedProbs2[i];
                const col = i === 0 ? 'var(--c-good)' : i === 1 ? 'var(--c-b)' : 'var(--c-a)';
                return (
                  <div key={label} style={{ fontSize: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <span>
                        {label} <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--mono)', fontSize: 11 }}>({CLIP_SAMPLE_COS[i].toFixed(2)})</span>
                      </span>
                      <strong style={{ fontFamily: 'var(--mono)', color: col }}>{pct1(r2.probs[i])}</strong>
                    </div>
                    <div style={{ height: 8, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${Math.max(1, p * 100)}%`, background: col }} />
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 10 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>Hard-negative share of repulsion (≥ 80%)</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: r2.hardShareOfNegs >= 0.8 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {pct1(r2.hardShareOfNegs)}
                  </strong>
                </div>
                <div style={{ position: 'relative', height: 11, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${r2.hardShareOfNegs * 100}%`, background: r2.hardShareOfNegs >= 0.8 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: '80%', width: 2, background: 'var(--text)' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>Logit scale 1/τ (FP16 ceiling ≤ 25)</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: r2.invTau <= 25 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {r2.invTau.toFixed(0)} / 25
                  </strong>
                </div>
                <div style={{ position: 'relative', height: 11, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.min(100, (r2.invTau / 50) * 100)}%`, background: r2.invTau <= 25 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: '50%', width: 2, background: 'var(--text)' }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'batch' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {BATCH_OPTIONS.map((o) => (
              <button
                key={o.value}
                type="button"
                className="btn"
                style={{
                  fontSize: 12.5,
                  padding: '5px 10px',
                  borderColor: st.b3Opt === o.value ? 'var(--accent)' : 'var(--border)',
                  background: st.b3Opt === o.value ? 'var(--bg-sunken)' : 'transparent',
                  fontWeight: st.b3Opt === o.value ? 600 : 400,
                }}
                onClick={() => updateB3(o.value)}
              >
                {o.label}
              </button>
            ))}
          </div>

          <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 12 }}>
              <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                Active batch: {r3.activeB} × 8 MiB = <strong>{((r3.activeB * 8) / 1024).toFixed(2)} GiB</strong>
              </span>
              {r3.value.startsWith('moco-') && (
                <>
                  <span>+</span>
                  <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--c-good)' }}>
                    FIFO queue: {r3.totalNegs} × 0.25 MiB = <strong>{((r3.totalNegs * 0.25) / 1024).toFixed(2)} GiB</strong>
                  </span>
                </>
              )}
              <span>→ {r3.totalNegs.toLocaleString()} negatives/step</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>Hard-negative coverage (≥ 95%)</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: r3.prob >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {pct1(r3.prob)}
                  </strong>
                </div>
                <div style={{ position: 'relative', height: 11, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.min(100, tweenedB3[0] * 100)}%`, background: r3.prob >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: '95%', width: 2, background: 'var(--text)' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span>Activation VRAM (ceiling = 8.0 GiB)</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: r3.vramGiB <= 8 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {tweenedB3[1].toFixed(2)} / 8.00 GiB
                  </strong>
                </div>
                <div style={{ position: 'relative', height: 11, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.min(100, (tweenedB3[1] / 16) * 100)}%`, background: r3.vramGiB <= 8 ? 'var(--c-good)' : 'var(--c-bad)' }} />
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: '50%', width: 2, background: 'var(--text)' }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <LiveCode note="PyTorch InfoNCE / CLIP / MoCo" code={realCode} />
    </LabShell>
  );
}
