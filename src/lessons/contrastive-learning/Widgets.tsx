// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  BATCH_OPTIONS,
  CLIP_SAMPLE_COS,
  CLIP_SAMPLE_LABELS,
  CONTRAST_LAB_BRIEFS,
  LAB_TAUS_B1,
  LAB_TAUS_B2,
  contrastLabExhaustive,
  evalContrastBrief1,
  evalContrastBrief2,
  evalContrastBrief3,
  hardNegativeFocus,
  trainCircleContrastive,
  type BatchOption,
  type ContrastBriefId,
} from './contrastive';

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
  solved: Record<ContrastBriefId, boolean>;
};

const CLAB_KEY = 'nl-contrast-lab-v1';
const CLAB_DEFAULT: CLabState = {
  b1Negs: false,
  b1Tau: 0.15,
  b2Tau: 1.0,
  b3Opt: 'simclr-64',
  solved: { circle: false, temp: false, batch: false },
};

function loadCLab(): CLabState {
  try {
    const raw = localStorage.getItem(CLAB_KEY);
    if (!raw) return CLAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...CLAB_DEFAULT, ...p, solved: { ...CLAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return CLAB_DEFAULT;
  }
}

export function ContrastiveLab() {
  const [briefId, setBriefId] = useState<ContrastBriefId>('circle');
  const [st, setSt] = useState<CLabState>(loadCLab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(CLAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalContrastBrief1(st.b1Negs, st.b1Tau), [st.b1Negs, st.b1Tau]);
  const r2 = useMemo(() => evalContrastBrief2(st.b2Tau), [st.b2Tau]);
  const r3 = useMemo(() => evalContrastBrief3(st.b3Opt), [st.b3Opt]);
  const exhaust = useMemo(() => contrastLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'circle' ? r1.pass : briefId === 'temp' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const brief = CONTRAST_LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Tune Contrastive Training"
      wide
      caption={
        <>
          Prevent collapse, tune temperature safely, and beat the batch-size VRAM wall with a MoCo queue — all three briefs exhaustively checked. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {CONTRAST_LAB_BRIEFS.map((b, i) => (
          <button
            key={b.id}
            type="button"
            className="btn"
            style={{
              borderColor: briefId === b.id ? 'var(--accent)' : 'var(--border)',
              background: briefId === b.id ? 'var(--bg-sunken)' : 'transparent',
              fontWeight: briefId === b.id ? 600 : 400,
            }}
            onClick={() => {
              setBriefId(b.id);
              setShowOpt(false);
            }}
          >
            {st.solved[b.id] ? '✓ ' : ''}Brief {i + 1}
          </button>
        ))}
      </div>

      <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{brief.title}</div>
      <p style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 0, marginBottom: 12, lineHeight: 1.5 }}>{brief.story}</p>

      {briefId === 'circle' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="In-batch negatives"
              value={st.b1Negs ? 'yes' : 'no'}
              onChange={(v) => setSt((p) => ({ ...p, b1Negs: v === 'yes' }))}
              options={[
                { value: 'no', label: 'Off (Alignment only)' },
                { value: 'yes', label: 'On (InfoNCE)' },
              ]}
            />
            <Segmented
              label="Temperature"
              value={st.b1Tau}
              onChange={(v) => setSt((p) => ({ ...p, b1Tau: v }))}
              options={LAB_TAUS_B1.map((t) => ({ value: t, label: `τ = ${t}` }))}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Top-1 (= 100%)</div>
              <div className="readout-value" style={{ color: r1.top1Acc >= 0.99 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r1.top1Acc)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Mean cos (≤ 0.05)</div>
              <div className="readout-value" style={{ color: r1.meanPairCos <= 0.05 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.meanPairCos.toFixed(3)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Uniformity</div>
              <div className="readout-value">{Math.abs(r1.uniformity) < 5e-4 ? '0.000' : r1.uniformity.toFixed(3)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'temp' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="Temperature"
            value={st.b2Tau}
            onChange={(v) => setSt((p) => ({ ...p, b2Tau: v }))}
            options={LAB_TAUS_B2.map((t) => ({ value: t, label: `${t}` }))}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Hard-neg share (≥ 80%)</div>
              <div className="readout-value" style={{ color: r2.hardShareOfNegs >= 0.8 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r2.hardShareOfNegs)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Logit scale 1/τ (≤ 25)</div>
              <div className="readout-value" style={{ color: r2.invTau <= 25 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.invTau.toFixed(0)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r2.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'batch' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
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
                onClick={() => setSt((p) => ({ ...p, b3Opt: o.value }))}
              >
                {o.label}
              </button>
            ))}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Hard-neg prob (≥ 95%)</div>
              <div className="readout-value" style={{ color: r3.prob >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(r3.prob)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Activation VRAM (≤ 8 GiB)</div>
              <div className="readout-value" style={{ color: r3.vramGiB <= 8 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.vramGiB.toFixed(2)} GiB</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r3.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r3.pass ? 'PASS' : 'FAIL'}</div>
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
        <button type="button" className="btn" onClick={() => setShowOpt((v) => !v)}>
          {showOpt ? 'Hide exhaustive check' : `Reveal exhaustive search (${exhaust.total} configs)`}
        </button>
        {showOpt && (
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            <strong>{exhaust.passing}/{exhaust.total}</strong> pass · {exhaust.summary}
          </div>
        )}
      </div>
    </Widget>
  );
}
