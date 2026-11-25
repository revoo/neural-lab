// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import {
  ALIGNMENT_BRIEFS,
  CANDIDATE_REPLIES,
  LAB_BETAS,
  LAB_STEPS,
  evaluateAlignmentBrief,
  evaluateDpoPoint,
  exactRlhfOptimum,
  runLiveAlignment,
  sigmoid,
  solveAlignmentBrief,
  traceGoodhartCurve,
  type AlignmentLabConfig,
  type LabMethod,
} from './dpo';

// ------------------------------------------------------------------ 1. KL Leash & Goodhart Over-Optimisation Explorer

const BETA_SLIDER_STOPS = [10, 6, 4, 2.5, 1.8, 1.4, 1.1, 0.9, 0.7, 0.5, 0.38, 0.28, 0.2, 0.14, 0.1];

export function RewardHackingExplorer() {
  const [stopIdx, setStopIdx] = useState(6); // default β = 1.1 (peak!)
  const beta = BETA_SLIDER_STOPS[stopIdx];

  const curve = useMemo(() => traceGoodhartCurve(), []);
  const cur = useMemo(() => exactRlhfOptimum(beta), [beta]);

  // SVG Goodhart chart: x = KL (0..6.5 nats), y = expected reward (-3.2 .. +5.4)
  const W = 320;
  const H = 195;
  const pad = { l: 38, r: 14, t: 16, b: 30 };
  const xMin = 0;
  const xMax = 6.5;
  const yMin = -3.2;
  const yMax = 6.2;

  const sx = (kl: number) => pad.l + ((Math.min(xMax, Math.max(xMin, kl)) - xMin) / (xMax - xMin)) * (W - pad.l - pad.r);
  const sy = (r: number) => H - pad.b - ((Math.min(yMax, Math.max(yMin, r)) - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);

  const proxyPts = curve.pts.map((p) => `${sx(p.klNats).toFixed(1)},${sy(p.expectedProxy).toFixed(1)}`).join(' ');
  const truePts = curve.pts.map((p) => `${sx(p.klNats).toFixed(1)},${sy(p.expectedTrue).toFixed(1)}`).join(' ');

  const barColor = (kind: (typeof CANDIDATE_REPLIES)[number]['kind']) =>
    kind === 'great' ? 'var(--c-good)' : kind === 'good' ? 'var(--c-a)' : kind === 'hack' ? 'var(--c-bad)' : 'var(--text-faint)';

  return (
    <Widget
      title="The KL leash & Goodhart’s law: when optimising harder makes the AI worse"
      wide
      caption={
        <>
          Exact closed-form KL-regularised RLHF optimum <code>π*(y) ∝ π_ref(y)·exp(r̂(y)/β)</code> over 8 replies. Drag <strong>β</strong> down from <strong>4.0</strong> to <strong>0.10</strong>: proxy reward climbs smoothly to <strong>+5.0</strong> while true human helpfulness peaks at <strong>β = 1.1 (+2.36)</strong> and then crashes to <strong>−2.80</strong> as the policy hacks the glitches!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ flex: '1 1 220px' }}>
          <Slider
            label={`KL penalty β = ${beta.toFixed(2)} (left = tight leash, right = no leash)`}
            min={0}
            max={BETA_SLIDER_STOPS.length - 1}
            step={1}
            value={stopIdx}
            onChange={setStopIdx}
            format={(i) => `β = ${BETA_SLIDER_STOPS[i]}`}
          />
        </div>
        <div className="btn-row">
          <button className="btn" onClick={() => setStopIdx(2)}>Tight (4.0)</button>
          <button className="btn primary" onClick={() => setStopIdx(6)}>Sweet spot (1.1)</button>
          <button className="btn" onClick={() => setStopIdx(9)}>Slipping (0.5)</button>
          <button className="btn" onClick={() => setStopIdx(14)}>Hacked (0.1)</button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(255px, 1fr))', gap: 12, marginBottom: 12 }}>
        {/* Left: Goodhart Curve */}
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>Proxy vs true reward across KL divergence</div>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {[-2, 0, 2, 4].map((v) => (
              <g key={v}>
                <line x1={pad.l} y1={sy(v)} x2={W - pad.r} y2={sy(v)} stroke={v === 0 ? 'var(--border-strong)' : 'var(--border)'} strokeDasharray={v === 0 ? undefined : '3 3'} />
                <text x={pad.l - 5} y={sy(v) + 3} textAnchor="end" fill="var(--text-faint)" fontSize={9} fontFamily="var(--font-mono)">
                  {v > 0 ? `+${v}` : v}
                </text>
              </g>
            ))}
            {[0, 2, 4, 6].map((kl) => (
              <text key={kl} x={sx(kl)} y={H - 10} textAnchor="middle" fill="var(--text-faint)" fontSize={9} fontFamily="var(--font-mono)">
                {kl}
              </text>
            ))}
            <text x={(pad.l + W - pad.r) / 2} y={H - 1} textAnchor="middle" fill="var(--text-muted)" fontSize={9}>
              KL divergence D_KL(π ‖ π_ref) in nats →
            </text>

            {/* Current KL vertical marker */}
            <line x1={sx(cur.klNats)} y1={pad.t} x2={sx(cur.klNats)} y2={H - pad.b} stroke="var(--accent)" strokeDasharray="3 3" opacity={0.75} />

            {/* Proxy & True curves */}
            <polyline fill="none" stroke="var(--c-b)" strokeWidth={2} strokeDasharray="5 3" points={proxyPts} />
            <polyline fill="none" stroke="var(--c-a)" strokeWidth={2.6} points={truePts} />

            {/* Live dots */}
            <circle cx={sx(cur.klNats)} cy={sy(cur.expectedProxy)} r={4.5} fill="var(--c-b)" />
            <circle cx={sx(cur.klNats)} cy={sy(cur.expectedTrue)} r={5.5} fill={cur.hackProb > 0.25 ? 'var(--c-bad)' : 'var(--c-good)'} stroke="var(--bg)" strokeWidth={1.5} />

            {/* Legend */}
            <text x={pad.l + 6} y={pad.t + 8} fill="var(--c-b)" fontSize={9.5} fontFamily="var(--font-mono)">
              - - Proxy r̂ ({cur.expectedProxy.toFixed(2)})
            </text>
            <text x={W - pad.r - 4} y={pad.t + 8} textAnchor="end" fill="var(--c-a)" fontSize={9.5} fontWeight={700} fontFamily="var(--font-mono)">
              ── True R* ({cur.expectedTrue.toFixed(2)})
            </text>
          </svg>
        </div>

        {/* Right: 8-reply policy bars */}
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
            <strong>Policy π*(y) over 8 replies</strong>
            <span className="mono" style={{ fontSize: 11, color: cur.hackProb > 0.2 ? 'var(--c-bad)' : 'var(--c-good)' }}>
              Hack: {(cur.hackProb * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {CANDIDATE_REPLIES.map((r, i) => {
              const p = cur.probs[i];
              return (
                <div key={r.id} style={{ display: 'grid', gridTemplateColumns: '1fr 78px 42px', alignItems: 'center', gap: 6 }}>
                  <div style={{ fontSize: 11, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: r.kind === 'hack' ? 'var(--c-bad)' : 'var(--text)' }} title={r.snippet}>
                    {r.label}
                  </div>
                  <div style={{ height: 11, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, p * 100)}%`, height: '100%', background: barColor(r.kind), transition: 'width 150ms ease' }} />
                  </div>
                  <span className="mono" style={{ fontSize: 10.5, textAlign: 'right', color: 'var(--text-muted)' }}>
                    {(p * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div
        className="live-note"
        style={{
          borderColor: cur.hackProb > 0.25 ? 'var(--c-bad)' : cur.expectedTrue > 2.2 ? 'var(--c-good)' : undefined,
          background:
            cur.hackProb > 0.25
              ? 'color-mix(in srgb, var(--c-bad) 10%, var(--bg-sunken))'
              : cur.expectedTrue > 2.2
                ? 'color-mix(in srgb, var(--c-good) 10%, var(--bg-sunken))'
                : undefined,
        }}
      >
        {cur.hackProb > 0.25 ? (
          <>
            <strong style={{ color: 'var(--c-bad)' }}>Reward hacking!</strong> At β = {beta.toFixed(2)}, the policy wandered{' '}
            <strong>{cur.klNats.toFixed(2)} nats</strong> away from SFT to score <strong>+{cur.expectedProxy.toFixed(2)}</strong> on the proxy reward model — putting{' '}
            <strong>{(cur.hackProb * 100).toFixed(1)}%</strong> probability on loophole junk and crashing true helpfulness to{' '}
            <strong>{cur.expectedTrue.toFixed(2)}</strong>!
          </>
        ) : cur.expectedTrue > 2.2 ? (
          <>
            <strong style={{ color: 'var(--c-good)' }}>Sweet spot (KL = {cur.klNats.toFixed(2)} nats):</strong> the KL leash keeps the policy close enough to SFT that rare glitches stay suppressed ({(cur.hackProb * 100).toFixed(1)}%), lifting true helpfulness from 1.34 up to{' '}
            <strong>+{cur.expectedTrue.toFixed(2)}</strong>!
          </>
        ) : (
          <>
            <strong>Tight KL leash (β = {beta.toFixed(2)}):</strong> safe (hack {(cur.hackProb * 100).toFixed(1)}%, KL {cur.klNats.toFixed(2)} nats), but leaves helpfulness on the table (+{cur.expectedTrue.toFixed(2)} vs peak +2.36 at β = 1.1).
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Inside the DPO Loss & Implicit Reward

export function DpoLossExplorer() {
  const [logRatioW, setLogRatioW] = useState(-1.0);
  const [logRatioL, setLogRatioL] = useState(1.0);
  const [beta, setBeta] = useState(0.5);

  const pt = useMemo(() => evaluateDpoPoint(logRatioW, logRatioL, beta), [logRatioW, logRatioL, beta]);

  // SVG plot of DPO loss & gradient weight vs margin Δ in [-5, +5]
  const W = 520;
  const H = 165;
  const pad = { l: 42, r: 16, t: 16, b: 28 };
  const mMin = -5;
  const mMax = 5;
  const yMax = 5.2;

  const sx = (m: number) => pad.l + ((Math.max(mMin, Math.min(mMax, m)) - mMin) / (mMax - mMin)) * (W - pad.l - pad.r);
  const syLoss = (v: number) => H - pad.b - (Math.min(yMax, Math.max(0, v)) / yMax) * (H - pad.t - pad.b);
  const syWeight = (w: number) => H - pad.b - w * (H - pad.t - pad.b); // 0..1 scaled onto full height

  const xs = Array.from({ length: 81 }, (_, i) => mMin + (i / 80) * (mMax - mMin));
  const lossPts = xs.map((m) => `${sx(m).toFixed(1)},${syLoss(Math.log1p(Math.exp(-m))).toFixed(1)}`).join(' ');
  const weightPts = xs.map((m) => `${sx(m).toFixed(1)},${syWeight(sigmoid(-m)).toFixed(1)}`).join(' ');

  const displacement = logRatioW < -0.1 && pt.margin > 0.8;

  return (
    <Widget
      title="Inside the DPO loss: implicit rewards & self-weighting gradients"
      wide
      caption={
        <>
          Drag the policy’s log-ratios for the chosen reply <code>y_w</code> and rejected reply <code>y_l</code> (or click <strong>Likelihood displacement!</strong>) to watch DPO’s implicit reward margin, loss, and self-weighting gradient factor <code>σ(−Δ)</code>.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 12 }}>
        <button className="btn" onClick={() => { setLogRatioW(-1.0); setLogRatioL(1.0); setBeta(0.5); }}>
          Model is wrong (Δ = −1.0)
        </button>
        <button className="btn primary" onClick={() => { setLogRatioW(1.5); setLogRatioL(-2.5); setBeta(0.5); }}>
          Well aligned (Δ = +2.0)
        </button>
        <button className="btn" onClick={() => { setLogRatioW(-1.2); setLogRatioL(-4.0); setBeta(1.0); }}>
          Likelihood displacement!
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(175px, 1fr))', gap: 12, marginBottom: 12 }}>
        <Slider
          label={`Chosen log-ratio ln(π/π_ref)`}
          min={-4}
          max={4}
          step={0.1}
          value={logRatioW}
          onChange={setLogRatioW}
          format={(v) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1))}
        />
        <Slider
          label={`Rejected log-ratio ln(π/π_ref)`}
          min={-4}
          max={4}
          step={0.1}
          value={logRatioL}
          onChange={setLogRatioL}
          format={(v) => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1))}
        />
        <Slider
          label={`KL scale β = ${beta.toFixed(2)}`}
          min={0.1}
          max={2.0}
          step={0.05}
          value={beta}
          onChange={setBeta}
          format={(v) => `β = ${v.toFixed(2)}`}
        />
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'var(--bg-sunken)', borderRadius: 8, border: '1px solid var(--border)', marginBottom: 12 }}>
        {[-4, -2, 0, 2, 4].map((m) => (
          <g key={m}>
            <line x1={sx(m)} y1={pad.t} x2={sx(m)} y2={H - pad.b} stroke={m === 0 ? 'var(--border-strong)' : 'var(--border)'} strokeDasharray={m === 0 ? undefined : '3 3'} />
            <text x={sx(m)} y={H - 8} textAnchor="middle" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
              {m > 0 ? `+${m}` : m}
            </text>
          </g>
        ))}
        <text x={W - pad.r - 4} y={pad.t + 10} textAnchor="end" fill="var(--c-a)" fontSize={10} fontWeight={700} fontFamily="var(--font-mono)">
          ── DPO loss −ln σ(Δ)
        </text>
        <text x={W - pad.r - 4} y={pad.t + 24} textAnchor="end" fill="var(--accent)" fontSize={10} fontWeight={700} fontFamily="var(--font-mono)">
          - - Grad weight σ(−Δ) ∈ (0, 1)
        </text>

        <polyline fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="5 3" points={weightPts} />
        <polyline fill="none" stroke="var(--c-a)" strokeWidth={2.4} points={lossPts} />

        {/* Live marker */}
        <line x1={sx(pt.margin)} y1={pad.t} x2={sx(pt.margin)} y2={H - pad.b} stroke="var(--c-b)" strokeDasharray="2 2" opacity={0.7} />
        <circle cx={sx(pt.margin)} cy={syLoss(pt.loss)} r={5.5} fill="var(--c-a)" stroke="var(--bg)" strokeWidth={1.5} />
        <circle cx={sx(pt.margin)} cy={syWeight(pt.gradWeight)} r={4.5} fill="var(--accent)" stroke="var(--bg)" strokeWidth={1.5} />
      </svg>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, marginBottom: 10 }}>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>IMPLICIT r_w / r_l</div>
          <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>
            {pt.implicitRw.toFixed(2)} / {pt.implicitRl.toFixed(2)}
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>MARGIN Δ</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: pt.margin >= 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {pt.margin >= 0 ? `+${pt.margin.toFixed(2)}` : pt.margin.toFixed(2)}
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>DPO LOSS</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--c-a)' }}>
            {pt.loss.toFixed(3)}
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>GRAD WEIGHT σ(−Δ)</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--accent)' }}>
            {pt.gradWeight.toFixed(3)}
          </div>
        </div>
      </div>

      <div className="live-note">
        {displacement ? (
          <>
            <strong style={{ color: 'var(--c-b)' }}>Likelihood displacement caught in the act!</strong> Notice that the chosen reply’s probability actually <strong>shrank</strong> (<code>ln(π/π_ref) = {logRatioW.toFixed(1)} &lt; 0</code>), yet DPO loss is tiny (<strong>{pt.loss.toFixed(3)}</strong>) simply because the rejected reply shrank even faster!
          </>
        ) : pt.margin < 0 ? (
          <>
            <strong>Wrong ranking (Δ = {pt.margin.toFixed(2)}):</strong> the policy currently prefers the rejected reply, so DPO’s gradient weight fires hard at <strong>{pt.gradWeight.toFixed(3)}</strong> — pushing <code>y_w</code> up and <code>y_l</code> down simultaneously.
          </>
        ) : (
          <>
            <strong>Aligned ranking (Δ = +{pt.margin.toFixed(2)}):</strong> the policy already prefers <code>y_w</code> over <code>y_l</code> ({(pt.winProb * 100).toFixed(1)}% implicit win probability), so the gradient weight gently dials back to <strong>{pt.gradWeight.toFixed(3)}</strong>.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Live SFT vs DPO Training Comparison

export function LiveDpoTrainer() {
  const [steps, setSteps] = useState(50);

  const sft = useMemo(() => runLiveAlignment({ mode: 'sft', beta: 0.5, steps }), [steps]);
  const dpo = useMemo(() => runLiveAlignment({ mode: 'dpo', beta: 0.5, steps }), [steps]);

  return (
    <Widget
      title="SFT on chosen only vs DPO: why rejecting losers matters"
      wide
      caption={
        <>
          Train the exact same SFT starting policy for <strong>{steps} steps</strong> using either <strong>SFT on chosen replies only</strong> (ignores losers) or <strong>DPO (β = 0.5)</strong> (pushes winners up AND explicitly pushes rejected hacks down).
        </>
      }
    >
      <div style={{ maxWidth: 360, marginBottom: 12 }}>
        <Slider
          label={`Training steps: ${steps}`}
          min={10}
          max={150}
          step={10}
          value={steps}
          onChange={setSteps}
          format={(s) => `${s} steps`}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(235px, 1fr))', gap: 12, marginBottom: 10 }}>
        {[
          { title: 'SFT (chosen replies only)', run: sft, accent: 'var(--c-b)' },
          { title: 'DPO (β = 0.5, winners + losers)', run: dpo, accent: 'var(--c-good)' },
        ].map(({ title, run, accent }) => (
          <div key={title} style={{ padding: 12, borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
              <strong style={{ fontSize: 13, color: accent }}>{title}</strong>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                KL {run.final.klNats.toFixed(2)} nats
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 8 }}>
              <div className="mono" style={{ fontSize: 12 }}>
                True R*: <strong style={{ color: accent }}>+{run.final.expectedTrue.toFixed(3)}</strong>
              </div>
              <div className="mono" style={{ fontSize: 12 }}>
                Hack prob: <strong style={{ color: run.final.hackProb > 0.01 ? 'var(--c-bad)' : 'var(--c-good)' }}>{(run.final.hackProb * 100).toFixed(2)}%</strong>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {CANDIDATE_REPLIES.slice(0, 8).map((r, i) => (
                <div key={r.id} style={{ display: 'grid', gridTemplateColumns: '1fr 70px 38px', alignItems: 'center', gap: 5 }}>
                  <span style={{ fontSize: 10.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: r.kind === 'hack' ? 'var(--c-bad)' : 'var(--text-muted)' }}>
                    {r.label}
                  </span>
                  <div style={{ height: 9, borderRadius: 3, background: 'var(--bg)', overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${Math.min(100, run.final.probs[i] * 100)}%`,
                        height: '100%',
                        background: r.kind === 'hack' ? 'var(--c-bad)' : accent,
                      }}
                    />
                  </div>
                  <span className="mono" style={{ fontSize: 10, textAlign: 'right', color: 'var(--text-faint)' }}>
                    {(run.final.probs[i] * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="live-note">
        At {steps} steps, <strong>DPO</strong> actively pushes rejected replies down — cutting hack probability to{' '}
        <strong>{(dpo.final.hackProb * 100).toFixed(2)}%</strong> (vs <strong>{(sft.final.hackProb * 100).toFixed(2)}%</strong> for SFT-only) and reaching higher true helpfulness (<strong>+{dpo.final.expectedTrue.toFixed(3)}</strong> vs <strong>+{sft.final.expectedTrue.toFixed(3)}</strong>).
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tune the Alignment Pipeline

const LAB_KEY = 'nl-dpo-lab-v1';

interface SavedAlignLab {
  configs: Record<string, AlignmentLabConfig>;
  solved: Record<string, boolean>;
}

function loadAlignLab(): SavedAlignLab {
  const defaults = Object.fromEntries(ALIGNMENT_BRIEFS.map((b) => [b.id, { ...b.defaultConfig }]));
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return { configs: defaults, solved: {} };
    const parsed = JSON.parse(raw);
    return {
      configs: { ...defaults, ...(parsed.configs ?? {}) },
      solved: parsed.solved ?? {},
    };
  } catch {
    return { configs: defaults, solved: {} };
  }
}

export function AlignmentLab() {
  const [briefIdx, setBriefIdx] = useState(0);
  const [state, setState] = useState<SavedAlignLab>(() => loadAlignLab());
  const [showOpt, setShowOpt] = useState(false);

  const brief = ALIGNMENT_BRIEFS[briefIdx];
  const cfg = state.configs[brief.id] ?? brief.defaultConfig;
  const ev = useMemo(() => evaluateAlignmentBrief(brief, cfg), [brief, cfg]);
  const sol = useMemo(() => solveAlignmentBrief(brief), [brief]);

  useEffect(() => {
    if (ev.passes && !state.solved[brief.id]) {
      setState((prev) => ({ ...prev, solved: { ...prev.solved, [brief.id]: true } }));
    }
  }, [ev.passes, brief.id, state.solved]);

  useEffect(() => {
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(state));
    } catch {
      /* ignore */
    }
  }, [state]);

  const update = (patch: Partial<AlignmentLabConfig>) => {
    setState((prev) => ({
      ...prev,
      configs: { ...prev.configs, [brief.id]: { ...cfg, ...patch } },
    }));
  };

  const effectiveMethod: LabMethod = brief.allowDpo ? cfg.method : 'rlhf-proxy';
  const solvedCount = ALIGNMENT_BRIEFS.filter((b) => state.solved[b.id]).length;
  const hitBest = ev.passes && ev.metrics.expectedTrue >= sol.bestEval.metrics.expectedTrue - 0.005;

  return (
    <Widget
      title="Lab: Tune the alignment pipeline without triggering reward hacking"
      wide
      caption={
        <>
          Three alignment briefs: tame a hackable proxy RM, beat SFT with clean DPO preference pairs, and survive noisy human raters who sometimes fall for sycophancy. Every legal grid setting is checked exhaustively.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="btn-row">
          {ALIGNMENT_BRIEFS.map((b, i) => (
            <button
              key={b.id}
              className={briefIdx === i ? 'btn primary' : 'btn'}
              onClick={() => {
                setBriefIdx(i);
                setShowOpt(false);
              }}
            >
              {state.solved[b.id] ? '✓ ' : ''}
              {i === 0 ? 'Brief 1 · Proxy RM' : i === 1 ? 'Brief 2 · Clean DPO' : 'Brief 3 · Noisy'}
            </button>
          ))}
        </div>
        <span className="mono" style={{ fontSize: 12, color: 'var(--text-faint)' }}>
          {solvedCount} / {ALIGNMENT_BRIEFS.length} briefs solved
        </span>
      </div>

      <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', marginBottom: 12, fontSize: 13 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ color: 'var(--text-muted)' }}>{brief.scenario}</div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        {brief.allowDpo ? (
          <Segmented
            label="alignment method"
            options={[
              { value: 'rlhf-proxy', label: 'RLHF Proxy' },
              { value: 'sft', label: 'SFT (chosen)' },
              { value: 'dpo', label: 'DPO' },
            ]}
            value={effectiveMethod}
            onChange={(v) => update({ method: v as LabMethod })}
          />
        ) : (
          <div>
            <div className="field-label">METHOD (LOCKED)</div>
            <div className="mono" style={{ padding: '6px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 12 }}>
              🔒 RLHF Proxy RM only
            </div>
          </div>
        )}

        {effectiveMethod !== 'sft' && (
          <Segmented
            label="KL penalty β"
            options={LAB_BETAS.map((b) => ({ value: b, label: `${b}` }))}
            value={cfg.beta}
            onChange={(b) => update({ beta: b })}
          />
        )}

        {effectiveMethod !== 'rlhf-proxy' && (
          <Segmented
            label="training steps"
            options={LAB_STEPS.map((s) => ({ value: s, label: `${s}` }))}
            value={cfg.steps}
            onChange={(s) => update({ steps: s })}
          />
        )}
      </div>

      {/* Checklist */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(185px, 1fr))', gap: 8, marginBottom: 12 }}>
        {[
          { ok: ev.checks.meetsTrue, label: `True R* ≥ ${brief.minTrueReward.toFixed(2)}`, val: ev.metrics.expectedTrue.toFixed(3) },
          { ok: ev.checks.meetsKl, label: `KL ≤ ${brief.maxKlNats.toFixed(2)} nats`, val: `${ev.metrics.klNats.toFixed(2)} nats` },
          { ok: ev.checks.meetsHack, label: `Hack ≤ ${(brief.maxHackProb * 100).toFixed(1)}%`, val: `${(ev.metrics.hackProb * 100).toFixed(2)}%` },
        ].map((c) => (
          <div
            key={c.label}
            className="mono"
            style={{
              padding: '8px 10px',
              borderRadius: 6,
              fontSize: 12,
              border: `1px solid color-mix(in srgb, ${c.ok ? 'var(--c-good)' : 'var(--c-bad)'} 45%, transparent)`,
              background: `color-mix(in srgb, ${c.ok ? 'var(--c-good)' : 'var(--c-bad)'} 10%, transparent)`,
              display: 'flex',
              justifyContent: 'space-between',
              gap: 6,
            }}
          >
            <span style={{ color: c.ok ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
              {c.ok ? '✓' : '✗'} {c.label}
            </span>
            <span>{c.val}</span>
          </div>
        ))}
      </div>

      <div
        className="live-note"
        style={{
          borderColor: ev.passes ? 'var(--c-good)' : undefined,
          background: ev.passes ? 'color-mix(in srgb, var(--c-good) 10%, var(--bg-sunken))' : undefined,
        }}
      >
        {ev.passes ? (
          <>
            <strong style={{ color: 'var(--c-good)' }}>
              {hitBest ? 'Brief solved — and you hit the exhaustive optimum!' : 'Brief solved!'}
            </strong>{' '}
            ({sol.passingCount} of {sol.totalConfigs} grid configs pass.)
            {!hitBest && (
              <>
                {' '}Optional goal: push True R* up to <strong>+{sol.bestEval.metrics.expectedTrue.toFixed(3)}</strong>.
              </>
            )}
          </>
        ) : (
          <>
            <strong>Not yet aligned.</strong>{' '}
            {!ev.checks.meetsHack
              ? 'Hack probability is too high — tighten β (larger KL leash) or switch to DPO so rejected hacks are explicitly pushed down.'
              : !ev.checks.meetsKl
                ? 'KL divergence wandered too far from the SFT reference — increase β or use fewer steps.'
                : 'True helpfulness is too low — loosen β slightly or train DPO longer.'}
          </>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn" onClick={() => update(brief.defaultConfig)}>Reset brief</button>
        <button className="btn" onClick={() => setShowOpt((s) => !s)}>
          {showOpt ? 'Hide exhaustive optimum' : 'Show exhaustive optimum'}
        </button>
      </div>

      {showOpt && (
        <div className="mono" style={{ marginTop: 8, padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 12 }}>
          Exhaustive search over all {sol.totalConfigs} grid configs ({sol.passingCount} pass): best is{' '}
          <strong>
            {sol.bestConfig.method.toUpperCase()}
            {sol.bestConfig.method !== 'sft' ? ` (β = ${sol.bestConfig.beta})` : ''}
            {sol.bestConfig.method !== 'rlhf-proxy' ? `, ${sol.bestConfig.steps} steps` : ''}
          </strong>{' '}
          → True R* = <strong>+{sol.bestEval.metrics.expectedTrue.toFixed(3)}</strong>, KL = {sol.bestEval.metrics.klNats.toFixed(2)} nats, Hack = {(sol.bestEval.metrics.hackProb * 100).toFixed(2)}%.
        </div>
      )}
    </Widget>
  );
}
