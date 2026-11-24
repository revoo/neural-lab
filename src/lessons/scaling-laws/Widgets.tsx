// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { LogChart } from '../../components/LogChart';
import { Plot } from '../../components/plot';
import {
  BUDGET_BRIEFS,
  CHINCHILLA,
  FRONTIER_RUNS,
  chinchillaOptimal,
  fmtCount,
  fmtFlops,
  h100Days,
  inferenceFlops,
  lifecycleOptimal,
  lossTerms,
  predictedLoss,
  tokensForLoss,
  trainFlops,
} from './scaling';

const f4 = (n: number) => n.toFixed(4);
const f2 = (n: number) => n.toFixed(2);

// ------------------------------------------------------------------ 1. Power-law explorer

const COMPUTE_CURVE = Array.from({ length: 61 }, (_, i) => {
  const C = 10 ** (19 + i * 0.1);
  const opt = chinchillaOptimal(C);
  return { C, ...opt };
});

export function PowerLawExplorer() {
  const [logC, setLogC] = useState(23.77); // ~Chinchilla 5.88e23
  const [view, setView] = useState<'excess' | 'total'>('excess');

  const C = 10 ** logC;
  const opt = useMemo(() => chinchillaOptimal(C), [C]);
  const days = h100Days(C);

  const series = useMemo(
    () => [
      {
        points: COMPUTE_CURVE.map((p): [number, number] => [p.C, view === 'excess' ? p.excess : p.total]),
        color: view === 'excess' ? 'var(--accent)' : 'var(--c-c)',
      },
    ],
    [view],
  );

  return (
    <Widget
      title="Power laws: straight lines on a log–log plot"
      wide
      caption={
        <>
          Compute-optimal predicted loss (Besiroglu et al. 2024 refit of Chinchilla’s 400+ runs) across six orders of magnitude of training compute. Toggle between <strong>Excess loss (L − E)</strong> — a razor-straight line on log–log axes — and <strong>Total loss L</strong>, which bends down toward the irreducible language entropy E = 1.817.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          options={[
            { value: 'excess', label: 'Excess loss (L − E)' },
            { value: 'total', label: 'Total loss (L)' },
          ]}
          value={view}
          onChange={setView}
        />
        <div className="btn-row">
          <button className="btn" onClick={() => setLogC(21)}>
            10²¹
          </button>
          <button className="btn" onClick={() => setLogC(Math.log10(3.15e23))}>
            GPT-3
          </button>
          <button className="btn primary" onClick={() => setLogC(Math.log10(5.88e23))}>
            Chinchilla
          </button>
          <button className="btn" onClick={() => setLogC(Math.log10(6.3e24))}>
            Llama 3 70B
          </button>
        </div>
      </div>

      <Slider
        label="training compute budget C (FLOPs)"
        value={logC}
        min={19}
        max={25}
        step={0.05}
        onChange={setLogC}
        format={(v) => fmtFlops(10 ** v)}
        color="accent"
      />

      <div className="widget-row" style={{ marginTop: 10 }}>
        <div className="widget-col" style={{ flex: '1.2 1 320px', minWidth: 0 }}>
          <LogChart
            series={series}
            xDomain={[1e19, 1e25]}
            yDomain={view === 'excess' ? [0.05, 2] : [1.5, 4]}
            logX
            xTicks={[1e19, 1e21, 1e23, 1e25]}
            xTickFormat={(t) => `10^${Math.round(Math.log10(t))}`}
            marker={C}
            xLabel="compute C (FLOPs)"
            ariaLabel="Scaling law loss against compute on a log-log plot"
            hlines={view === 'total' ? [{ y: CHINCHILLA.E, label: `irreducible floor E = ${CHINCHILLA.E.toFixed(2)}`, color: 'var(--c-b)' }] : []}
          />
        </div>

        <div className="widget-col" style={{ flex: '1 1 230px', minWidth: 0 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
            {[
              { label: 'optimal params N*', val: fmtCount(opt.N), color: 'var(--c-a)' },
              { label: 'optimal tokens D*', val: fmtCount(opt.D), color: 'var(--c-b)' },
              { label: 'predicted loss L*', val: f4(opt.total), sub: `PPL ${f2(opt.ppl)}`, color: 'var(--c-c)' },
              {
                label: 'H100 GPU-days',
                val: days >= 1000 ? `${(days / 1000).toFixed(1)}k` : days >= 10 ? `${Math.round(days)}` : days.toFixed(1),
                sub: 'at ~300 TF/s (30% MFU)',
                color: 'var(--accent-text)',
              },
            ].map((card) => (
              <div
                key={card.label}
                style={{
                  background: 'var(--bg-sunken)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 10px',
                }}
              >
                <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{card.label}</div>
                <div className="mono" style={{ fontSize: 17, fontWeight: 700, color: card.color, marginTop: 2 }}>
                  {card.val}
                </div>
                {card.sub && <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>{card.sub}</div>}
              </div>
            ))}
          </div>
          <div className="readout" style={{ marginTop: 8, fontSize: 12 }}>
            ratio D*/N* = <strong>{opt.ratio.toFixed(1)} tokens / param</strong>
            <br />
            excess above floor = <strong>{f4(opt.excess)}</strong> nats
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {view === 'excess' ? (
          <>
            <strong>A straight line on log–log axes = a power law.</strong> Every <strong>10× increase in compute</strong> multiplies optimal parameters by ~3.2× and optimal tokens by ~3.1×, cutting excess loss (L − E) by a steady ~34%. Notice how painful diminishing returns get: going from 10²³ to 10²⁴ FLOPs costs 10× more money for a smaller drop in raw loss.
          </>
        ) : (
          <>
            <strong>Approaching the irreducible floor E = {CHINCHILLA.E.toFixed(2)} (PPL ≈ {Math.exp(CHINCHILLA.E).toFixed(1)}):</strong> even an infinitely large model trained on infinite text cannot predict genuinely unpredictable choices in human language. At {fmtFlops(C)} FLOPs, optimal loss is {f4(opt.total)} — only {f4(opt.excess)} nats above the floor.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. IsoFLOP U-curve

const ISO_PRESETS = [
  { id: 'gpt3', label: 'GPT-3 budget (3.15×10²³)', C: 3.15e23, run: FRONTIER_RUNS[0] },
  { id: 'chinchilla', label: 'Chinchilla budget (5.88×10²³)', C: 5.88e23, run: FRONTIER_RUNS[2] },
  { id: 'llama3-70b', label: 'Llama 3 70B budget (6.3×10²⁴)', C: 6.3e24, run: FRONTIER_RUNS[4] },
];

export function IsoflopTradeoff() {
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = ISO_PRESETS[presetIdx];
  const [logN, setLogN] = useState(Math.log10(preset.run.N));

  const switchPreset = (i: number) => {
    setPresetIdx(i);
    setLogN(Math.log10(ISO_PRESETS[i].run.N));
  };

  const C = preset.C;
  const opt = useMemo(() => chinchillaOptimal(C), [C]);
  const minLogN = Math.log10(opt.N) - 1.1;
  const maxLogN = Math.log10(opt.N) + 1.1;
  const clampedLogN = Math.min(maxLogN, Math.max(minLogN, logN));
  const N = 10 ** clampedLogN;
  const D = C / (6 * N);
  const cur = lossTerms(N, D);
  const runTerms = lossTerms(preset.run.N, preset.run.D);

  const curve = useMemo(() => {
    const pts: [number, number][] = [];
    for (let i = 0; i <= 80; i++) {
      const ln = minLogN + (i / 80) * (maxLogN - minLogN);
      const n = 10 ** ln;
      pts.push([ln, predictedLoss(n, C / (6 * n))]);
    }
    return pts;
  }, [C, minLogN, maxLogN]);

  const yMin = Math.floor((opt.total - 0.03) * 20) / 20;
  const yMax = Math.ceil((opt.total + 0.16) * 20) / 20;
  const excessSum = cur.paramPenalty + cur.dataPenalty;

  return (
    <Widget
      title="The IsoFLOP U-curve: too big vs too small on one fixed budget"
      wide
      caption={
        <>
          Lock the training budget C and slide model size N — training tokens D = C / (6N) adjusts automatically. Watch the finite-model penalty (blue) and finite-data penalty (orange) trade off, and see where famous models actually landed on their budget’s U-curve.
        </>
      }
    >
      <div style={{ marginBottom: 10 }}>
        <Segmented
          label="fixed compute budget"
          options={ISO_PRESETS.map((p, i) => ({ value: i, label: p.label }))}
          value={presetIdx}
          onChange={switchPreset}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 10 }}>
        <div style={{ flex: '1 1 240px' }}>
          <Slider
            label="model size N (tokens D = C / 6N adjusts automatically)"
            value={clampedLogN}
            min={minLogN}
            max={maxLogN}
            step={0.01}
            onChange={setLogN}
            format={(v) => `${fmtCount(10 ** v)} params`}
            color="accent"
          />
        </div>
        <div className="btn-row">
          <button className="btn" onClick={() => setLogN(Math.log10(preset.run.N))}>
            {preset.run.name}
          </button>
          <button className="btn primary" onClick={() => setLogN(Math.log10(opt.N))}>
            Chinchilla opt ({fmtCount(opt.N)})
          </button>
        </div>
      </div>

      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1.2 1 320px', minWidth: 0 }}>
          <Plot
            width={450}
            height={225}
            x={[minLogN, maxLogN]}
            y={[yMin, yMax]}
            pad={48}
            grid
            axes
            tickLabels={false}
            xLabel="model size N (log scale)"
            yLabel="loss L"
            ariaLabel="IsoFLOP curve of predicted loss against model size at fixed compute"
            style={{ width: '100%', height: 'auto', maxWidth: 470, display: 'block' }}
          >
            {({ x, y }) => {
              const clampedY = (v: number) => y(Math.min(yMax, Math.max(yMin, v)));
              return (
                <g>
                  {[yMin, (yMin + yMax) / 2, yMax].map((t) => (
                    <text key={t} className="plot-tick" x={x(minLogN) - 6} y={y(t) + 3.5} textAnchor="end">
                      {t.toFixed(2)}
                    </text>
                  ))}
                  {[minLogN, Math.log10(opt.N), maxLogN].map((t, i) => (
                    <text
                      key={i}
                      className="plot-tick"
                      x={x(t)}
                      y={y(yMin) + 14}
                      textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}
                    >
                      {fmtCount(10 ** t)}
                    </text>
                  ))}
                  <polyline
                    points={curve.map(([a, b]) => `${x(a).toFixed(1)},${clampedY(b).toFixed(1)}`).join(' ')}
                    fill="none"
                    stroke="var(--c-c)"
                    strokeWidth={2.4}
                  />
                  {/* Chinchilla optimum */}
                  <circle cx={x(Math.log10(opt.N))} cy={clampedY(opt.total)} r={4.5} fill="var(--c-c)" />
                  <text x={x(Math.log10(opt.N))} y={clampedY(opt.total) + 16} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--c-c)">
                    opt {fmtCount(opt.N)}
                  </text>
                  {/* Historical frontier run */}
                  <circle cx={x(Math.log10(preset.run.N))} cy={clampedY(runTerms.total)} r={5} fill="var(--c-b)" />
                  <text
                    x={x(Math.log10(preset.run.N))}
                    y={clampedY(runTerms.total) - 9}
                    textAnchor="middle"
                    fontSize={10.5}
                    fontFamily="var(--font-mono)"
                    fill="var(--c-b)"
                  >
                    {preset.run.name}
                  </text>
                  {/* Current slider dot */}
                  <circle cx={x(clampedLogN)} cy={clampedY(cur.total)} r={6} fill="var(--accent)" stroke="var(--bg-elev)" strokeWidth={1.5} />
                </g>
              );
            }}
          </Plot>
        </div>

        <div className="widget-col" style={{ flex: '1 1 220px', minWidth: 0 }}>
          <div className="readout" style={{ fontSize: 12.5, display: 'grid', gap: 5 }}>
            <div>
              N = <strong>{fmtCount(N)}</strong> · D = <strong>{fmtCount(D)}</strong>
            </div>
            <div>
              ratio D / N = <strong>{(D / N).toFixed(1)} tokens / param</strong>
            </div>
            <div>
              predicted loss = <strong style={{ color: 'var(--accent-text)' }}>{f4(cur.total)}</strong> (gap +{f4(cur.total - opt.total)})
            </div>
          </div>

          <div className="field-label" style={{ marginTop: 10 }}>
            excess loss split (above E = {CHINCHILLA.E.toFixed(2)})
          </div>
          <div style={{ display: 'flex', height: 18, borderRadius: 5, overflow: 'hidden', border: '1px solid var(--border)' }}>
            <div
              style={{
                width: `${(cur.paramPenalty / excessSum) * 100}%`,
                background: 'var(--c-a)',
              }}
              title="Finite-model penalty A / N^α"
            />
            <div
              style={{
                width: `${(cur.dataPenalty / excessSum) * 100}%`,
                background: 'var(--c-b)',
              }}
              title="Finite-data penalty B / D^β"
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }} className="mono">
            <span style={{ color: 'var(--c-a)' }}>model: +{cur.paramPenalty.toFixed(3)}</span>
            <span style={{ color: 'var(--c-b)' }}>data: +{cur.dataPenalty.toFixed(3)}</span>
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {D / N < 6 ? (
          <>
            <strong>Under-trained (too big for this budget, D/N = {(D / N).toFixed(1)}):</strong> the orange finite-data penalty ({cur.dataPenalty.toFixed(3)}) dominates — like {preset.run.id === 'gpt3' ? 'GPT-3 175B, which saw only 1.7 tokens per parameter' : 'pre-Chinchilla giant models'}. Shrinking N and feeding it more tokens lowers the loss for the exact same FLOPs.
          </>
        ) : D / N > 60 ? (
          <>
            <strong>Over-trained (smaller model, D/N = {(D / N).toFixed(0)}):</strong> the blue finite-model penalty ({cur.paramPenalty.toFixed(3)}) dominates. That pays a small loss penalty (+{f4(cur.total - opt.total)} nats) at training time, but buys a much smaller, faster model for inference — exactly why Meta trained Llama 3 70B on 15T tokens.
          </>
        ) : (
          <>
            <strong>Near the Chinchilla sweet spot (D/N = {(D / N).toFixed(1)}):</strong> finite-model penalty ({cur.paramPenalty.toFixed(3)}) and finite-data penalty ({cur.dataPenalty.toFixed(3)}) are evenly balanced, sitting right at the bottom of the IsoFLOP U-curve (opt = {fmtCount(opt.N)} params on {fmtCount(opt.D)} tokens).
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Inference vs Training lifecycle compute

export function InferenceVsTraining() {
  const [dInfT, setDInfT] = useState(20); // trillions of lifetime inference tokens
  const targetL = 2.0;

  const dInf = dInfT * 1e12;
  const opt = useMemo(() => lifecycleOptimal(targetL, dInf), [dInf]);

  const points = useMemo(() => {
    const out: { logN: number; N: number; trainC: number; infC: number; totalC: number }[] = [];
    for (let i = 0; i <= 70; i++) {
      const logN = 9.9 + (i / 70) * 1.1; // ~8B to 100B
      const N = 10 ** logN;
      const D = tokensForLoss(N, targetL);
      if (D === null) continue;
      const trainC = trainFlops(N, D);
      const infC = inferenceFlops(N, dInf);
      out.push({ logN, N, trainC, infC, totalC: trainC + infC });
    }
    return out;
  }, [dInf]);

  const chinchilla50D = tokensForLoss(50e9, targetL)!;
  const chinchilla50Total = trainFlops(50e9, chinchilla50D) + inferenceFlops(50e9, dInf);

  return (
    <Widget
      title="Why Llama 3 broke Chinchilla on purpose: training + inference total cost"
      wide
      caption={
        <>
          Suppose you need a model of quality <strong>L = 2.00</strong>. Slide how many lifetime inference tokens your users will generate (2 N FLOPs each) from 0T (a research paper nobody deploys) to 50T (a global product). Watch the optimal model size shrink from 47B down to 14B.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 10 }}>
        <div style={{ flex: '1 1 240px' }}>
          <Slider
            label="lifetime inference tokens served (D_inf)"
            value={dInfT}
            min={0}
            max={50}
            step={1}
            onChange={setDInfT}
            format={(v) => `${v}T tokens`}
            color="c-b"
          />
        </div>
        <div className="btn-row">
          <button className="btn" onClick={() => setDInfT(0)}>
            0T (paper)
          </button>
          <button className="btn" onClick={() => setDInfT(5)}>
            5T
          </button>
          <button className="btn primary" onClick={() => setDInfT(20)}>
            20T (product)
          </button>
          <button className="btn" onClick={() => setDInfT(50)}>
            50T (huge scale)
          </button>
        </div>
      </div>

      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1.2 1 320px', minWidth: 0 }}>
          <LogChart
            series={[
              { points: points.map((p): [number, number] => [p.N, p.totalC / 1e23]), color: 'var(--c-c)', label: 'total' },
              { points: points.map((p): [number, number] => [p.N, p.trainC / 1e23]), color: 'var(--c-a)', dashed: true, label: 'train' },
              ...(dInf > 0 ? [{ points: points.map((p): [number, number] => [p.N, p.infC / 1e23]), color: 'var(--c-b)', dashed: true, label: 'inference' }] : []),
            ]}
            xDomain={[8e9, 1e11]}
            yDomain={[1, 100]}
            logX
            xTicks={[1e10, 2e10, 5e10, 1e11]}
            xTickFormat={(t) => fmtCount(t)}
            marker={opt.N}
            xLabel="model size N (y: ×10²³ FLOPs)"
            ariaLabel="Total lifecycle compute in units of 10^23 FLOPs versus model size at target loss 2.00"
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            <span>
              <strong style={{ color: 'var(--c-c)' }}>—</strong> total compute
            </span>
            <span>
              <strong style={{ color: 'var(--c-a)' }}>--</strong> training (6ND)
            </span>
            {dInf > 0 && (
              <span>
                <strong style={{ color: 'var(--c-b)' }}>--</strong> inference (2ND_inf)
              </span>
            )}
          </div>
        </div>

        <div className="widget-col" style={{ flex: '1 1 220px', minWidth: 0 }}>
          <div className="readout" style={{ fontSize: 12.5, display: 'grid', gap: 6 }}>
            <div>
              lifecycle-optimal size: <strong style={{ color: 'var(--c-c)' }}>{fmtCount(opt.N)}</strong>
            </div>
            <div>
              train on: <strong>{fmtCount(opt.D)}</strong> ({Math.round(opt.ratio)} tok/param)
            </div>
            <div>
              total FLOPs: <strong>{fmtFlops(opt.totalC)}</strong>
            </div>
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 5, color: 'var(--text-muted)' }}>
              vs Chinchilla 50B: <strong>{fmtFlops(chinchilla50Total)}</strong>
              {dInf > 0 && (
                <>
                  {' '}
                  (<strong style={{ color: 'var(--c-good)' }}>{(chinchilla50Total / opt.totalC).toFixed(2)}× cheaper</strong>)
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {dInfT === 0 ? (
          <>
            <strong>Zero inference (research run only):</strong> total compute equals training compute, so classic Chinchilla wins — <strong>{fmtCount(opt.N)}</strong> parameters trained on <strong>{fmtCount(opt.D)}</strong> tokens ({Math.round(opt.ratio)} tokens/param).
          </>
        ) : (
          <>
            <strong>Serving {dInfT}T inference tokens:</strong> inference compute scales linearly with model size N, so a big Chinchilla 50B model becomes painfully expensive to run. Shrinking to <strong>{fmtCount(opt.N)}</strong> and over-training on <strong>{fmtCount(opt.D)}</strong> tokens ({Math.round(opt.ratio)} tok/param) saves{' '}
            <strong>{((1 - opt.totalC / chinchilla50Total) * 100).toFixed(0)}%</strong> of total lifecycle compute at the exact same L = 2.00 quality!
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Spend the compute budget

const SCALING_LAB_KEY = 'nl-scaling-lab-v1';

function loadLabDone(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(SCALING_LAB_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore
  }
  return {};
}

export function ComputeBudgetLab() {
  const [briefIdx, setBriefIdx] = useState(0);
  const [logN, setLogN] = useState(Math.log10(150e9)); // deliberately starts under-trained!
  const [logD, setLogD] = useState(Math.log10(300e9));
  const [doneMap, setDoneMap] = useState<Record<string, boolean>>(loadLabDone);

  const brief = BUDGET_BRIEFS[briefIdx];
  const N = 10 ** logN;
  const D = 10 ** logD;
  const tC = trainFlops(N, D);
  const iC = brief.dInf ? inferenceFlops(N, brief.dInf) : 0;
  const totalC = tC + iC;
  const L = predictedLoss(N, D);
  const passes = brief.check(N, D);

  const exactOpt = useMemo(() => {
    if (brief.id === 'research') {
      const o = chinchillaOptimal(5.0e23);
      return { N: o.N, D: o.D, L: o.total, totalC: 5.0e23 };
    }
    if (brief.id === 'edge') {
      const n = 20e9;
      const d = 6.0e23 / (6 * n);
      return { N: n, D: d, L: predictedLoss(n, d), totalC: 6.0e23 };
    }
    const o = lifecycleOptimal(2.0, 20e12);
    return { N: o.N, D: o.D, L: 2.0, totalC: o.totalC };
  }, [brief.id]);

  useEffect(() => {
    if (passes && !doneMap[brief.id]) {
      setDoneMap((prev) => {
        const next = { ...prev, [brief.id]: true };
        try {
          localStorage.setItem(SCALING_LAB_KEY, JSON.stringify(next));
        } catch {
          // ignore
        }
        return next;
      });
    }
  }, [passes, brief.id, doneMap]);

  const switchBrief = (i: number) => {
    setBriefIdx(i);
    if (i === 0) {
      setLogN(Math.log10(200e9));
      setLogD(Math.log10(350e9));
    } else if (i === 1) {
      setLogN(Math.log10(35e9));
      setLogD(Math.log10(1.5e12));
    } else {
      setLogN(Math.log10(50e9));
      setLogD(Math.log10(0.9e12));
    }
  };

  const lossOk = L <= brief.targetLoss;
  const trainCOk = brief.maxTrainC ? tC <= brief.maxTrainC * 1.001 : true;
  const maxNOk = brief.maxN ? N <= brief.maxN * 1.001 : true;
  const totalCOk = brief.maxTotalC ? totalC <= brief.maxTotalC : true;

  return (
    <Widget
      title="Lab: Spend the compute budget"
      wide
      caption={
        <>
          Three real-world training briefs. Choose parameters N and training tokens D to satisfy every constraint — then see how close your design gets to the exact calculus / lifecycle optimum.
        </>
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="engineering brief"
          options={BUDGET_BRIEFS.map((b, i) => ({
            value: i,
            label: `${doneMap[b.id] ? '✓ ' : ''}${b.name}`,
          }))}
          value={briefIdx}
          onChange={switchBrief}
        />
      </div>

      <div
        style={{
          padding: '10px 12px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          marginBottom: 12,
          fontSize: 13.5,
        }}
      >
        <strong>Brief:</strong> {brief.brief}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 12 }}>
        <Slider
          label="model parameters N"
          value={logN}
          min={9.7}
          max={11.5}
          step={0.01}
          onChange={setLogN}
          format={(v) => fmtCount(10 ** v)}
          color="c-a"
        />
        <Slider
          label="training tokens D"
          value={logD}
          min={11.0}
          max={13.2}
          step={0.01}
          onChange={setLogD}
          format={(v) => fmtCount(10 ** v)}
          color="c-b"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 8, marginBottom: 12 }}>
        <div style={{ background: 'var(--bg-sunken)', border: `1px solid ${lossOk ? 'var(--c-good)' : 'var(--border)'}`, borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>predicted loss L</div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: lossOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {f4(L)}
          </div>
          <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
            {lossOk ? '✓ ' : '○ '}target ≤ {brief.targetLoss.toFixed(3)}
          </div>
        </div>

        {brief.maxTrainC && (
          <div style={{ background: 'var(--bg-sunken)', border: `1px solid ${trainCOk ? 'var(--c-good)' : 'var(--border)'}`, borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>train compute 6ND</div>
            <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: trainCOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
              {fmtFlops(tC)}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
              {trainCOk ? '✓ ' : '○ '}limit ≤ {fmtFlops(brief.maxTrainC)}
            </div>
          </div>
        )}

        {brief.maxN && (
          <div style={{ background: 'var(--bg-sunken)', border: `1px solid ${maxNOk ? 'var(--c-good)' : 'var(--border)'}`, borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>GPU VRAM fit (N)</div>
            <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: maxNOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
              {fmtCount(N)}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
              {maxNOk ? '✓ ' : '○ '}limit ≤ {fmtCount(brief.maxN)}
            </div>
          </div>
        )}

        {brief.maxTotalC && (
          <div style={{ background: 'var(--bg-sunken)', border: `1px solid ${totalCOk ? 'var(--c-good)' : 'var(--border)'}`, borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>lifecycle FLOPs</div>
            <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: totalCOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
              {fmtFlops(totalC)}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
              {totalCOk ? '✓ ' : '○ '}limit ≤ {fmtFlops(brief.maxTotalC)}
            </div>
          </div>
        )}

        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>tokens / param</div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-text)' }}>
            {(D / N).toFixed(1)}×
          </div>
          <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>Chinchilla ≈ 18–20×</div>
        </div>
      </div>

      <div className="live-note">
        {passes ? (
          <>
            <strong>Brief solved!</strong> Your design ({fmtCount(N)} params, {fmtCount(D)} tokens, {(D / N).toFixed(0)} tok/param) hits loss <strong>{f4(L)}</strong>
            {brief.maxTotalC ? ` at ${fmtFlops(totalC)} total lifecycle FLOPs` : ''}. Exact optimum for this brief:{' '}
            <strong>
              {fmtCount(exactOpt.N)} params on {fmtCount(exactOpt.D)} tokens
            </strong>{' '}
            ({brief.maxTotalC ? `${fmtFlops(exactOpt.totalC)} total FLOPs` : `loss ${f4(exactOpt.L)}`}).
          </>
        ) : (
          <>
            <strong>Not quite yet.</strong>{' '}
            {!maxNOk
              ? `Model is too big for a single 24 GB GPU (${fmtCount(N)} > ${fmtCount(brief.maxN!)}). Shrink N and compensate with far more training tokens D!`
              : !totalCOk
                ? `Lifecycle compute is over budget (${fmtFlops(totalC)}). Notice inference alone burns ${fmtFlops(iC)} — shrink N and over-train on more tokens to cut serving cost.`
                : !trainCOk
                  ? `Over training compute budget (${fmtFlops(tC)} > ${fmtFlops(brief.maxTrainC!)}). Lower N or D so 6ND stays under budget.`
                  : `Inside budget, but loss ${f4(L)} is still above ${brief.targetLoss.toFixed(3)}. Move closer to the optimal N/D balance and use your available budget!`}
          </>
        )}
      </div>
    </Widget>
  );
}
