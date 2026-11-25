import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Plot, usePlot } from '../../components/plot';
import { Heatmap, diverging, fieldImage, sequential } from '../../components/heatmap';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { NetDiagram } from '../../components/NetDiagram';
import {
  DATASET_LABELS,
  DEFAULT_CONFIG,
  DOMAIN,
  FEATURES,
  Trainer,
  featurize,
  generate,
  paramCount,
  sizesOf,
  split,
  type Act,
  type Config,
  type DatasetName,
  type FeatureName,
  type Metrics,
  type OptName,
  type Pt,
} from './lab';
import { BEST_FOUND, CHALLENGES, CHALLENGE_DATA, challengeData } from './challenges';
import { PRESETS } from './presets';

const MAX_LAYERS = 4;
const MAX_WIDTH = 8;
const STEPS_PER_FRAME = 8;
const MAX_STEPS = 30000;
const LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];
const BATCHES = [1, 4, 10, 30, 100];
const L2S = [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1];
const SNAP_COLORS = ['#8f8b82', '#b57d06', '#16a06d'];
const STORE = 'nl-training-lab-v1';

type Hist = { step: number; train: number; test: number; trainAcc: number; testAcc: number }[];
type Snapshot = { name: string; color: string; hist: Hist; m: Metrics; summary: string };

const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—');
const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—');

function loadBests(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? '{}');
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------- small pieces

function Points({ train, test, showTest, trainer }: { train: Pt[]; test: Pt[]; showTest: boolean; trainer: Trainer | null }) {
  const { x, y } = usePlot();
  return (
    <g>
      {train.map((p, i) => (
        <circle key={`a${i}`} cx={x(p.x)} cy={y(p.y)} r={3.8} fill={p.label ? 'var(--c-b)' : 'var(--c-a)'} stroke="#fff" strokeWidth={1} />
      ))}
      {showTest &&
        test.map((p, i) => {
          const wrong = trainer ? (trainer.predictXY(p.x, p.y) > 0.5 ? 1 : 0) !== p.label : false;
          return <circle key={`t${i}`} cx={x(p.x)} cy={y(p.y)} r={3.8} fill="var(--bg-elev)" stroke={wrong ? 'var(--c-bad)' : p.label ? 'var(--c-b)' : 'var(--c-a)'} strokeWidth={wrong ? 2.4 : 1.6} />;
        })}
    </g>
  );
}

function LossChart({ hist, snaps }: { hist: Hist; snaps: Snapshot[] }) {
  const W = 330;
  const H = 150;
  const pad = { l: 34, r: 8, t: 8, b: 20 };
  const all = [hist, ...snaps.map((s) => s.hist)];
  const maxStep = Math.max(100, ...all.map((h) => (h.length ? h[h.length - 1].step : 0)));
  const maxLoss = Math.min(2, Math.max(0.8, ...all.flatMap((h) => h.map((p) => Math.max(p.train, p.test))).filter(Number.isFinite)));
  const X = (s: number) => pad.l + (s / maxStep) * (W - pad.l - pad.r);
  const Y = (v: number) => pad.t + (1 - Math.min(v, maxLoss) / maxLoss) * (H - pad.t - pad.b);
  const line = (h: Hist, k: 'train' | 'test') => h.map((p) => `${X(p.step).toFixed(1)},${Y(p[k]).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Training and test loss over time" style={{ background: 'var(--bg-sunken)', borderRadius: 8 }}>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(f * maxLoss)} y2={Y(f * maxLoss)} className="plot-grid" />
          <text x={pad.l - 4} y={Y(f * maxLoss) + 3.5} className="plot-tick" textAnchor="end">
            {(f * maxLoss).toFixed(1)}
          </text>
        </g>
      ))}
      <text x={W - pad.r} y={H - 5} className="plot-tick" textAnchor="end">
        {maxStep.toLocaleString()} steps
      </text>
      {snaps.map((s) => (
        <g key={s.name} opacity={0.75}>
          <polyline points={line(s.hist, 'test')} fill="none" stroke={s.color} strokeWidth={1.4} strokeDasharray="4 3" />
          <polyline points={line(s.hist, 'train')} fill="none" stroke={s.color} strokeWidth={1.4} />
        </g>
      ))}
      <polyline points={line(hist, 'test')} fill="none" stroke="var(--c-b)" strokeWidth={2} strokeDasharray="5 3" />
      <polyline points={line(hist, 'train')} fill="none" stroke="var(--c-a)" strokeWidth={2} />
    </svg>
  );
}

function featureThumb(id: FeatureName) {
  const f = FEATURES.find((q) => q.id === id)!.f;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i <= 20; i++)
    for (let j = 0; j <= 20; j++) {
      const v = f(-DOMAIN + (i / 10) * DOMAIN, -DOMAIN + (j / 10) * DOMAIN);
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  return fieldImage((x, y) => (f(x, y) - lo) / (hi - lo || 1), [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN], 24, sequential);
}

// ---------------------------------------------------------------- the lab


/** Neuron maps (what each hidden neuron and the output respond to over the input square) for a trained network. */
function neuronMaps(t: Trainer, cfg: Config) {
  const sizes = sizesOf(cfg);
  const res = 22;
  const vals = sizes.map((sz) => Array.from({ length: sz }, () => new Float64Array(res * res)));
  for (let r = 0; r < res; r++)
    for (let c = 0; c < res; c++) {
      const x = -DOMAIN + ((c + 0.5) / res) * 2 * DOMAIN;
      const y = DOMAIN - ((r + 0.5) / res) * 2 * DOMAIN;
      t.net.forward(featurize(cfg.features, x, y)).a.forEach((col, ci) => col.forEach((v, j) => (vals[ci][j][r * res + c] = v)));
    }
  const last = sizes.length - 1;
  return vals.slice(1).map((col, k) =>
    col.map((arr) => {
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of arr) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
      let i = 0;
      return k + 1 === last
        ? fieldImage(() => arr[i++], [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN], res, (v) => diverging(1 - v))
        : fieldImage(() => (arr[i++] - lo) / (hi - lo || 1), [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN], res, sequential);
    }),
  );
}

function MapRow({ title, maps }: { title: string; maps: string[][] }) {
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 3 }}>{title}</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
        {maps.map((layer, l) => (
          <div key={l} style={{ display: 'flex', gap: 3 }}>
            {layer.map((src, j) => (
              <img key={j} src={src} width={40} height={40} alt={l === maps.length - 1 ? 'output map' : `layer ${l + 1} neuron ${j + 1} map`} style={{ borderRadius: 4, border: '1px solid var(--border)' }} />
            ))}
            {l < maps.length - 1 && <span className="muted" style={{ alignSelf: 'center' }}>→</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

export function TrainingLab() {
  const [mode, setMode] = useState<string>('free');
  const challenge = CHALLENGES.find((c) => c.id === mode) ?? null;

  const [dataset, setDataset] = useState<DatasetName>('circle');
  const [noise, setNoise] = useState(0.1);
  const [nPoints, setNPoints] = useState(200);
  const [trainFrac, setTrainFrac] = useState(0.5);
  const [dataSeed, setDataSeed] = useState(1);
  const [drawn, setDrawn] = useState<Pt[]>([]);
  const [brush, setBrush] = useState<0 | 1>(1);
  const [cfg, setCfg] = useState<Config>(DEFAULT_CONFIG);
  const [showTest, setShowTest] = useState(true);
  const [running, setRunning] = useState(false);
  const [hist, setHist] = useState<Hist>([]);
  // Event log: flag the moments worth noticing while a run trains.
  const events = useMachineLog();
  const flags = useRef<{ overfit: boolean; perfect: boolean; half: boolean; diverged: boolean }>({ overfit: false, perfect: false, half: false, diverged: false });
  useEffect(() => {
    if (hist.length <= 1) {
      flags.current = { overfit: false, perfect: false, half: false, diverged: false };
      return;
    }
    const last = hist[hist.length - 1];
    const f = flags.current;
    const k = 4; // compare checkpoints a few frames apart, so noise doesn't count as a trend
    if (!f.overfit && hist.length > 3 * k) {
      const a = hist[hist.length - 1 - 3 * k], b = hist[hist.length - 1 - 2 * k], c = hist[hist.length - 1 - k];
      if (a.test < b.test && b.test < c.test && c.test < last.test && last.train < a.train) {
        f.overfit = true;
        events.push(`step ${last.step}: test loss up 3 checks in a row (${a.test.toFixed(3)} → ${last.test.toFixed(3)}) while train loss falls: overfitting has started`, 'bad');
      }
    }
    if (!f.half && last.testAcc >= 0.9) {
      f.half = true;
      events.push(`step ${last.step}: test accuracy reached ${(100 * last.testAcc).toFixed(0)}%`, 'good');
    }
    if (!f.perfect && last.trainAcc >= 1) {
      f.perfect = true;
      events.push(`step ${last.step}: every training point right (test ${(100 * last.testAcc).toFixed(0)}%)`);
    }
    if (!f.diverged && !Number.isFinite(last.train)) {
      f.diverged = true;
      events.push(`step ${last.step}: the loss is no longer a number: training diverged`, 'bad');
    }
  }, [hist]); // eslint-disable-line react-hooks/exhaustive-deps
  const [version, setVersion] = useState(0);
  const [snaps, setSnaps] = useState<Snapshot[]>([]);
  const [bests, setBests] = useState<Record<string, number>>(loadBests);
  const [showRef, setShowRef] = useState(false);
  const [compare, setCompare] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const raf = useRef(0);
  const snapCount = useRef(0);

  const data = useMemo(() => {
    if (challenge) return challengeData(challenge);
    if (dataset === 'draw') return split(drawn, drawn.length > 3 ? trainFrac : 1, dataSeed);
    return split(generate(dataset, nPoints, noise, dataSeed), trainFrac, dataSeed);
  }, [challenge, dataset, drawn, nPoints, noise, dataSeed, trainFrac]);

  const trainer = useMemo(() => (data.train.length >= 2 ? new Trainer(cfg, data.train, data.test.length ? data.test : data.train) : null), [cfg, data]);
  useEffect(() => {
    setRunning(false);
    setHist(trainer ? [{ step: 0, ...pick(trainer.metrics()) }] : []);
    setVersion((v) => v + 1);
  }, [trainer]);

  // training loop
  useEffect(() => {
    if (!running || !trainer) return;
    const tick = () => {
      trainer.step(STEPS_PER_FRAME);
      const m = trainer.metrics();
      setHist((h) => {
        const next = [...h, { step: trainer.steps, ...pick(m) }];
        return next.length > 400 ? next.filter((_, i) => i % 2 === 0 || i === next.length - 1) : next;
      });
      setVersion((v) => v + 1);
      if (trainer.diverged || trainer.steps >= MAX_STEPS) {
        setRunning(false);
        return;
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [running, trainer]);

  const m = useMemo(() => (trainer ? trainer.metrics() : null), [trainer, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const params = paramCount(cfg);
  const ruleBreak = challenge ? challenge.rules(cfg) : null;
  const achieved = !!(challenge && m && !ruleBreak && m.testAcc >= challenge.target);
  useEffect(() => {
    if (!achieved || !challenge) return;
    setBests((b) => {
      if (b[challenge.id] !== undefined && b[challenge.id] <= params) return b;
      const next = { ...b, [challenge.id]: params };
      try {
        localStorage.setItem(STORE, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }, [achieved, challenge, params]);

  // visuals
  const bg = useMemo(() => (trainer ? (x: number, y: number) => trainer.predictXY(x, y) : () => 0.5), [trainer, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const cmap = useMemo(() => (t: number) => diverging(1 - t), []);
  const sizes = sizesOf(cfg);
  const thumbs = useMemo(() => {
    if (!trainer) return undefined;
    const res = 22;
    const vals = sizes.map((s) => Array.from({ length: s }, () => new Float64Array(res * res)));
    for (let r = 0; r < res; r++)
      for (let c = 0; c < res; c++) {
        const x = -DOMAIN + ((c + 0.5) / res) * 2 * DOMAIN;
        const y = DOMAIN - ((r + 0.5) / res) * 2 * DOMAIN;
        const t = trainer.net.forward(featurize(cfg.features, x, y));
        t.a.forEach((col, ci) => col.forEach((v, j) => (vals[ci][j][r * res + c] = v)));
      }
    const last = sizes.length - 1;
    return vals.map((col, ci) =>
      col.map((arr) => {
        let lo = Infinity;
        let hi = -Infinity;
        for (const v of arr) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
        let k = 0;
        return ci === last
          ? fieldImage(() => arr[k++], [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN], res, (v) => diverging(1 - v))
          : fieldImage(() => (arr[k++] - lo) / (hi - lo || 1), [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN], res, sequential);
      }),
    );
  }, [trainer, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const featThumbs = useMemo(() => Object.fromEntries(FEATURES.map((f) => [f.id, featureThumb(f.id)])) as Record<FeatureName, string>, []);

  // config helpers
  const set = (patch: Partial<Config>) => setCfg((c) => ({ ...c, ...patch }));
  const toggleFeature = (id: FeatureName) => {
    const has = cfg.features.includes(id);
    if (has && cfg.features.length === 1) return;
    set({ features: FEATURES.map((f) => f.id).filter((f) => (f === id ? !has : cfg.features.includes(f))) });
  };
  const setWidth = (li: number, d: number) => set({ hidden: cfg.hidden.map((v, k) => (k === li ? Math.max(1, Math.min(MAX_WIDTH, v + d)) : v)) });
  const reset = () => setCfg((c) => ({ ...c })); // new object → new Trainer from the same seed
  const loadPreset = (key: string) => {
    const p = PRESETS[key];
    setMode('free');
    setDataset(p.dataset);
    setNoise(p.noise);
    setNPoints(p.n);
    setTrainFrac(p.trainFrac);
    setDataSeed(1);
    setCfg({ ...DEFAULT_CONFIG, ...p.cfg });
  };
  const enterChallenge = (id: string) => {
    setMode(id);
    setShowRef(false);
    setShowHint(false);
    setCfg({ ...DEFAULT_CONFIG });
  };
  const saveSnap = () => {
    if (!m || hist.length < 2) return;
    const k = snaps.length % 3;
    const name = `Run ${String.fromCharCode(65 + (snapCount.current++ % 26))}`;
    const summary = `${challenge ? challenge.title : DATASET_LABELS[dataset]} · inputs ${cfg.features.map((f) => FEATURES.find((q) => q.id === f)!.label).join(', ')} · ${cfg.hidden.length ? cfg.hidden.join('×') : 'no hidden'} ${cfg.activation} · ${cfg.optimizer} ${cfg.lr} · batch ${cfg.batch}${cfg.l2 ? ` · L2 ${cfg.l2}` : ''}`;
    setSnaps([...snaps.slice(-2), { name, color: SNAP_COLORS[k], hist, m, summary }]);
  };

  const onPlotDown = (p: { x: number; y: number }) => {
    if (challenge || dataset !== 'draw') return;
    if (Math.abs(p.x) > DOMAIN || Math.abs(p.y) > DOMAIN) return;
    setDrawn((d) => [...d, { x: p.x, y: p.y, label: brush }]);
  };

  // live note
  let note: React.ReactNode;
  if (!trainer) note = <>Click on the plot to add points (switch colour with the buttons). You need at least a few of each colour.</>;
  else if (trainer.diverged) note = <><strong>Diverged.</strong> The weights blew up to infinity. The step size is too big for this setup — lower it and reset.</>;
  else if (!m || trainer.steps === 0) note = <>Untrained network: its predictions (the background) are a meaningless smear. Press <strong>Train</strong>.</>;
  else if (challenge) {
    note = ruleBreak ? (
      <>Breaks the challenge rules: {ruleBreak}</>
    ) : achieved ? (
      <>
        <strong>✓ Target reached</strong> with {params} parameters ({pct(m.testAcc)} on the test points). Optional: can you reach it with fewer parameters?
      </>
    ) : (
      <>
        Test accuracy {pct(m.testAcc)}; target {pct(challenge.target)}. {trainer.steps > 3000 && m.testAcc < challenge.target - 0.1 ? 'It seems stuck — this setup may not be able to get there. Change something and reset.' : ''}
      </>
    );
  } else if (m.trainAcc - m.testAcc > 0.1)
    note = (
      <>
        Training accuracy {pct(m.trainAcc)}, test accuracy {pct(m.testAcc)}. The network is getting the points it trains on right and the new ones wrong: it is{' '}
        <strong>memorising</strong> rather than learning the pattern. That gap is the subject of the next module.
      </>
    );
  else note = <>Training: step {trainer.steps.toLocaleString()} (epoch {trainer.epochs.toFixed(1)}). Train accuracy {pct(m.trainAcc)}, test accuracy {pct(m.testAcc)}.</>;

  const ref = challenge ? BEST_FOUND[challenge.id] : null;
  const yourBest = challenge ? bests[challenge.id] : undefined;
  // The reference network, trained once (same data, same steps as the search recorded).
  const refMaps = useMemo(() => {
    if (!compare || !ref || !challenge) return null;
    const refT = new Trainer(ref.cfg, data.train, data.test);
    refT.step(ref.reachedAt);
    return neuronMaps(refT, ref.cfg);
  }, [compare, ref, challenge, data]);

  return (
    <Widget
      title="The training laboratory"
      wide
      caption={
        <>
          Everything is real: a network built from the parts you’ve met, trained live with mini-batch gradient descent and backpropagation. Filled points are the
          training data; hollow ones are <strong>test</strong> data the network never trains on (red rings: currently misclassified). Choose data, inputs, architecture and
          settings, then press <strong>Train</strong>. Any change of setup starts a fresh network from the same random start, so runs are reproducible.
        </>
      }
    >
      {/* mode bar */}
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button className={`btn small${mode === 'free' ? ' primary' : ''}`} onClick={() => setMode('free')}>
          Free play
        </button>
        {CHALLENGES.map((c) => (
          <button key={c.id} className={`btn small${mode === c.id ? ' primary' : ''}`} onClick={() => enterChallenge(c.id)}>
            {bests[c.id] !== undefined ? '✓ ' : ''}
            {c.title}
          </button>
        ))}
      </div>

      {challenge && (
        <div className="live-note" style={{ marginBottom: 10 }}>
          <strong>{challenge.title}.</strong> {challenge.brief} <span className="muted">Rules: {challenge.rulesText} Data: {CHALLENGE_DATA.n} points, half for testing.</span>{' '}
          <button className="btn small" onClick={() => setShowHint(!showHint)}>
            {showHint ? 'Hide hint' : 'Hint'}
          </button>
          {showHint && <div style={{ marginTop: 6 }}>{challenge.hint}</div>}
        </div>
      )}

      {/* run controls */}
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button className={`btn${running ? '' : ' primary'}`} disabled={!trainer || trainer.diverged} onClick={() => setRunning(!running)}>
          {running ? '❚❚ Pause' : '▶ Train'}
        </button>
        <button className="btn small" disabled={!trainer || running || trainer.diverged} onClick={() => { trainer!.step(1); setHist((h) => [...h, { step: trainer!.steps, ...pick(trainer!.metrics()) }]); setVersion((v) => v + 1); }}>
          Step
        </button>
        <button className="btn small" disabled={!trainer} onClick={reset}>
          Reset
        </button>
        <span className="mono muted" style={{ fontSize: 12.5 }}>
          step {trainer?.steps.toLocaleString() ?? 0} · epoch {trainer ? trainer.epochs.toFixed(1) : '0'} · {params} parameters
        </span>
        {mode === 'free' && (
          <select className="num-input" style={{ width: 'auto', textAlign: 'left' }} value="" onChange={(e) => e.target.value && loadPreset(e.target.value)} aria-label="Load an experiment">
            <option value="">Load an experiment…</option>
            {Object.entries(PRESETS).map(([k, p]) => (
              <option key={k} value={k}>
                {p.label}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="widget-row">
        {/* data + features */}
        <div className="widget-col controls" style={{ flex: '0 1 190px' }}>
          {!challenge && (
            <>
              <div>
                <div className="field-label">Data</div>
                <select className="num-input" style={{ width: '100%', textAlign: 'left' }} value={dataset} onChange={(e) => setDataset(e.target.value as DatasetName)} aria-label="dataset">
                  {(Object.keys(DATASET_LABELS) as DatasetName[]).map((d) => (
                    <option key={d} value={d}>
                      {DATASET_LABELS[d]}
                    </option>
                  ))}
                </select>
              </div>
              {dataset === 'draw' ? (
                <div className="btn-row">
                  <button className={`btn small${brush === 0 ? ' primary' : ''}`} onClick={() => setBrush(0)}>
                    ● blue
                  </button>
                  <button className={`btn small${brush === 1 ? ' primary' : ''}`} onClick={() => setBrush(1)}>
                    ● orange
                  </button>
                  <button className="btn small" onClick={() => setDrawn([])}>
                    Clear
                  </button>
                </div>
              ) : (
                <>
                  <Slider label="noise" value={noise} min={0} max={0.5} step={0.05} onChange={setNoise} />
                  <Segmented label="points" value={nPoints} onChange={setNPoints} options={[50, 100, 200, 400]} />
                  <button className="btn small" onClick={() => setDataSeed(dataSeed + 1)}>
                    New sample of data
                  </button>
                </>
              )}
              <Slider label="training share" value={trainFrac} min={0.1} max={0.9} step={0.1} onChange={setTrainFrac} format={(v) => `${Math.round(v * 100)}%`} />
            </>
          )}
          <div>
            <div className="field-label">Inputs (features)</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(54px, 1fr))', gap: 6 }}>
              {FEATURES.map((f) => {
                const on = cfg.features.includes(f.id);
                return (
                  <button
                    key={f.id}
                    onClick={() => toggleFeature(f.id)}
                    aria-pressed={on}
                    style={{ border: `2px solid ${on ? 'var(--accent)' : 'var(--border)'}`, borderRadius: 8, background: 'var(--bg)', padding: 3, cursor: 'pointer', opacity: on ? 1 : 0.55 }}
                  >
                    <img src={featThumbs[f.id]} alt="" width={40} height={40} style={{ display: 'block', margin: '0 auto', borderRadius: 4 }} />
                    <span style={{ fontSize: 11.5, fontFamily: 'var(--font-mono)' }}>{f.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* network */}
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="btn-row" style={{ marginBottom: 6 }}>
            <span className="field-label" style={{ margin: 0 }}>Hidden layers</span>
            <button className="btn small" disabled={cfg.hidden.length >= MAX_LAYERS} onClick={() => set({ hidden: [...cfg.hidden, cfg.hidden[cfg.hidden.length - 1] ?? 4] })}>
              + layer
            </button>
            <button className="btn small" disabled={!cfg.hidden.length} onClick={() => set({ hidden: cfg.hidden.slice(0, -1) })}>
              − layer
            </button>
          </div>
          <div className="btn-row" style={{ marginBottom: 4 }}>
            {cfg.hidden.map((v, li) => (
              <span key={li} className="seg" style={{ alignItems: 'center' }}>
                <button onClick={() => setWidth(li, -1)} aria-label={`fewer neurons in layer ${li + 1}`}>−</button>
                <span className="mono" style={{ padding: '0 5px', fontWeight: 700 }}>{v}</span>
                <button onClick={() => setWidth(li, 1)} aria-label={`more neurons in layer ${li + 1}`}>+</button>
              </span>
            ))}
            {!cfg.hidden.length && <span className="muted" style={{ fontSize: 12.5 }}>none: a single neuron</span>}
          </div>
          <NetDiagram
            sizes={sizes}
            W={trainer ? trainer.net.W : sizes.slice(1).map((n, l) => Array.from({ length: n }, () => Array(sizes[l]).fill(0)))}
            thumbs={thumbs}
            showValues={false}
            inputLabels={cfg.features.map((f) => FEATURES.find((q) => q.id === f)!.label)}
            width={340}
            height={Math.max(200, Math.max(...sizes) * 34 + 20)}
            node={26}
            maxAbsW={3}
            ariaLabel="The network, with each neuron's response map"
          />
        </div>

        {/* output */}
        <div className="widget-col" style={{ flex: '0 1 310px' }}>
          <Plot
            x={[-DOMAIN, DOMAIN]}
            y={[-DOMAIN, DOMAIN]}
            width={310}
            height={310}
            grid={false}
            tickLabels={false}
            ariaLabel="Data and the network's decision regions"
            onPointerDown={onPlotDown}
            style={{ cursor: dataset === 'draw' && !challenge ? 'crosshair' : undefined }}
          >
            <Heatmap f={bg} range={[0, 1]} colormap={cmap} resolution={56} opacity={0.7} />
            <Points train={data.train} test={data.test} showTest={showTest} trainer={trainer} />
          </Plot>
          <Toggle label="Show test data" checked={showTest} onChange={setShowTest} />
          {m && (
            <div className="readout" style={{ marginTop: 6, fontSize: 12.5 }}>
              <span className="c-a">train</span> <span className="k">loss</span> {f3(m.trainLoss)} <span className="k">accuracy</span> {pct(m.trainAcc)}
              {'\n'}
              <span className="c-b">test </span> <span className="k">loss</span> {f3(m.testLoss)} <span className="k">accuracy</span> <strong>{pct(m.testAcc)}</strong>
            </div>
          )}
        </div>
      </div>

      {/* hyperparameters */}
      <div className="controls inline" style={{ marginTop: 12 }}>
        <Segmented label="Activation" value={cfg.activation} onChange={(v: Act) => set({ activation: v })} options={['tanh', 'relu', 'sigmoid', 'linear'].map((v) => ({ value: v as Act, label: v === 'relu' ? 'ReLU' : v }))} />
        <Segmented label="Optimizer" value={cfg.optimizer} onChange={(v: OptName) => set({ optimizer: v })} options={[{ value: 'sgd', label: 'SGD' }, { value: 'momentum', label: 'Momentum' }, { value: 'adam', label: 'Adam' }]} />
        <Segmented label="Step size" value={cfg.lr} onChange={(v: number) => set({ lr: v })} options={LRS.map((v) => ({ value: v, label: String(v) }))} />
        <Segmented label="Batch" value={cfg.batch} onChange={(v: number) => set({ batch: v })} options={BATCHES.map((v) => ({ value: v, label: String(v) }))} />
        <Segmented label="L2 (weight decay)" value={cfg.l2} onChange={(v: number) => set({ l2: v })} options={L2S.map((v) => ({ value: v, label: String(v) }))} />
        <div>
          <div className="field-label">Random start</div>
          <span className="seg" style={{ alignItems: 'center' }}>
            <button onClick={() => set({ seed: Math.max(1, cfg.seed - 1) })} aria-label="previous random start">‹</button>
            <span className="mono" style={{ padding: '0 6px', fontSize: 12.5 }}>#{cfg.seed}</span>
            <button onClick={() => set({ seed: cfg.seed + 1 })} aria-label="next random start">›</button>
          </span>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>

      {/* curves + comparisons */}
      <div className="widget-row" style={{ marginTop: 12 }}>
        <div className="widget-col" style={{ flex: '0 1 340px' }}>
          <div className="field-label">
            Loss: <span className="c-a">train ——</span> <span className="c-b">test - - -</span>
          </div>
          <LossChart hist={hist} snaps={snaps} />
          <MachineLog lines={events.lines} onClear={events.clear} title="Events" empty="Moments worth noticing during a run are flagged here." />
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          {challenge ? (
            <div style={{ border: '1px solid var(--border)', borderRadius: 10, padding: '10px 12px', fontSize: 13.5 }}>
              <div style={{ fontWeight: 700, marginBottom: 4 }}>{challenge.title}</div>
              <div>
                Target: test accuracy ≥ {pct(challenge.target)} · now <strong className={achieved ? 'c-good' : ''}>{m ? pct(m.testAcc) : '—'}</strong> {achieved ? '✓' : ''}
              </div>
              <div>Your fewest parameters so far: <strong>{yourBest ?? '—'}</strong></div>
              {yourBest !== undefined && ref && (
                <div style={{ marginTop: 6 }}>
                  Fewest found by our search: <strong>{ref.params}</strong>.{' '}
                  {yourBest <= ref.params ? (yourBest < ref.params ? 'You beat it!' : 'You matched it.') : ''}{' '}
                  <button className="btn small" onClick={() => setShowRef(!showRef)}>
                    {showRef ? 'Hide' : 'Show'} how
                  </button>
                  {achieved && (
                    <button className="btn small" style={{ marginLeft: 6 }} onClick={() => setCompare(!compare)}>
                      {compare ? 'Hide' : 'Compare'} neuron maps
                    </button>
                  )}
                  {compare && achieved && trainer && refMaps && (() => {
                    return (
                      <div style={{ marginTop: 6 }}>
                        <MapRow title={`Yours (${params} parameters)`} maps={neuronMaps(trainer, cfg)} />
                        <MapRow title={`The search's smallest (${ref.params} parameters, trained ${ref.reachedAt.toLocaleString()} steps)`} maps={refMaps} />
                        <div className="muted" style={{ fontSize: 12 }}>Each square is what one neuron responds to over the input square; the last is the output. Different networks often find different pieces that add up to the same answer.</div>
                      </div>
                    );
                  })()}
                  {showRef && (
                    <div className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>
                      Inputs {ref.cfg.features.map((f) => FEATURES.find((q) => q.id === f)!.label).join(', ')}; {ref.cfg.hidden.length ? `hidden ${ref.cfg.hidden.join('×')} ${ref.cfg.activation}` : 'no hidden layer'}; Adam, step size {ref.cfg.lr}, batch {ref.cfg.batch}; random start #{ref.cfg.seed}; reached the target at step {ref.reachedAt.toLocaleString()}. The search tried {ref.space} ({ref.spaceSize.toLocaleString()} runs), smallest first — so it is the best in that space, not a proof that nothing smaller exists.{' '}
                      <button className="btn small" onClick={() => setCfg({ ...ref.cfg })}>
                        Load it
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="btn-row" style={{ marginBottom: 6 }}>
                <button className="btn small" disabled={!m || hist.length < 2} onClick={saveSnap}>
                  Save this run for comparison
                </button>
                {snaps.length > 0 && (
                  <button className="btn small" onClick={() => setSnaps([])}>
                    Clear saved runs
                  </button>
                )}
              </div>
              {snaps.length === 0 ? (
                <div className="muted" style={{ fontSize: 13 }}>Saved runs appear here and on the chart, so you can change one thing and compare.</div>
              ) : (
                <table className="num-table" style={{ width: '100%' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left' }}>run</th>
                      <th>steps</th>
                      <th>train acc</th>
                      <th>test acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snaps.map((s) => (
                      <tr key={s.name} title={s.summary}>
                        <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)' }}>
                          <span style={{ color: s.color, fontWeight: 700 }}>{s.name}</span> <span className="muted" style={{ fontSize: 11.5 }}>{s.summary}</span>
                        </td>
                        <td>{s.hist[s.hist.length - 1].step}</td>
                        <td>{pct(s.m.trainAcc)}</td>
                        <td>{pct(s.m.testAcc)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
        </div>
      </div>
    </Widget>
  );
}

function pick(m: Metrics) {
  return { train: m.trainLoss, test: m.testLoss, trainAcc: m.trainAcc, testAcc: m.testAcc };
}
