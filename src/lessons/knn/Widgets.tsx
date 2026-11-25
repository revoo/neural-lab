// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
import { Segmented } from '../../components/controls';
import { Plot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import {
  LAB_BRIEFS,
  LAB_KS,
  LAB_METRICS,
  LAB_WEIGHTS,
  MOONS_KS,
  briefPasses,
  evaluateKnnLab,
  evaluateScalingTrap,
  getMoonsData,
  getNoiseDimCurve,
  knnLabHint,
  knnLabKCurve,
  knnMetrics,
  knnQuery,
  labExhaustive,
  type KnnBrief,
  type KnnLabConfig,
  type Metric,
  type Weighting,
} from './knn';
import { sklearnKnnLines } from './realCode';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

// ------------------------------------------------------------------ 1. Interactive Moons Boundary & Clickable Query

export function KnnBoundaryWidget() {
  const [k, setK] = useState<number>(5);
  const [metric, setMetric] = useState<Metric>('l2');
  const [weight, setWeight] = useState<Weighting>('uniform');
  const [query, setQuery] = useState<[number, number]>([0.55, 0.25]);

  const moons = useMemo(() => getMoonsData(), []);
  const qRes = useMemo(() => knnQuery(moons.train, query, k, metric, weight), [moons.train, query, k, metric, weight]);
  const trainAcc = useMemo(() => knnMetrics(moons.train, moons.train, k, metric, weight).accuracy, [moons.train, k, metric, weight]);
  const testAcc = useMemo(() => knnMetrics(moons.train, moons.test, k, metric, weight).accuracy, [moons.train, moons.test, k, metric, weight]);

  const heatFn = useMemo(
    () => (x1: number, x2: number) => knnQuery(moons.train, [x1, x2], k, metric, weight).prob1,
    [moons.train, k, metric, weight],
  );

  const votes1 = qRes.neighbors.filter((n) => n.y === 1).length;
  const votes0 = qRes.neighbors.length - votes1;

  return (
    <Widget
      title="Nearest neighbours live: click anywhere to query the neighbourhood"
      wide
      caption={
        <>
          80 training points (with 6 deliberate noisy label flips) and 180 held-out test points. Click anywhere on the plot to move the query star and see its <strong>k</strong> closest neighbours vote. Compare <code>k = 1</code> (100.0% train, jagged outlier islands, 84.4% test), <code>k = 9</code> (93.9% test peak) and <code>k = 45</code> (81.1% underfit).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 10 }}>
        <Segmented label="Neighbours" value={k} onChange={setK} options={MOONS_KS.map((v) => ({ value: v as number, label: String(v) }))} />
        <Segmented
          label="Distance"
          value={metric}
          onChange={setMetric}
          options={[
            { value: 'l2', label: 'Euclidean (L2)' },
            { value: 'l1', label: 'Manhattan (L1)' },
          ]}
        />
        <Segmented
          label="Voting"
          value={weight}
          onChange={setWeight}
          options={[
            { value: 'uniform', label: 'Uniform' },
            { value: 'distance', label: 'Distance (1/d)' },
          ]}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <Plot
            width={380}
            height={300}
            x={[-1.6, 2.6]}
            y={[-1.1, 1.7]}
            pad={28}
            xLabel="x₁"
            yLabel="x₂"
            ariaLabel="kNN decision surface with clickable query point and k nearest neighbours"
            onPointerDown={({ x, y }) => setQuery([Math.max(-1.5, Math.min(2.5, x)), Math.max(-1.0, Math.min(1.6, y))])}
            style={{ cursor: 'crosshair' }}
          >
            <Heatmap f={heatFn} range={[0, 1]} colormap={diverging} resolution={110} opacity={0.55} />
            {({ x, y }) => {
              const qx = x(query[0]);
              const qy = y(query[1]);
              const r = qRes.kthDist;
              const rx = Math.abs(x(query[0] + r) - qx);
              const ry = Math.abs(y(query[1] + r) - qy);
              const nbSet = new Set(qRes.neighbors.map((n) => n.idx));
              return (
                <>
                  {/* Neighbourhood boundary: circle for L2, diamond for L1 */}
                  {metric === 'l2' ? (
                    <ellipse cx={qx} cy={qy} rx={rx} ry={ry} fill="none" stroke="var(--text)" strokeWidth={1.3} strokeDasharray="4 3" opacity={0.75} />
                  ) : (
                    <polygon
                      points={`${qx},${qy - ry} ${qx + rx},${qy} ${qx},${qy + ry} ${qx - rx},${qy}`}
                      fill="none"
                      stroke="var(--text)"
                      strokeWidth={1.3}
                      strokeDasharray="4 3"
                      opacity={0.75}
                    />
                  )}
                  {/* Spokes to the k nearest neighbours */}
                  {qRes.neighbors.map((nb) => {
                    const pt = moons.train[nb.idx];
                    return (
                      <line
                        key={nb.idx}
                        x1={qx}
                        y1={qy}
                        x2={x(pt.x[0])}
                        y2={y(pt.x[1])}
                        stroke={nb.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                        strokeWidth={1.4}
                        opacity={0.85}
                      />
                    );
                  })}
                  {/* Training points */}
                  {moons.train.map((pt, idx) => {
                    const isNb = nbSet.has(idx);
                    return (
                      <circle
                        key={idx}
                        cx={x(pt.x[0])}
                        cy={y(pt.x[1])}
                        r={isNb ? 5.2 : 3.2}
                        fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                        stroke={isNb ? 'var(--text)' : 'var(--bg)'}
                        strokeWidth={isNb ? 1.8 : 0.7}
                      />
                    );
                  })}
                  {/* Query marker */}
                  <circle cx={qx} cy={qy} r={6.5} fill={qRes.pred === 1 ? 'var(--c-a)' : 'var(--c-b)'} stroke="var(--text)" strokeWidth={2.2} />
                </>
              );
            }}
          </Plot>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">query point </span>
            ({query[0].toFixed(2)}, {query[1].toFixed(2)})
            {'\n'}
            <span className="k">nearest {k} votes </span>
            <span style={{ color: 'var(--c-a)', fontWeight: 700 }}>{votes1} blue</span> vs{' '}
            <span style={{ color: 'var(--c-b)', fontWeight: 700 }}>{votes0} orange</span>
            {'\n'}
            <span className="k">predicted P(blue) </span>
            <strong>{pct1(qRes.prob1)}</strong> → {qRes.pred === 1 ? 'Blue' : 'Orange'}
            {'\n'}
            <span className="k">train accuracy </span>
            {pct1(trainAcc)}
            {'\n'}
            <span className="k">held-out test accuracy </span>
            <span style={{ color: testAcc >= 0.92 ? 'var(--c-good)' : 'var(--c-b)', fontWeight: 700 }}>{pct1(testAcc)}</span>
          </div>
          <div className="live-note">
            {k === 1
              ? `At k = 1, every training point matches itself (distance 0), so train accuracy is 100.0% — including all 6 mislabelled outliers! Those jagged islands drag held-out test accuracy down to ${pct1(testAcc)}.`
              : k >= 45
                ? `At k = ${k}, more than half the entire training set votes on every query. The curved moons wash out into a crude flat split (test accuracy drops to ${pct1(testAcc)}).`
                : `At k = ${k} (${metric === 'l2' ? 'circular L2' : 'diamond L1'} neighbourhood, ${weight} vote), noise islands disappear while the crescent curve stays sharp — held-out accuracy ${pct1(testAcc)}.`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. The Feature Scaling Trap

export function ScalingTrapWidget() {
  const [scaled, setScaled] = useState(false);
  const [k, setK] = useState(5);

  const res = useMemo(() => evaluateScalingTrap(k, scaled), [k, scaled]);
  const rawAcc = useMemo(() => evaluateScalingTrap(k, false).accuracy, [k]);
  const scaledAcc = useMemo(() => evaluateScalingTrap(k, true).accuracy, [k]);

  // Pick a representative query point in the middle of the dataset
  const probeRaw = [44, 65000];
  const probe = scaled ? [(probeRaw[0] - res.scaler.mean[0]) / res.scaler.std[0], (probeRaw[1] - res.scaler.mean[1]) / res.scaler.std[1]] : probeRaw;
  const q = useMemo(() => knnQuery(res.train, probe, k, 'l2', 'uniform'), [res.train, probe, k]);
  const nbSet = new Set(q.neighbors.map((n) => n.idx));

  // Always plot in standardised-looking visual coordinates [Age 18..74, Income 15k..135k] so the learner sees
  // which neighbours were ACTUALLY picked in raw vs standardised space!
  const rawData = useMemo(() => evaluateScalingTrap(k, false), [k]);

  return (
    <Widget
      title="The feature scaling trap: Age (20–70) vs Income (€20k–€120k)"
      wide
      caption={
        <>
          Both Age and Annual Income drive loan approval (Age actually carries slightly more weight). Toggle between <strong>Raw units</strong> and <strong>Standardised (z-score)</strong> — and watch which {k} neighbours get picked around the middle applicant (44 yrs, €65,000)!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 10 }}>
        <Segmented
          label="Preprocessing"
          value={scaled ? 1 : 0}
          onChange={(v) => setScaled(v === 1)}
          options={[
            { value: 0, label: 'Raw units (years & euros)' },
            { value: 1, label: 'Standardised (z-score, train-only)' },
          ]}
        />
        <Segmented label="Neighbours" value={k} onChange={setK} options={[3, 5, 9].map((v) => ({ value: v, label: String(v) }))} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <Plot width={380} height={270} x={[18, 74]} y={[15, 135]} pad={36} xLabel="Age (years)" yLabel="Income (€ thousands)" ariaLabel="Neighbours selected in raw units vs standardised units">
            {({ x, y }) => (
              <>
                {q.neighbors.map((nb) => {
                  const pt = rawData.train[nb.idx];
                  return (
                    <line
                      key={nb.idx}
                      x1={x(probeRaw[0])}
                      y1={y(probeRaw[1] / 1000)}
                      x2={x(pt.x[0])}
                      y2={y(pt.x[1] / 1000)}
                      stroke="var(--text)"
                      strokeWidth={1.6}
                      opacity={0.85}
                    />
                  );
                })}
                {rawData.train.map((pt, idx) => {
                  const isNb = nbSet.has(idx);
                  return (
                    <circle
                      key={idx}
                      cx={x(pt.x[0])}
                      cy={y(pt.x[1] / 1000)}
                      r={isNb ? 5.5 : 3}
                      fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                      stroke={isNb ? 'var(--text)' : 'var(--bg)'}
                      strokeWidth={isNb ? 1.8 : 0.6}
                      opacity={isNb ? 1 : 0.7}
                    />
                  );
                })}
                <circle cx={x(probeRaw[0])} cy={y(probeRaw[1] / 1000)} r={6.5} fill="var(--accent)" stroke="var(--text)" strokeWidth={2} />
              </>
            )}
          </Plot>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">feature SDs (train) </span>
            Age {res.scaler.std[0].toFixed(1)} yrs · Income €{Math.round(res.scaler.std[1]).toLocaleString()} ({Math.round(res.scaler.std[1] / res.scaler.std[0])}× wider)
            {'\n'}
            <span className="k">raw k = {k} test accuracy </span>
            <span style={{ color: 'var(--c-bad)', fontWeight: 700 }}>{pct1(rawAcc)}</span>
            {'\n'}
            <span className="k">standardised k = {k} test accuracy </span>
            <span style={{ color: 'var(--c-good)', fontWeight: 700 }}>{pct1(scaledAcc)}</span>
            {'\n'}
            <span className="k">gain from scaling alone </span>
            <strong>+{((scaledAcc - rawAcc) * 100).toFixed(1)} pp</strong>
          </div>
          <div className="live-note">
            {scaled
              ? `With z-score standardisation, the ${k} spokes reach evenly in both Age and Income — test accuracy jumps from ${pct1(rawAcc)} to ${pct1(scaledAcc)} (+${((scaledAcc - rawAcc) * 100).toFixed(1)} pp)!`
              : `Look at the spokes in Raw units: they stretch horizontally all across the Age axis while barely moving vertically! Because a €1,000 income gap squares to 1,000,000 while a 30-year age gap squares to only 900, Euclidean distance picks whoever has the closest income — even if they are 25 years older or younger.`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Curse of Dimensionality (Noise Columns)

export function NoiseDimsWidget() {
  const curve = useMemo(() => getNoiseDimCurve(), []);
  const [idx, setIdx] = useState(0);
  const cur = curve[idx];

  return (
    <Widget
      title="Curse of dimensionality: burying 2 real features under random noise columns"
      wide
      caption={
        <>
          The true label depends only on 2 features (a ring). Now append <strong>0 to 80 pure random noise columns</strong> and run the exact same <code>k = 5</code> kNN classifier. Watch both test accuracy and the median <code>farthest / nearest</code> distance ratio collapse together.
        </>
      }
    >
      <Segmented
        label="Irrelevant noise columns added"
        value={idx}
        onChange={setIdx}
        options={curve.map((r, i) => ({ value: i, label: `+${r.dNoise}` }))}
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, marginTop: 12, alignItems: 'start' }}>
        <div>
          <Plot width={440} height={240} x={[0, 82]} y={[0.4, 1.0]} pad={36} xLabel="noise columns added" yLabel="k=5 test accuracy" ariaLabel="kNN test accuracy collapsing as noise dimensions increase">
            {({ x, y }) => {
              const path = curve.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(r.dNoise).toFixed(1)},${y(r.testAcc).toFixed(1)}`).join(' ');
              return (
                <>
                  <line x1={x(0)} x2={x(82)} y1={y(0.5)} y2={y(0.5)} stroke="var(--text-muted)" strokeWidth={1.1} strokeDasharray="4 4" opacity={0.6} />
                  <path d={path} fill="none" stroke="var(--c-a)" strokeWidth={2.5} />
                  {curve.map((r, i) => (
                    <circle
                      key={r.dNoise}
                      cx={x(r.dNoise)}
                      cy={y(r.testAcc)}
                      r={i === idx ? 6 : 3.5}
                      fill={i === idx ? 'var(--c-b)' : 'var(--c-a)'}
                      stroke="var(--bg)"
                      strokeWidth={1.5}
                    />
                  ))}
                  <text x={x(80)} y={y(0.5) - 6} textAnchor="end" fontSize={10.5} fill="var(--text-muted)">
                    50% coin flip
                  </text>
                </>
              );
            }}
          </Plot>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">dimensions (2 signal + {cur.dNoise} noise) </span>
            <strong>{cur.totalDims}D</strong>
            {'\n'}
            <span className="k">k = 5 held-out test accuracy </span>
            <span style={{ color: cur.testAcc >= 0.85 ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>{pct1(cur.testAcc)}</span>
            {'\n'}
            <span className="k">median farthest / nearest ratio </span>
            <strong>{cur.farOverNear.toFixed(2)}×</strong>
          </div>
          <div className="live-note">
            {cur.dNoise === 0
              ? 'With 0 noise columns (2D), the farthest training point is 30.1× farther than the nearest one — “nearest neighbour” is genuinely local, and kNN hits 96.0%.'
              : cur.dNoise >= 40
                ? `With ${cur.dNoise} random noise columns (${cur.totalDims}D), the farthest point is only ${cur.farOverNear.toFixed(2)}× farther than the nearest! Distances are dominated by random noise, collapsing test accuracy to ${pct1(cur.testAcc)}.`
                : `Even ${cur.dNoise} irrelevant columns cut farthest/nearest down to ${cur.farOverNear.toFixed(2)}× and drop test accuracy from 96.0% to ${pct1(cur.testAcc)}. Unlike linear models or trees, kNN cannot assign zero weight to useless columns.`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Rescue the kNN Classifier

const LAB_KEY = 'nl-knn-lab-v1';

type LabEntry = {
  cfg: KnnLabConfig;
  solved: boolean;
  bestMetric: number | null;
};

type LabState = Record<KnnBrief['id'], LabEntry>;

function defaultCfg(id: KnnBrief['id']): KnnLabConfig {
  if (id === 'unscaled') return { scaled: false, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1] };
  if (id === 'noisy-cols') return { scaled: false, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1, 2, 3, 4, 5] };
  return { scaled: false, metric: 'l2', weight: 'uniform', k: 25, cols: [0, 1] };
}

function loadLab(): LabState {
  const fresh: LabState = {
    unscaled: { cfg: defaultCfg('unscaled'), solved: false, bestMetric: null },
    'noisy-cols': { cfg: defaultCfg('noisy-cols'), solved: false, bestMetric: null },
    pockets: { cfg: defaultCfg('pockets'), solved: false, bestMetric: null },
  };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return fresh;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of LAB_BRIEFS) {
      const e = parsed[b.id];
      if (!e || !e.cfg) continue;
      const c = e.cfg;
      if (
        typeof c.scaled !== 'boolean' ||
        !LAB_METRICS.includes(c.metric) ||
        !LAB_WEIGHTS.includes(c.weight) ||
        !(LAB_KS as readonly number[]).includes(c.k) ||
        !Array.isArray(c.cols) ||
        c.cols.length === 0
      )
        continue;
      fresh[b.id] = {
        cfg: { ...c, scaled: b.allowScalingToggle ? c.scaled : false, cols: [...c.cols] },
        solved: !!e.solved,
        bestMetric: typeof e.bestMetric === 'number' ? e.bestMetric : null,
      };
    }
  } catch {
    // ignore storage errors
  }
  return fresh;
}

function seedShellProgress() {
  const old = loadLab();
  const solved: Record<string, boolean> = {};
  const optimal: Record<string, boolean> = {};
  for (const b of LAB_BRIEFS) {
    if (old[b.id].solved) solved[b.id] = true;
    const m = old[b.id].bestMetric;
    if (m !== null) {
      if (b.id === 'unscaled' && m >= 99.4) optimal[b.id] = true;
      if (b.id === 'noisy-cols' && m <= 2) optimal[b.id] = true;
      if (b.id === 'pockets' && m >= 100) optimal[b.id] = true;
    }
  }
  return { solved, optimal };
}

export function KnnLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [probeIdx, setProbeIdx] = useState<number | null>(null);
  const progress = useLabProgress('nl-knn-lab-v1-shell', seedShellProgress);
  const log = useMachineLog();

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const res = useMemo(() => evaluateKnnLab(brief, entry.cfg), [brief, entry.cfg]);
  const pass = briefPasses(brief, res);
  const ex = useMemo(() => labExhaustive(brief), [brief]);
  const kCurve = useMemo(() => knnLabKCurve(brief, entry.cfg), [brief, entry.cfg]);

  const isOpt =
    pass &&
    (brief.id === 'unscaled'
      ? res.accuracy >= ex.bestAcc - 1e-9
      : brief.id === 'noisy-cols'
        ? res.colsUsed <= ex.bestCols && res.accuracy >= ex.bestAcc - 1e-9
        : res.posRecall >= ex.bestRecall - 1e-9 && res.accuracy >= ex.bestAcc - 1e-9);

  // Pick a dramatic default test point to inspect (a missed fault in pockets, or first error, else 0)
  const activeProbe = useMemo(() => {
    if (probeIdx !== null && probeIdx < res.rawTest.length) return probeIdx;
    if (brief.id === 'pockets') {
      const fnIdx = res.rawTest.findIndex((p, i) => p.y === 1 && res.queries[i].pred === 0);
      if (fnIdx >= 0) return fnIdx;
      const anyPos = res.rawTest.findIndex((p) => p.y === 1);
      if (anyPos >= 0) return anyPos;
    }
    const errIdx = res.rawTest.findIndex((p, i) => res.queries[i].pred !== p.y);
    return errIdx >= 0 ? errIdx : 0;
  }, [probeIdx, brief.id, res.rawTest, res.queries]);

  const probePt = res.rawTest[activeProbe];
  const probeQ = res.queries[activeProbe];

  // Animated vote shares [class1Share, class0Share]
  const voteShares = useTweenedArray([probeQ.prob1 * 100, (1 - probeQ.prob1) * 100], 240);

  // Memoised heatmap for 2D pockets brief (fast: 140 training points)
  const pocketsHeat = useMemo(() => {
    if (brief.id !== 'pockets') return null;
    const { train } = res;
    const { k, metric, weight } = entry.cfg;
    return (x1: number, x2: number) => knnQuery(train, [x1, x2], k, metric, weight).prob1;
  }, [brief.id, res, entry.cfg]);

  // Plot geometry for each brief (always plotted in raw interpretable coordinates so spokes visibly snap!)
  const plotSpec = useMemo(() => {
    if (brief.id === 'unscaled') {
      return {
        x: [0, 5.2] as [number, number],
        y: [0, 5200] as [number, number],
        xLabel: 'Sensor A (Volts)',
        yLabel: 'Sensor B (RPM)',
        c0: 0,
        c1: 1,
        subtitle: entry.cfg.scaled
          ? 'Standardised distances: spokes pull from a balanced circle'
          : 'Raw units: spokes stretch wildly across Volts to hug one RPM band',
      };
    }
    if (brief.id === 'noisy-cols') {
      const c0 = entry.cfg.cols[0] ?? 0;
      const c1 = entry.cfg.cols[1] ?? (c0 === 0 ? 1 : 0);
      const labels = brief.featureLabels!;
      return {
        x: [-3.1, 3.1] as [number, number],
        y: [-3.1, 3.1] as [number, number],
        xLabel: labels[c0],
        yLabel: labels[c1],
        c0,
        c1,
        subtitle: `Plotted on (${labels[c0]}, ${labels[c1]}) · spokes found in all ${entry.cfg.cols.length} active dim${entry.cfg.cols.length === 1 ? '' : 's'}`,
      };
    }
    return {
      x: [-2.4, 2.4] as [number, number],
      y: [-2.4, 2.4] as [number, number],
      xLabel: 'x₁',
      yLabel: 'x₂',
      c0: 0,
      c1: 1,
      subtitle: 'Two rare fault pockets · click any dot to inspect its voting spokes',
    };
  }, [brief.id, brief.featureLabels, entry.cfg.scaled, entry.cfg.cols]);

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

  const logRun = (b: KnnBrief, cfg: KnnLabConfig) => {
    const r = evaluateKnnLab(b, cfg);
    const ok = briefPasses(b, r);
    const extra =
      b.id === 'unscaled'
        ? cfg.scaled
          ? 'z-score'
          : 'raw'
        : b.id === 'noisy-cols'
          ? `${cfg.cols.length} cols [${cfg.cols.map((c) => b.featureLabels![c].replace('Col ', '')).join(',')}]`
          : `rec ${pct1(r.posRecall)}`;
    log.push(
      `k=${cfg.k} ${cfg.metric.toUpperCase()} ${cfg.weight} (${extra}) → far/near ${r.farOverNear.toFixed(2)}× · acc ${pct1(r.accuracy)} ${ok ? '✓' : '✗'}`,
      ok ? 'good' : 'info',
    );
  };

  const update = (patch: Partial<KnnLabConfig>) => {
    const nextCfg = { ...entry.cfg, ...patch };
    persist({ ...lab, [brief.id]: { ...entry, cfg: nextCfg } });
    logRun(brief, nextCfg);
  };

  const toggleCol = (colIdx: number) => {
    const has = entry.cfg.cols.includes(colIdx);
    if (has && entry.cfg.cols.length === 1) return;
    const next = has ? entry.cfg.cols.filter((c) => c !== colIdx) : [...entry.cfg.cols, colIdx].sort((a, b) => a - b);
    update({ cols: next });
  };

  const resetBrief = () => {
    const base = defaultCfg(brief.id);
    persist({ ...lab, [brief.id]: { ...entry, cfg: base } });
    setProbeIdx(null);
    logRun(brief, base);
  };

  const jumpToError = () => {
    const errors = res.rawTest
      .map((p, i) => (res.queries[i].pred !== p.y ? i : -1))
      .filter((i) => i >= 0);
    if (errors.length === 0) return;
    const curPos = errors.indexOf(activeProbe);
    setProbeIdx(errors[(curPos + 1) % errors.length]);
  };

  const goalVal = brief.id === 'noisy-cols' ? res.colsUsed : brief.id === 'pockets' ? Math.round(res.posRecall * 1000) / 10 : Math.round(res.accuracy * 1000) / 10;

  useEffect(() => {
    if (!pass) return;
    const cur = saved.current[brief.id];
    const better =
      cur.bestMetric === null ||
      (brief.id === 'noisy-cols' ? goalVal < cur.bestMetric : goalVal > cur.bestMetric);
    if (cur.solved && !better) return;
    persist({
      ...saved.current,
      [brief.id]: { ...cur, solved: true, bestMetric: better ? goalVal : cur.bestMetric },
    });
  }, [pass, brief.id, goalVal]); // eslint-disable-line react-hooks/exhaustive-deps

  const liveCode = useMemo(() => sklearnKnnLines(brief, entry.cfg), [brief, entry.cfg]);

  const shellBriefs = useMemo(
    () =>
      LAB_BRIEFS.map((b, i) => ({
        id: b.id,
        tab: `Brief ${i + 1}`,
        title: b.title,
        text: b.story,
      })),
    [],
  );

  const statusNode = pass ? (
    <>
      <strong>{isOpt ? 'Optimal — matched the exhaustive best!' : 'Meets the brief!'}</strong>{' '}
      Held-out accuracy {pct1(res.accuracy)}
      {brief.minPosRecall !== undefined ? ` · fault recall ${pct1(res.posRecall)}` : ''}
      {brief.featureLabels ? ` · ${res.colsUsed} of ${brief.featureLabels.length} columns` : ''}
      {!isOpt &&
        (brief.id === 'noisy-cols'
          ? ' · Optional: can you drop any remaining dead-weight column and still pass?'
          : brief.id === 'pockets'
            ? ' · Optional: can you reach 100.0% fault recall without losing overall accuracy?'
            : ' · Optional: can you push held-out accuracy even higher?')}
    </>
  ) : (
    <>
      <strong>Not yet.</strong> Held-out accuracy {pct1(res.accuracy)} (need ≥ {pct1(brief.minAcc)})
      {brief.minPosRecall !== undefined ? ` · fault recall ${pct1(res.posRecall)} (need ≥ ${pct1(brief.minPosRecall)})` : ''}
      {' · '}median farthest/nearest ratio <strong>{res.farOverNear.toFixed(2)}×</strong>. Watch the neighbour spokes and the vote breakdown below.
    </>
  );

  // Post-solve reveal: exhaustive summary + k bias-variance curve (Uniform vs Distance)
  const revealContent = (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ fontSize: 13.5, lineHeight: 1.55 }}>
        All <strong>{ex.total.toLocaleString()}</strong> combinations were checked exhaustively: <strong>{ex.passing}</strong> pass. Best found:{' '}
        <strong>
          {brief.id === 'noisy-cols'
            ? `${ex.bestCols} columns (${ex.best[0].cfg.cols.map((c) => brief.featureLabels![c]).join(', ')}) at ${pct1(ex.bestAcc)} accuracy`
            : brief.id === 'pockets'
              ? `${pct1(ex.bestRecall)} fault recall & ${pct1(ex.bestAcc)} accuracy (${ex.best[0].cfg.metric.toUpperCase()}, ${ex.best[0].cfg.weight}, k = ${ex.best[0].cfg.k})`
              : `${pct1(ex.bestAcc)} accuracy (standardised, e.g. ${ex.best[0].cfg.metric.toUpperCase()}, ${ex.best[0].cfg.weight}, k = ${ex.best[0].cfg.k})`}
        </strong>
        .
      </div>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <Plot
          width={420}
          height={180}
          x={[0, 26]}
          y={[0.35, 1.04]}
          pad={36}
          xLabel="k (neighbours)"
          yLabel={brief.id === 'pockets' ? 'fault recall' : 'held-out accuracy'}
          ariaLabel="Uniform versus distance-weighted performance across k"
        >
          {({ x, y }) => {
            const metricU = (r: (typeof kCurve)[number]) => (brief.id === 'pockets' ? r.uniformRec : r.uniformAcc);
            const metricD = (r: (typeof kCurve)[number]) => (brief.id === 'pockets' ? r.distRec : r.distAcc);
            const pathOf = (fn: (r: (typeof kCurve)[number]) => number) =>
              kCurve.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(r.k).toFixed(1)},${y(Math.max(0.36, fn(r))).toFixed(1)}`).join(' ');
            const targetY = brief.id === 'pockets' ? brief.minPosRecall! : brief.minAcc;
            const curY = brief.id === 'pockets' ? res.posRecall : res.accuracy;
            return (
              <>
                <line x1={x(0)} x2={x(26)} y1={y(targetY)} y2={y(targetY)} stroke="var(--c-good)" strokeDasharray="4 3" strokeWidth={1.2} />
                <path d={pathOf(metricU)} fill="none" stroke="var(--c-b)" strokeWidth={2} />
                <path d={pathOf(metricD)} fill="none" stroke="var(--c-a)" strokeWidth={2} />
                {kCurve.map((r) => (
                  <g key={r.k}>
                    <circle cx={x(r.k)} cy={y(Math.max(0.36, metricU(r)))} r={3} fill="var(--c-b)" />
                    <circle cx={x(r.k)} cy={y(Math.max(0.36, metricD(r)))} r={3} fill="var(--c-a)" />
                  </g>
                ))}
                <circle cx={x(entry.cfg.k)} cy={y(Math.max(0.36, curY))} r={6} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
              </>
            );
          }}
        </Plot>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
        Under your current columns & scaling: Class-A coloured line = Distance (1/d) weighting, Class-B coloured line = Uniform voting, dashed line = brief target, accent ring = your current setting.
      </div>
    </div>
  );

  const totalVoteW = probeQ.w0 + probeQ.w1;
  const topNeighbors = probeQ.neighbors.slice(0, 5);
  const errorCount = res.rawTest.reduce((acc, p, i) => acc + (res.queries[i].pred !== p.y ? 1 : 0), 0);

  return (
    <LabShell
      title="Lab: Rescue the kNN Classifier"
      caption={
        <>
          Each brief starts with a broken kNN deployment. Inspect the live neighbour spokes on the left and the weighted vote table on the right to see what is corrupting distances, then fix it. Click any test point (or “Next error”) to probe its neighbourhood.
        </>
      }
      briefs={shellBriefs}
      active={briefIdx}
      onSelect={(i) => {
        setBriefIdx(i);
        setProbeIdx(null);
      }}
      progress={progress}
      passing={pass}
      optimal={isOpt}
      status={statusNode}
      statusTone={pass ? 'good' : 'neutral'}
      hint={knnLabHint(brief, entry.cfg, res)}
      reveal={{ label: 'the exhaustive search & k-curve', content: revealContent }}
      log={{
        lines: log.lines,
        onClear: log.clear,
        title: 'kNN evaluation log',
        empty: 'Change scaling, columns, k, distance or voting to record a test evaluation.',
      }}
      actions={
        <>
          <button type="button" className="btn small" onClick={resetBrief}>
            Reset brief
          </button>
          {errorCount > 0 && (
            <button type="button" className="btn small" onClick={jumpToError}>
              Next error ({errorCount})
            </button>
          )}
        </>
      }
    >
      {/* Controls */}
      <div style={{ display: 'grid', gap: 10, marginBottom: 12 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'end' }}>
          {brief.allowScalingToggle && (
            <Segmented
              label="Feature scaling"
              value={entry.cfg.scaled ? 1 : 0}
              onChange={(v) => update({ scaled: v === 1 })}
              options={[
                { value: 0, label: 'Raw units' },
                { value: 1, label: 'Standardised (z-score)' },
              ]}
            />
          )}
          <Segmented label="Neighbours" value={entry.cfg.k} onChange={(k) => update({ k })} options={LAB_KS.map((v) => ({ value: v as number, label: String(v) }))} />
          <Segmented
            label="Distance"
            value={entry.cfg.metric}
            onChange={(metric) => update({ metric })}
            options={[
              { value: 'l2', label: 'L2' },
              { value: 'l1', label: 'L1' },
            ]}
          />
          <Segmented
            label="Voting"
            value={entry.cfg.weight}
            onChange={(weight) => update({ weight })}
            options={[
              { value: 'uniform', label: 'Uniform' },
              { value: 'distance', label: 'Distance (1/d)' },
            ]}
          />
        </div>

        {brief.featureLabels && (
          <div>
            <div className="field-label">Active columns ({entry.cfg.cols.length} of {brief.featureLabels.length})</div>
            <div className="btn-row" style={{ marginTop: 4 }}>
              {brief.featureLabels.map((lbl, idx) => {
                const on = entry.cfg.cols.includes(idx);
                return (
                  <button key={lbl} type="button" className={`btn small${on ? ' primary' : ''}`} onClick={() => toggleCol(idx)}>
                    {on ? '✓ ' : ''}
                    {lbl}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Main machinery: Live spokes map (left) + Inside the Vote X-ray (right) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-muted)', textAlign: 'center' }}>{plotSpec.subtitle}</div>
          <Plot width={310} height={290} x={plotSpec.x} y={plotSpec.y} pad={34} xLabel={plotSpec.xLabel} yLabel={plotSpec.yLabel} ariaLabel="kNN neighbourhood spokes and test points">
            {pocketsHeat && <Heatmap f={pocketsHeat} range={[0, 1]} colormap={diverging} resolution={110} opacity={0.42} />}
            {({ x, y }) => {
              const px = x(probePt.x[plotSpec.c0]);
              const py = y(probePt.x[plotSpec.c1]);
              return (
                <>
                  {/* Faint training points */}
                  {res.rawTrain.map((pt, i) => (
                    <circle
                      key={`tr-${i}`}
                      cx={x(pt.x[plotSpec.c0])}
                      cy={y(pt.x[plotSpec.c1])}
                      r={2.2}
                      fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                      opacity={0.32}
                    />
                  ))}
                  {/* Spokes to the k nearest training neighbours */}
                  {probeQ.neighbors.map((nb) => {
                    const tPt = res.rawTrain[nb.idx];
                    return (
                      <line
                        key={`sp-${nb.idx}`}
                        x1={px}
                        y1={py}
                        x2={x(tPt.x[plotSpec.c0])}
                        y2={y(tPt.x[plotSpec.c1])}
                        stroke={nb.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                        strokeWidth={1.35}
                        strokeDasharray="3 2"
                        opacity={0.85}
                      />
                    );
                  })}
                  {/* Highlighted voting training neighbours */}
                  {probeQ.neighbors.map((nb) => {
                    const tPt = res.rawTrain[nb.idx];
                    return (
                      <circle
                        key={`nb-${nb.idx}`}
                        cx={x(tPt.x[plotSpec.c0])}
                        cy={y(tPt.x[plotSpec.c1])}
                        r={3.6}
                        fill={nb.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                        stroke="var(--bg)"
                        strokeWidth={0.9}
                      />
                    );
                  })}
                  {/* Clickable test points (errors ringed) */}
                  {res.rawTest.map((pt, i) => {
                    const wrong = res.queries[i].pred !== pt.y;
                    const isProbe = i === activeProbe;
                    const cx = x(pt.x[plotSpec.c0]);
                    const cy = y(pt.x[plotSpec.c1]);
                    return (
                      <g key={`te-${i}`} style={{ cursor: 'pointer' }} onClick={() => setProbeIdx(i)}>
                        {wrong && <circle cx={cx} cy={cy} r={5} fill="none" stroke="var(--c-bad)" strokeWidth={1.2} opacity={0.85} />}
                        {isProbe && <circle cx={cx} cy={cy} r={7.5} fill="none" stroke="var(--accent)" strokeWidth={2.2} />}
                        <circle
                          cx={cx}
                          cy={cy}
                          r={isProbe ? 4.2 : 2.5}
                          fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                          stroke="var(--bg)"
                          strokeWidth={0.6}
                          opacity={isProbe ? 1 : 0.78}
                        />
                      </g>
                    );
                  })}
                </>
              );
            }}
          </Plot>
        </div>

        {/* Right panel: metrics + live vote breakdown for probed query */}
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout" style={{ margin: 0 }}>
            <span style={{ color: res.accuracy >= brief.minAcc - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
              {res.accuracy >= brief.minAcc - 1e-9 ? '✓' : '✗'}
            </span>{' '}
            <span className="k">held-out accuracy </span>
            <strong>{pct1(res.accuracy)}</strong> (need ≥ {pct1(brief.minAcc)})
            {brief.minPosRecall !== undefined && (
              <>
                {'\n'}
                <span style={{ color: res.posRecall >= brief.minPosRecall - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {res.posRecall >= brief.minPosRecall - 1e-9 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">fault pocket recall </span>
                <strong>{pct1(res.posRecall)}</strong> (need ≥ {pct1(brief.minPosRecall)})
              </>
            )}
            {'\n'}
            <span className="k">median farthest / nearest </span>
            <strong>{res.farOverNear.toFixed(2)}×</strong>
            {' · '}
            <span className="k">errors </span>
            {errorCount} / {res.rawTest.length}
          </div>

          <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 4 }}>
              <span className="field-label" style={{ margin: 0 }}>
                Probe test point #{activeProbe + 1} (true class {probePt.y})
              </span>
              <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: probeQ.pred === probePt.y ? 'var(--c-good)' : 'var(--c-bad)' }}>
                pred {probeQ.pred} {probeQ.pred === probePt.y ? '✓' : '✗ misclassified'}
              </span>
            </div>

            {/* Animated weighted vote bar */}
            <div style={{ display: 'grid', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5 }} className="mono">
                <span>Class 1 vote: {voteShares[0].toFixed(1)}%</span>
                <span>Class 0 vote: {voteShares[1].toFixed(1)}%</span>
              </div>
              <div style={{ height: 12, borderRadius: 6, overflow: 'hidden', display: 'flex', border: '1px solid var(--border)' }}>
                <div style={{ width: `${voteShares[0]}%`, background: 'var(--c-a)' }} />
                <div style={{ width: `${voteShares[1]}%`, background: 'var(--c-b)' }} />
              </div>
            </div>

            {/* Top-5 closest neighbours table */}
            <div style={{ overflowX: 'auto' }}>
              <table className="num-table" style={{ width: '100%', fontSize: 11.5 }}>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Class</th>
                    <th>Distance</th>
                    <th>Vote share</th>
                  </tr>
                </thead>
                <tbody>
                  {topNeighbors.map((nb, idx) => {
                    const w = entry.cfg.weight === 'uniform' ? 1 : 1 / (nb.dist + 1e-6);
                    const share = (100 * w) / Math.max(1e-9, totalVoteW);
                    return (
                      <tr key={nb.idx}>
                        <td className="mono">#{idx + 1}</td>
                        <td className="mono" style={{ color: nb.y === 1 ? 'var(--c-a)' : 'var(--c-b)', fontWeight: 600 }}>
                          {nb.y === 1 ? 'Class 1' : 'Class 0'}
                        </td>
                        <td className="mono">{nb.dist.toFixed(3)}</td>
                        <td className="mono">{share.toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {entry.cfg.k > 5 && (
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                Showing closest 5 of {entry.cfg.k} voting neighbours (farthest voter at d = {probeQ.kthDist.toFixed(3)}).
              </div>
            )}
          </div>
        </div>
      </div>

      <LiveCode code={liveCode} note="scikit-learn KNeighborsClassifier" />
    </LabShell>
  );
}
