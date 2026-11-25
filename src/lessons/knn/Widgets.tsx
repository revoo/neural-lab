// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
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
  knnMetrics,
  knnQuery,
  labExhaustive,
  type KnnBrief,
  type KnnLabConfig,
  type Metric,
  type Weighting,
} from './knn';

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

export function KnnLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [showOpt, setShowOpt] = useState(false);

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const res = useMemo(() => evaluateKnnLab(brief, entry.cfg), [brief, entry.cfg]);
  const pass = briefPasses(brief, res);
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
  const update = (patch: Partial<KnnLabConfig>) => persist({ ...lab, [brief.id]: { ...entry, cfg: { ...entry.cfg, ...patch } } });

  const toggleCol = (colIdx: number) => {
    const has = entry.cfg.cols.includes(colIdx);
    if (has && entry.cfg.cols.length === 1) return;
    const next = has ? entry.cfg.cols.filter((c) => c !== colIdx) : [...entry.cfg.cols, colIdx].sort((a, b) => a - b);
    update({ cols: next });
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

  return (
    <Widget
      title="Lab: Rescue the kNN Classifier"
      wide
      caption={
        <>
          Each brief starts with a broken kNN setup. Diagnose what is hurting distance quality — scale mismatch, irrelevant noise columns, or over-smoothed majority voting — and fix it. Every configuration is scored on held-out test data and saved in this browser.
        </>
      }
    >
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

      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
      </div>

      <div style={{ display: 'grid', gap: 10 }}>
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

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
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
      </div>

      <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
        <div className="btn-row">
          <span className={`status ${pass ? 'good' : 'bad'}`}>{pass ? 'Meets the brief' : 'Not yet'}</span>
          {entry.solved && (
            <span className="status good">
              Solved
              {entry.bestMetric !== null
                ? brief.id === 'noisy-cols'
                  ? ` · fewest cols: ${entry.bestMetric}`
                  : brief.id === 'pockets'
                    ? ` · best recall: ${entry.bestMetric.toFixed(1)}%`
                    : ` · best acc: ${entry.bestMetric.toFixed(1)}%`
                : ''}
            </span>
          )}
        </div>

        <div className="readout">
          <span style={{ color: res.accuracy >= brief.minAcc - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {res.accuracy >= brief.minAcc - 1e-9 ? '✓' : '✗'}
          </span>{' '}
          <span className="k">held-out accuracy </span>
          {pct1(res.accuracy)} (need ≥ {pct1(brief.minAcc)})
          {brief.minPosRecall !== undefined && (
            <>
              {'\n'}
              <span style={{ color: res.posRecall >= brief.minPosRecall - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {res.posRecall >= brief.minPosRecall - 1e-9 ? '✓' : '✗'}
              </span>{' '}
              <span className="k">fault pocket recall </span>
              {pct1(res.posRecall)} (need ≥ {pct1(brief.minPosRecall)})
            </>
          )}
          {brief.featureLabels && (
            <>
              {'\n'}
              <span className="k">columns kept </span>
              {res.colsUsed} of {brief.featureLabels.length}
            </>
          )}
        </div>

        <div className="live-note">
          {pass
            ? brief.id === 'noisy-cols'
              ? `Passing with ${res.colsUsed} column${res.colsUsed === 1 ? '' : 's'} at ${pct1(res.accuracy)}! Optional: can you drop any remaining irrelevant column?`
              : brief.id === 'pockets'
                ? `Passing at ${pct1(res.accuracy)} accuracy and ${pct1(res.posRecall)} fault recall! Optional: can you reach 100% fault recall?`
                : `Passing at ${pct1(res.accuracy)}! Optional: can you tune k and distance to push accuracy even higher?`
            : brief.id === 'unscaled'
              ? 'Raw RPM (0–5,000) swamps Voltage (0–5) by 1,000×. Standardise both sensors first, then tune k.'
              : brief.id === 'noisy-cols'
                ? 'Four of these six columns are pure random noise drowning out the real distances. Toggle columns off systematically to find which two carry the signal.'
                : 'Uniform k = 25 lets the surrounding normal background outvote the small fault pockets (recall collapses). Try smaller k or 1/d distance weighting!'}
        </div>

        {entry.solved && (
          <div>
            <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>
              {showOpt ? 'Hide' : 'Show'} the exhaustive search
            </button>
            {showOpt && ex && (
              <div className="live-note" style={{ marginTop: 10 }}>
                All {ex.total.toLocaleString()} combinations were checked: <strong>{ex.passing}</strong> pass. Best found:{' '}
                {brief.id === 'noisy-cols'
                  ? `${ex.bestCols} columns (${ex.best[0].cfg.cols.map((c) => brief.featureLabels![c]).join(', ')}) at ${pct1(ex.bestAcc)} accuracy`
                  : brief.id === 'pockets'
                    ? `${pct1(ex.bestRecall)} fault recall and ${pct1(ex.bestAcc)} accuracy (${ex.best[0].cfg.metric.toUpperCase()}, ${ex.best[0].cfg.weight}, k = ${ex.best[0].cfg.k})`
                    : `${pct1(ex.bestAcc)} accuracy (standardised, e.g. ${ex.best[0].cfg.metric.toUpperCase()}, ${ex.best[0].cfg.weight}, k = ${ex.best[0].cfg.k})`}
                .
              </div>
            )}
          </div>
        )}
      </div>
    </Widget>
  );
}
