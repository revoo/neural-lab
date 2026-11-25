// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import {
  LAB_BRIEFS,
  LAB_DEPTHS,
  LAB_METHODS,
  LAB_NTREES,
  MESSY_COL_NAMES,
  accuracyOf,
  bestGiniSplit,
  countLeaves,
  evaluateSplit,
  evaluateTreeLab,
  fitClassTree,
  fitRandomForest,
  getGiniData,
  getMessyBench,
  getNoisy2DData,
  labExhaustive,
  predictClassTree,
  predictForest,
  type TreeBrief,
  type TreeLabConfig,
  type TreeMethod,
} from './trees';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

// ------------------------------------------------------------------ 1. Gini Split Explorer

export function GiniSplitWidget() {
  const data = useMemo(() => getGiniData(), []);
  const best = useMemo(() => bestGiniSplit(data, [0, 1], 1)!, [data]);
  const [feature, setFeature] = useState<number>(0);
  const [thresh, setThresh] = useState<number>(3.0);

  const sp = useMemo(() => evaluateSplit(data, feature, thresh), [data, feature, thresh]);
  const isNearBest = feature === best.feature && Math.abs(thresh - best.threshold) < 0.2;

  return (
    <Widget
      title="Twenty Questions with maths: find the purest Gini split"
      wide
      caption={
        <>
          Pick a feature (<code>x₁</code> or <code>x₂</code>) and slide the split line across these 60 points to maximise <strong>Gini impurity reduction</strong>. A pure bucket has Gini 0.000; a 50/50 bucket has Gini 0.500. Click “Snap to exact CART split” to see the greedy optimum (<code>x₁ ≤ 5.19</code>).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 8 }}>
        <Segmented
          label="Split axis"
          value={feature}
          onChange={setFeature}
          options={[
            { value: 0, label: 'Vertical cut (x₁)' },
            { value: 1, label: 'Horizontal cut (x₂)' },
          ]}
        />
        <button
          type="button"
          className="btn primary small"
          onClick={() => {
            setFeature(best.feature);
            setThresh(Number(best.threshold.toFixed(2)));
          }}
        >
          Snap to exact CART split
        </button>
      </div>

      <Slider
        label={`threshold (${feature === 0 ? 'x₁' : 'x₂'} ≤ t)`}
        value={thresh}
        min={0.5}
        max={9.5}
        step={0.05}
        onChange={setThresh}
        format={(v) => v.toFixed(2)}
        color="c-a"
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, marginTop: 10, alignItems: 'start' }}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <Plot width={340} height={270} x={[0, 10]} y={[0, 10]} pad={30} xLabel="x₁" yLabel="x₂" ariaLabel="2D scatter plot with draggable axis-aligned decision tree split">
            {({ x, y }) => (
              <>
                {feature === 0 ? (
                  <>
                    <rect x={x(0)} y={y(10)} width={Math.max(0, x(thresh) - x(0))} height={y(0) - y(10)} fill="var(--c-b)" opacity={0.08} />
                    <rect x={x(thresh)} y={y(10)} width={Math.max(0, x(10) - x(thresh))} height={y(0) - y(10)} fill="var(--c-a)" opacity={0.08} />
                    <line x1={x(thresh)} x2={x(thresh)} y1={y(0)} y2={y(10)} stroke="var(--text)" strokeWidth={2.2} strokeDasharray="5 3" />
                  </>
                ) : (
                  <>
                    <rect x={x(0)} y={y(thresh)} width={x(10) - x(0)} height={Math.max(0, y(0) - y(thresh))} fill="var(--c-b)" opacity={0.08} />
                    <rect x={x(0)} y={y(10)} width={x(10) - x(0)} height={Math.max(0, y(thresh) - y(10))} fill="var(--c-a)" opacity={0.08} />
                    <line x1={x(0)} x2={x(10)} y1={y(thresh)} y2={y(thresh)} stroke="var(--text)" strokeWidth={2.2} strokeDasharray="5 3" />
                  </>
                )}
                {data.map((pt, i) => (
                  <circle key={i} cx={x(pt.x[0])} cy={y(pt.x[1])} r={4.2} fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'} stroke="var(--bg)" strokeWidth={0.9} />
                ))}
              </>
            )}
          </Plot>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">parent Gini (all 60 points) </span>
            {sp.parentGini.toFixed(3)}
            {'\n'}
            <span className="k">left / bottom bucket ({sp.nL} pts: {sp.l1} blue, {sp.l0} orange) </span>
            G = {sp.leftGini.toFixed(3)}
            {'\n'}
            <span className="k">right / top bucket ({sp.nR} pts: {sp.r1} blue, {sp.r0} orange) </span>
            G = {sp.rightGini.toFixed(3)}
            {'\n'}
            <span className="k">weighted child Gini </span>
            {sp.weightedGini.toFixed(3)}
            {'\n'}
            <span className="k">impurity reduction (gain) </span>
            <span style={{ color: isNearBest ? 'var(--c-good)' : 'var(--c-a)', fontWeight: 700 }}>{sp.gain.toFixed(3)}</span>{' '}
            (best possible: {best.gain.toFixed(3)})
          </div>
          <div className="live-note">
            {isNearBest
              ? `Right at the greedy CART optimum! Splitting at x₁ ≤ 5.19 sends ${sp.l0} of ${sp.nL} left points to orange (Gini ${sp.leftGini.toFixed(3)}) and ${sp.r1} of ${sp.nR} right points to blue (Gini ${sp.rightGini.toFixed(3)}), cutting overall impurity almost in half.`
              : `This split reduces Gini impurity by ${sp.gain.toFixed(3)} (the best greedy single cut reaches ${best.gain.toFixed(3)}). Try both axes and slide toward the densest colour transition.`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Single Tree vs Random Forest

export function TreeVsForestWidget() {
  const [mode, setMode] = useState<'tree' | 'rf'>('tree');
  const [depth, setDepth] = useState<number>(3);
  const [nTrees, setNTrees] = useState<number>(15);

  const data = useMemo(() => getNoisy2DData(), []);

  const model = useMemo(() => {
    if (mode === 'tree') {
      const t = fitClassTree(data.train, { maxDepth: depth, minLeaf: 1 });
      return {
        trainAcc: accuracyOf(data.train, (x) => predictClassTree(t, x)),
        testAcc: accuracyOf(data.test, (x) => predictClassTree(t, x)),
        leaves: countLeaves(t),
        predict: (x1: number, x2: number) => predictClassTree(t, [x1, x2]),
      };
    }
    const rf = fitRandomForest(data.train, { nTrees, maxDepth: 8, minLeaf: 2, maxFeatures: 1, seed: 202 });
    return {
      trainAcc: accuracyOf(data.train, (x) => predictForest(rf, x)),
      testAcc: accuracyOf(data.test, (x) => predictForest(rf, x)),
      leaves: rf.leaves,
      predict: (x1: number, x2: number) => predictForest(rf, [x1, x2]),
    };
  }, [data, mode, depth, nTrees]);

  const heatFn = useMemo(() => model.predict, [model]);

  return (
    <Widget
      title="One deep tree memorises noise — a Random Forest averages it away"
      wide
      caption={
        <>
          150 noisy training points (12% randomly flipped labels) scored on 250 clean held-out test points (sine boundary). Push a <strong>Single Tree</strong> to depth 10 (99.3% train, 84.8% test — carving narrow boxes around every flipped dot!), then switch to <strong>Random Forest</strong> and raise the tree count from 1 to 40 (test jumps to 98.8%!).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 10 }}>
        <Segmented
          label="Model"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'tree', label: 'Single Decision Tree' },
            { value: 'rf', label: 'Random Forest (depth 8)' },
          ]}
        />
        {mode === 'tree' ? (
          <Segmented label="Max depth" value={depth} onChange={setDepth} options={[1, 2, 3, 5, 10].map((d) => ({ value: d, label: String(d) }))} />
        ) : (
          <Segmented label="Trees in forest" value={nTrees} onChange={setNTrees} options={[1, 5, 15, 40].map((n) => ({ value: n, label: String(n) }))} />
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <Plot width={340} height={300} x={[-2.25, 2.25]} y={[-2.25, 2.25]} pad={28} xLabel="x₁" yLabel="x₂" ariaLabel="Decision surface of single tree vs random forest">
            <Heatmap f={heatFn} range={[0, 1]} colormap={diverging} resolution={120} opacity={0.6} />
            {({ x, y }) => (
              <>
                {data.train.map((pt, i) => (
                  <circle key={i} cx={x(pt.x[0])} cy={y(pt.x[1])} r={3} fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'} stroke="var(--bg)" strokeWidth={0.7} />
                ))}
              </>
            )}
          </Plot>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">train accuracy (12% flipped labels!) </span>
            {pct1(model.trainAcc)}
            {'\n'}
            <span className="k">clean held-out test accuracy </span>
            <span style={{ color: model.testAcc >= 0.95 ? 'var(--c-good)' : 'var(--c-b)', fontWeight: 700 }}>{pct1(model.testAcc)}</span>
            {'\n'}
            <span className="k">total leaves </span>
            {model.leaves}
          </div>
          <div className="live-note">
            {mode === 'tree'
              ? depth >= 5
                ? `Depth ${depth} achieves ${pct1(model.trainAcc)} on noisy training data by boxing in individual flipped points — dragging held-out test accuracy down to ${pct1(model.testAcc)}. Switch to Random Forest to watch averaging erase those boxes.`
                : `A shallow depth-${depth} tree resists the flipped labels (${pct1(model.testAcc)} test) because it only has ${model.leaves} leaves, but axis-aligned boxes can only approximate a smooth sine curve coarsely.`
              : nTrees === 1
                ? `A single bootstrap tree (depth 8) scores only ${pct1(model.testAcc)} — high variance! Now raise the forest to 15 or 40 trees.`
                : `Averaging ${nTrees} bootstrap trees cancels out the random noise boxes (each flipped point only appears in ~63% of bootstrap samples and gets outvoted by the other trees), lifting held-out accuracy to ${pct1(model.testAcc)}!`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Boosting Rounds & Messy Tabular Benchmark

export function BoostingBenchWidget() {
  const bench = useMemo(() => getMessyBench(), []);
  const [roundIdx, setRoundIdx] = useState(2); // index 2 = round 5 (peak 93.1%)
  const curRound = bench.gbtRoundAccs[roundIdx];

  const models = [
    { name: 'kNN (raw, k=5)', acc: bench.knnRaw, color: 'var(--c-b)' },
    { name: 'kNN (z-score, k=5)', acc: bench.knnScaled, color: 'var(--c-b)' },
    { name: 'Single Tree (d=8)', acc: bench.singleTree, color: 'var(--c-a)' },
    { name: 'Random Forest (30)', acc: bench.randomForest, color: 'var(--c-c)' },
    { name: `GBT (d=2, ${curRound.round} rounds)`, acc: curRound.testAcc, color: 'var(--accent)' },
  ];

  return (
    <Widget
      title="Messy 8-column table: trees vs kNN, feature importances & boosting rounds"
      wide
      caption={
        <>
          180 train / 260 test rows: Temp (0–10), RPM (0–10,000 — 1,000× wider!), <strong>6 pure random noise columns</strong>, and 6% flipped labels. Step through Gradient Boosting rounds: watch GBT peak at <strong>round 5 (93.1%)</strong> and then gradually overfit the 6% label flips by round 30 (89.2%).
        </>
      }
    >
      <Segmented
        label="Gradient boosting rounds"
        value={roundIdx}
        onChange={setRoundIdx}
        options={bench.gbtRoundAccs.map((r, i) => ({ value: i, label: String(r.round) }))}
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, marginTop: 12, alignItems: 'start' }}>
        {/* Left: Fair model accuracy bars */}
        <div style={{ display: 'grid', gap: 8 }}>
          <div className="field-label">Held-out accuracy on 8-column messy table</div>
          {models.map((m) => (
            <div key={m.name} style={{ display: 'grid', gap: 2 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5 }}>
                <span>{m.name}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 650 }}>{pct1(m.acc)}</span>
              </div>
              <div style={{ height: 14, borderRadius: 4, background: 'var(--bg-sunken)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ width: `${m.acc * 100}%`, height: '100%', background: m.color, opacity: 0.85 }} />
              </div>
            </div>
          ))}
          <div className="readout" style={{ marginTop: 4 }}>
            <span className="k">GBT round {curRound.round} train / test </span>
            {pct1(curRound.trainAcc)} / <strong>{pct1(curRound.testAcc)}</strong>
          </div>
        </div>

        {/* Right: Random Forest Gini feature importances */}
        <div style={{ display: 'grid', gap: 6 }}>
          <div className="field-label">Random Forest Gini feature importance</div>
          {MESSY_COL_NAMES.map((col, i) => {
            const v = bench.rfImportances[i];
            const isSignal = i < 2;
            return (
              <div key={col} style={{ display: 'grid', gridTemplateColumns: '98px 1fr 48px', gap: 8, alignItems: 'center', fontSize: 12 }}>
                <span style={{ fontWeight: isSignal ? 650 : 400, color: isSignal ? 'var(--text)' : 'var(--text-muted)' }}>{col}</span>
                <div style={{ height: 11, borderRadius: 4, background: 'var(--bg-sunken)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, v * 220)}%`, height: '100%', background: isSignal ? 'var(--c-c)' : 'var(--text-muted)', opacity: isSignal ? 0.9 : 0.45 }} />
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', textAlign: 'right', color: isSignal ? 'var(--c-c)' : 'var(--text-muted)' }}>
                  {pct1(v)}
                </span>
              </div>
            );
          })}
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.45 }}>
            The 2 real features capture <strong>{pct1(bench.rfImportances[0] + bench.rfImportances[1])}</strong> of Gini gain. Noise columns still absorb ~6% each because random feature subsampling occasionally offers a split only noise columns, and deep splits chase the 6% label flips.
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {curRound.round <= 5
          ? `At round ${curRound.round}, GBT climbs to ${pct1(curRound.testAcc)} — beating both raw kNN (70.0%) and scaled kNN (81.2%) without touching feature scaling. Now advance to 20 or 30 rounds.`
          : `By round ${curRound.round}, GBT train accuracy rises to ${pct1(curRound.trainAcc)} while held-out test accuracy slips to ${pct1(curRound.testAcc)}! Unlike bagging, boosting keeps chasing residual errors — which eventually means fitting the 6% mislabelled training rows.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Build the Tabular Ensemble

const LAB_KEY = 'nl-trees-lab-v1';

type LabEntry = {
  cfg: TreeLabConfig;
  solved: boolean;
  bestLeaves: number | null;
};

type LabState = Record<TreeBrief['id'], LabEntry>;

const START: TreeLabConfig = { method: 'tree', maxDepth: 1, nTrees: 5 };

function loadLab(): LabState {
  const fresh: LabState = {
    rules: { cfg: { ...START }, solved: false, bestLeaves: null },
    'noisy-table': { cfg: { ...START }, solved: false, bestLeaves: null },
    diagonal: { cfg: { ...START }, solved: false, bestLeaves: null },
  };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return fresh;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of LAB_BRIEFS) {
      const e = parsed[b.id];
      if (!e || !e.cfg) continue;
      if (
        !LAB_METHODS.includes(e.cfg.method) ||
        !(LAB_DEPTHS as readonly number[]).includes(e.cfg.maxDepth) ||
        !(e.cfg.method === 'tree' || (LAB_NTREES as readonly number[]).includes(e.cfg.nTrees))
      )
        continue;
      fresh[b.id] = {
        cfg: { ...e.cfg, nTrees: e.cfg.method === 'tree' ? 1 : e.cfg.nTrees },
        solved: !!e.solved,
        bestLeaves: typeof e.bestLeaves === 'number' ? e.bestLeaves : null,
      };
    }
  } catch {
    // ignore storage errors
  }
  return fresh;
}

export function TreeEnsembleLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [showOpt, setShowOpt] = useState(false);

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const effectiveCfg: TreeLabConfig = entry.cfg.method === 'tree' ? { ...entry.cfg, nTrees: 1 } : entry.cfg;
  const res = useMemo(() => evaluateTreeLab(brief.id, effectiveCfg), [brief.id, effectiveCfg]);
  const pass = res.testAcc >= brief.minAcc - 1e-9;
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
  const update = (patch: Partial<TreeLabConfig>) => {
    const nextCfg: TreeLabConfig = { ...entry.cfg, ...patch };
    if (nextCfg.method === 'tree') nextCfg.nTrees = 1;
    else if (!(LAB_NTREES as readonly number[]).includes(nextCfg.nTrees)) nextCfg.nTrees = 5;
    persist({ ...lab, [brief.id]: { ...entry, cfg: nextCfg } });
  };

  useEffect(() => {
    if (!pass) return;
    const cur = saved.current[brief.id];
    if (cur.solved && cur.bestLeaves !== null && cur.bestLeaves <= res.leaves) return;
    persist({
      ...saved.current,
      [brief.id]: {
        ...cur,
        solved: true,
        bestLeaves: cur.bestLeaves === null ? res.leaves : Math.min(cur.bestLeaves, res.leaves),
      },
    });
  }, [pass, brief.id, res.leaves]); // eslint-disable-line react-hooks/exhaustive-deps

  const methodName = (m: TreeMethod) => (m === 'tree' ? 'Single Tree' : m === 'rf' ? 'Random Forest' : 'Gradient Boosting');

  return (
    <Widget
      title="Lab: Build the Tabular Ensemble"
      wide
      caption={
        <>
          Pick a tree family (Single Tree, Random Forest, or Gradient Boosting), max depth and ensemble size. Each brief trains live on real data and scores on held-out test rows. Meet the accuracy target; optionally, minimise <strong>total leaves</strong> (model footprint and latency).
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

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
        <Segmented
          label="Method"
          value={entry.cfg.method}
          onChange={(method) => update({ method })}
          options={[
            { value: 'tree', label: 'Single Tree' },
            { value: 'rf', label: 'Random Forest' },
            { value: 'gbt', label: 'Gradient Boosting' },
          ]}
        />
        <Segmented
          label="Max depth"
          value={entry.cfg.maxDepth}
          onChange={(maxDepth) => update({ maxDepth })}
          options={LAB_DEPTHS.map((d) => ({ value: d as number, label: String(d) }))}
        />
        {entry.cfg.method !== 'tree' && (
          <Segmented
            label="Trees / rounds"
            value={entry.cfg.nTrees}
            onChange={(nTrees) => update({ nTrees })}
            options={LAB_NTREES.map((n) => ({ value: n as number, label: String(n) }))}
          />
        )}
      </div>

      <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
        <div className="btn-row">
          <span className={`status ${pass ? 'good' : 'bad'}`}>{pass ? 'Meets the brief' : 'Not yet'}</span>
          {entry.solved && (
            <span className="status good">Solved{entry.bestLeaves !== null ? ` · fewest leaves so far: ${entry.bestLeaves}` : ''}</span>
          )}
        </div>

        <div className="readout">
          <span style={{ color: pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{pass ? '✓' : '✗'}</span>{' '}
          <span className="k">held-out accuracy </span>
          {pct1(res.testAcc)} (need ≥ {pct1(brief.minAcc)})
          {'\n'}
          <span className="k">train accuracy </span>
          {pct1(res.trainAcc)}
          {'\n'}
          <span className="k">total leaves (model footprint) </span>
          <strong>{res.leaves}</strong>
        </div>

        <div className="live-note">
          {pass
            ? `Meets the brief at ${pct1(res.testAcc)} using ${res.leaves} leaves! Optional: can you hit ${pct1(brief.minAcc)} with even fewer leaves?`
            : brief.id === 'rules'
              ? 'This rule is clean and axis-aligned — a shallow Single Tree handles it with almost no leaves at all.'
              : brief.id === 'noisy-table'
                ? 'With 6 noise columns, Random Forest’s feature subsampling dilutes splits while a single tree caps at 97.7%. Try Gradient Boosting with moderate depth and a small number of rounds (before it overfits the 6% label noise!).'
                : 'With 12% flipped training labels on a curved boundary, boosting chases the flipped labels. Bagging (Random Forest) averages the label noise away!'}
        </div>

        {entry.solved && (
          <div>
            <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>
              {showOpt ? 'Hide' : 'Show'} the exhaustive search
            </button>
            {showOpt && ex && (
              <div className="live-note" style={{ marginTop: 10 }}>
                All {ex.total} configurations were trained and scored: <strong>{ex.passing}</strong> pass. Fewest total leaves is{' '}
                <strong>{ex.bestLeaves}</strong> ({pct1(ex.bestAccAtMin)} held-out accuracy), with{' '}
                {ex.best
                  .slice(0, 2)
                  .map((w) => `${methodName(w.cfg.method)} (depth ${w.cfg.maxDepth}${w.cfg.method === 'tree' ? '' : `, ${w.cfg.nTrees} trees`})`)
                  .join('; or ')}
                .
              </div>
            )}
          </div>
        )}
      </div>
    </Widget>
  );
}
