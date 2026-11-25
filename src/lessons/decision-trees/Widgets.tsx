// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import {
  BRIEF_COL_NAMES,
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
  getTreeLabData,
  labExhaustive,
  predictClassTree,
  predictForest,
  treeLabHint,
  type ClassTreeNode,
  type TreeBrief,
  type TreeLabConfig,
  type TreeMethod,
} from './trees';
import { sklearnTreeLines } from './realCode';

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

const OPT_LEAVES: Record<TreeBrief['id'], number> = { rules: 3, 'noisy-table': 40, diagonal: 30 };

function seedShellProgress() {
  const old = loadLab();
  const solved: Record<string, boolean> = {};
  const optimal: Record<string, boolean> = {};
  for (const b of LAB_BRIEFS) {
    if (old[b.id].solved) solved[b.id] = true;
    if (old[b.id].bestLeaves !== null && old[b.id].bestLeaves! <= OPT_LEAVES[b.id]) optimal[b.id] = true;
  }
  return { solved, optimal };
}

function formatTreeRules(node: ClassTreeNode, cols: string[], maxLines = 14): string[] {
  const lines: string[] = [];
  const walk = (n: ClassTreeNode, indent: string, prefix: string) => {
    if (lines.length >= maxLines) return;
    if (n.leaf) {
      const cls = n.prob1 >= 0.5 ? 'Class 1' : 'Class 0';
      lines.push(`${indent}${prefix}→ ${cls} (p=${(100 * n.prob1).toFixed(0)}%, n=${n.n})`);
      return;
    }
    const col = cols[n.feature] ?? `x${n.feature}`;
    const thr = Math.abs(n.threshold) >= 100 ? n.threshold.toFixed(0) : n.threshold.toFixed(2);
    lines.push(`${indent}${prefix}if ${col} ≤ ${thr}:`);
    walk(n.left, indent + '  ', 'yes ');
    walk(n.right, indent + '  ', 'no  ');
  };
  walk(node, '', '');
  if (lines.length >= maxLines) lines.push('  … (deeper branches omitted)');
  return lines;
}

export function TreeEnsembleLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const progress = useLabProgress('nl-trees-lab-v1-shell', seedShellProgress);
  const log = useMachineLog();

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const effectiveCfg: TreeLabConfig = entry.cfg.method === 'tree' ? { ...entry.cfg, nTrees: 1 } : entry.cfg;
  const data = useMemo(() => getTreeLabData(brief.id), [brief.id]);
  const res = useMemo(() => evaluateTreeLab(brief.id, effectiveCfg), [brief.id, effectiveCfg]);
  const pass = res.testAcc >= brief.minAcc - 1e-9;
  const isOpt = pass && res.leaves <= OPT_LEAVES[brief.id];
  const ex = useMemo(() => labExhaustive(brief), [brief]);

  const colNames = BRIEF_COL_NAMES[brief.id];
  const shownShares = useTweenedArray(
    res.splitShares.map((s) => s * 100),
    240,
  );

  // Memoised heatmap on the two real sensor axes
  const heatFn = useMemo(() => (x0: number, x1: number) => res.predictXY(x0, x1), [res]);

  const plotRange = useMemo(() => {
    if (brief.id === 'rules') return { x: [0, 10] as [number, number], y: [0, 10] as [number, number], xl: 'Sensor A', yl: 'Sensor B' };
    if (brief.id === 'noisy-table') return { x: [0, 10] as [number, number], y: [0, 10000] as [number, number], xl: 'Temp (°C)', yl: 'RPM' };
    return { x: [-2.25, 2.25] as [number, number], y: [-2.25, 2.25] as [number, number], xl: 'x₁', yl: 'x₂' };
  }, [brief.id]);

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

  const methodName = (m: TreeMethod) => (m === 'tree' ? 'Single Tree' : m === 'rf' ? 'Random Forest' : 'Gradient Boosting');

  const logRun = (b: TreeBrief, cfg: TreeLabConfig) => {
    const eff: TreeLabConfig = cfg.method === 'tree' ? { ...cfg, nTrees: 1 } : cfg;
    const r = evaluateTreeLab(b.id, eff);
    const ok = r.testAcc >= b.minAcc - 1e-9;
    const nPart = eff.method === 'tree' ? '' : `, n=${eff.nTrees}`;
    const noisePct = r.splitShares.length > 2 ? ` · noise splits ${(r.splitShares.slice(2).reduce((a, v) => a + v, 0) * 100).toFixed(0)}%` : '';
    log.push(
      `${methodName(eff.method)} d=${eff.maxDepth}${nPart} → ${r.leaves} leaves · train ${pct1(r.trainAcc)} · test ${pct1(r.testAcc)}${noisePct} ${ok ? '✓' : '✗'}`,
      ok ? 'good' : 'info',
    );
  };

  const update = (patch: Partial<TreeLabConfig>) => {
    const nextCfg: TreeLabConfig = { ...entry.cfg, ...patch };
    if (nextCfg.method === 'tree') nextCfg.nTrees = 1;
    else if (!(LAB_NTREES as readonly number[]).includes(nextCfg.nTrees)) nextCfg.nTrees = 5;
    persist({ ...lab, [brief.id]: { ...entry, cfg: nextCfg } });
    logRun(brief, nextCfg);
  };

  const resetBrief = () => {
    const base: TreeLabConfig = { method: 'tree', maxDepth: 1, nTrees: 5 };
    persist({ ...lab, [brief.id]: { ...entry, cfg: base } });
    logRun(brief, base);
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

  const liveCode = useMemo(() => sklearnTreeLines(effectiveCfg), [effectiveCfg]);
  const ruleLines = useMemo(() => (res.tree ? formatTreeRules(res.tree, colNames) : []), [res.tree, colNames]);

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

  const errorCount = data.test.reduce((acc, s, i) => acc + (res.testPreds[i] !== s.y ? 1 : 0), 0);
  const gap = res.trainAcc - res.testAcc;

  const statusNode = pass ? (
    <>
      <strong>{isOpt ? 'Optimal — fewest possible total leaves!' : 'Meets the brief!'}</strong> {methodName(effectiveCfg.method)} ·{' '}
      <strong>{res.leaves} leaves</strong> · held-out accuracy {pct1(res.testAcc)} (train {pct1(res.trainAcc)}).{' '}
      {!isOpt && `Optional: can you still clear ${pct1(brief.minAcc)} using ≤ ${OPT_LEAVES[brief.id]} leaves?`}
    </>
  ) : (
    <>
      <strong>Not yet.</strong> Held-out accuracy {pct1(res.testAcc)} (need ≥ {pct1(brief.minAcc)}) · train {pct1(res.trainAcc)} ({gap > 0.04 ? `gap +${(gap * 100).toFixed(1)} pp — overfitting!` : `${res.leaves} leaves`}).
    </>
  );

  // Post-solve reveal: all 35 models plotted as log10(leaves) vs testAcc, coloured by family
  const revealContent = (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ fontSize: 13.5, lineHeight: 1.55 }}>
        All <strong>{ex.total}</strong> configurations across all three families were trained and scored: <strong>{ex.passing}</strong> pass. Fewest total leaves is{' '}
        <strong>{ex.bestLeaves}</strong> ({pct1(ex.bestAccAtMin)} held-out accuracy), achieved by:{' '}
        <strong>
          {ex.best
            .slice(0, 2)
            .map((w) => `${methodName(w.cfg.method)} (depth ${w.cfg.maxDepth}${w.cfg.method === 'tree' ? '' : `, ${w.cfg.nTrees} trees`})`)
            .join('; or ')}
        </strong>
        .
      </div>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <Plot
          width={430}
          height={185}
          x={[0.2, 3.2]}
          y={[0.72, 1.02]}
          pad={36}
          xLabel="log₁₀(total leaves)  [0.5≈3 · 1.5≈32 · 2.5≈316]"
          yLabel="held-out accuracy"
          ariaLabel="All 35 tree and ensemble models: footprint versus held-out accuracy"
        >
          {({ x, y }) => {
            const colorOf = (m: TreeMethod) => (m === 'tree' ? 'var(--c-c)' : m === 'rf' ? 'var(--c-a)' : 'var(--c-b)');
            const lx = (leaves: number) => Math.log10(Math.max(1, leaves));
            const opt = ex.best[0];
            return (
              <>
                <line x1={x(0.2)} x2={x(3.2)} y1={y(brief.minAcc)} y2={y(brief.minAcc)} stroke="var(--c-good)" strokeDasharray="4 3" strokeWidth={1.2} />
                {ex.all.map((row, i) => (
                  <circle
                    key={i}
                    cx={x(lx(row.leaves))}
                    cy={y(Math.max(0.73, row.testAcc))}
                    r={row.pass ? 4 : 2.8}
                    fill={colorOf(row.cfg.method)}
                    opacity={row.pass ? 0.9 : 0.3}
                    stroke={row.pass ? 'var(--bg)' : 'none'}
                    strokeWidth={0.7}
                  />
                ))}
                {opt && (
                  <circle cx={x(lx(opt.leaves))} cy={y(Math.max(0.73, opt.testAcc))} r={7.5} fill="none" stroke="var(--c-good)" strokeWidth={2} />
                )}
                <circle cx={x(lx(res.leaves))} cy={y(Math.max(0.73, res.testAcc))} r={5.5} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
              </>
            );
          }}
        </Plot>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 14, fontSize: 12, color: 'var(--text-muted)' }}>
        <span><span style={{ color: 'var(--c-c)', fontWeight: 700 }}>●</span> Single Tree</span>
        <span><span style={{ color: 'var(--c-a)', fontWeight: 700 }}>●</span> Random Forest</span>
        <span><span style={{ color: 'var(--c-b)', fontWeight: 700 }}>●</span> Gradient Boosting</span>
        <span>◎ outer ring = fewest-leaf optimum · inner ring = yours</span>
      </div>
    </div>
  );

  return (
    <LabShell
      title="Lab: Build the Tabular Ensemble"
      caption={
        <>
          Pick a tree family (Single Tree, Random Forest, or Gradient Boosting), max depth, and ensemble size. Watch the live decision regions on the left, and the learned split rules / round-by-round train–test curve / column split shares on the right. Meet the accuracy target; optionally, minimise <strong>total leaves</strong> (model footprint and inference latency).
        </>
      }
      briefs={shellBriefs}
      active={briefIdx}
      onSelect={(i) => setBriefIdx(i)}
      progress={progress}
      passing={pass}
      optimal={isOpt}
      status={statusNode}
      statusTone={pass ? 'good' : 'neutral'}
      hint={treeLabHint(brief, effectiveCfg, res)}
      reveal={{ label: 'the exhaustive 35-model map', content: revealContent }}
      log={{
        lines: log.lines,
        onClear: log.clear,
        title: 'Ensemble training log',
        empty: 'Switch tree family, depth or ensemble size to record a training run.',
      }}
      actions={
        <button type="button" className="btn small" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 12, alignItems: 'end' }}>
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

      {/* Main machinery: 2D boundary (left) + Internals (right) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-muted)', textAlign: 'center' }}>
            Learned decision surface ({plotRange.xl}, {plotRange.yl}) · {errorCount} {errorCount === 1 ? 'error' : 'errors'} ringed
          </div>
          <Plot width={305} height={275} x={plotRange.x} y={plotRange.y} pad={34} xLabel={plotRange.xl} yLabel={plotRange.yl} ariaLabel="Learned tree ensemble decision boundary">
            <Heatmap f={heatFn} range={[0, 1]} colormap={diverging} resolution={110} opacity={0.55} />
            {({ x, y }) => (
              <>
                {data.test.map((pt, i) => {
                  const wrong = res.testPreds[i] !== pt.y;
                  const cx = x(pt.x[0]);
                  const cy = y(pt.x[1]);
                  return (
                    <g key={i}>
                      {wrong && <circle cx={cx} cy={cy} r={5} fill="none" stroke="var(--c-bad)" strokeWidth={1.3} opacity={0.9} />}
                      <circle cx={cx} cy={cy} r={2.5} fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'} stroke="var(--bg)" strokeWidth={0.5} opacity={0.9} />
                    </g>
                  );
                })}
              </>
            )}
          </Plot>
        </div>

        {/* Right panel: Readout + Tree Rules OR Ensemble Curve + Split Shares */}
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout" style={{ margin: 0 }}>
            <span style={{ color: pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{pass ? '✓' : '✗'}</span>{' '}
            <span className="k">held-out accuracy </span>
            <strong>{pct1(res.testAcc)}</strong> (need ≥ {pct1(brief.minAcc)})
            {'\n'}
            <span className="k">train accuracy </span>
            {pct1(res.trainAcc)}
            {' · '}
            <span className="k">total leaves </span>
            <strong>{res.leaves}</strong>
          </div>

          {effectiveCfg.method === 'tree' ? (
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 6 }}>
              <div className="field-label" style={{ margin: 0 }}>Learned decision rules ({res.leaves} leaves)</div>
              <pre className="mono" style={{ margin: 0, fontSize: 11.5, lineHeight: 1.5, overflowX: 'auto', whiteSpace: 'pre' }}>
                {ruleLines.join('\n')}
              </pre>
            </div>
          ) : (
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontSize: 11.5 }}>
                <span className="field-label" style={{ margin: 0 }}>
                  {effectiveCfg.method === 'rf' ? 'Accuracy as bagged trees accumulate' : 'Accuracy across boosting rounds'}
                </span>
                <span className="mono" style={{ color: 'var(--text-muted)' }}>
                  dashed = train · solid = test
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <Plot
                  width={300}
                  height={155}
                  x={[1, Math.max(2, effectiveCfg.nTrees)]}
                  y={[0.68, 1.02]}
                  pad={30}
                  xLabel={effectiveCfg.method === 'rf' ? 'trees' : 'rounds'}
                  yLabel="accuracy"
                  ariaLabel="Ensemble learning curve across trees or boosting rounds"
                >
                  {({ x, y }) => {
                    const trPath = res.curve.map((c, i) => `${i === 0 ? 'M' : 'L'}${x(c.step).toFixed(1)},${y(Math.max(0.69, c.trainAcc)).toFixed(1)}`).join(' ');
                    const tePath = res.curve.map((c, i) => `${i === 0 ? 'M' : 'L'}${x(c.step).toFixed(1)},${y(Math.max(0.69, c.testAcc)).toFixed(1)}`).join(' ');
                    return (
                      <>
                        <line x1={x(1)} x2={x(effectiveCfg.nTrees)} y1={y(brief.minAcc)} y2={y(brief.minAcc)} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1.1} />
                        <path d={trPath} fill="none" stroke="var(--c-b)" strokeDasharray="4 3" strokeWidth={1.6} opacity={0.85} />
                        <path d={tePath} fill="none" stroke="var(--c-a)" strokeWidth={2.2} />
                      </>
                    );
                  }}
                </Plot>
              </div>
            </div>
          )}

          {/* Split-share bars across columns (especially illuminating on 6- and 8-col tables!) */}
          {colNames.length > 2 && (
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 6 }}>
              <div className="field-label" style={{ margin: 0 }}>Where the splits actually went (% of all nodes)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 6 }}>
                {colNames.map((name, idx) => {
                  const pct = shownShares[idx] ?? 0;
                  const isNoise = idx >= 2;
                  return (
                    <div key={name} style={{ display: 'grid', gap: 2 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }} className="mono">
                        <span>{name}</span>
                        <span>{pct.toFixed(0)}%</span>
                      </div>
                      <div style={{ height: 7, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min(100, pct)}%`, height: '100%', background: isNoise ? 'var(--c-b)' : 'var(--c-a)' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <LiveCode code={liveCode} note="scikit-learn tree / ensemble API" />
    </LabShell>
  );
}
