// authored-by: Claude Opus 5.5
import { useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { useFramePlayer } from '../../components/motion';
import { createRng } from '../../lib/rng';
import { Segmented, Slider } from '../../components/controls';
import { Handle, Plot, usePlot } from '../../components/plot';
import {
  DICE,
  MAX_EACH,
  OPENING,
  SOURCES,
  WORKSHOP,
  accuracySE,
  bellGap,
  binomial,
  checkRecipe,
  distMean,
  distSd,
  mean,
  meanAbsDev,
  median,
  normalPdf,
  probLooksBetter,
  recipeChangeLine,
  recipeDist,
  rollRecipe,
  solveWorkshop,
  sumOf,
  variance,
  type DieId,
  type Dist,
  type Recipe,
  type SourceId,
} from './stats';

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

// ------------------------------------------------------------------ shared: bars + bell curve

function Bars({ d, color = 'var(--c-a)', scaleX = 1 }: { d: Dist; color?: string; scaleX?: number }) {
  const { x, y } = usePlot();
  const w = Math.max(1, Math.abs(x(scaleX) - x(0)) * 0.86);
  return (
    <g>
      {d.p.map((q, i) =>
        q < 1e-6 ? null : (
          <rect key={i} x={x((d.min + i) * scaleX) - w / 2} y={y(q)} width={w} height={Math.max(0, y(0) - y(q))} fill={color} opacity={0.75} />
        ),
      )}
    </g>
  );
}

function Bell({ mu, sigma, scale = 1, color = 'var(--c-b)' }: { mu: number; sigma: number; scale?: number; color?: string }) {
  const { x, y } = usePlot();
  const [a, b] = x.domain;
  let d = '';
  for (let k = 0; k <= 200; k++) {
    const xv = a + ((b - a) * k) / 200;
    const yv = normalPdf(xv, mu, sigma) * scale;
    d += `${k ? 'L' : 'M'}${x(xv).toFixed(1)},${y(yv).toFixed(1)}`;
  }
  return <path d={d} fill="none" stroke={color} strokeWidth={2.4} />;
}

// ------------------------------------------------------------------ 1. Spread explorer

const START = [2, 3, 4, 5, 6, 7, 8];

function DeviationSquares({ pts, m }: { pts: number[]; m: number }) {
  const { x, y } = usePlot();
  const ppu = x(1) - x(0);
  return (
    <g>
      {pts.map((p, i) => {
        const side = Math.abs(p - m) * ppu;
        if (side < 0.5) return null;
        const left = Math.min(x(p), x(m));
        return <rect key={i} x={left} y={y(0) - side} width={side} height={side} fill="var(--c-b)" opacity={0.13} stroke="var(--c-b)" strokeOpacity={0.6} />;
      })}
    </g>
  );
}

export function SpreadExplorer() {
  const [pts, setPts] = useState<number[]>(START);
  const [showSquares, setShowSquares] = useState(true);
  const m = mean(pts);
  const v = variance(pts);
  const s = Math.sqrt(v);
  const med = median(pts);
  const mad = meanAbsDev(pts);
  const width = 480;
  const pad = 28;
  const ppu = (width - 8 - pad) / 16;
  const height = Math.round(pad + 8 + 10.7 * ppu);

  return (
    <Widget
      title="Spread: the average of the squares"
      wide
      caption={
        <>
          Seven measurements on a number line. Drag any of them. The triangle is the <strong>mean</strong> (the point where the line would balance), the tick is the <strong>median</strong> (the middle value). Each orange square has a side equal to one point’s distance from the mean; the <strong>variance</strong> is the average area of those squares. The dashed purple square has exactly that average area, and its side is the <strong>standard deviation</strong>. Press <strong>Make an outlier</strong> and compare what happens to the mean, the median and the SD.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <button type="button" className="btn" onClick={() => setPts([...START.slice(0, 6), 15])}>Make an outlier</button>
        <button type="button" className="btn" onClick={() => setPts(START)}>Reset</button>
        <button type="button" className="btn" onClick={() => setShowSquares((x) => !x)}>{showSquares ? 'Hide squares' : 'Show squares'}</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <Plot width={width} height={height + 20} x={[0, 16]} y={[-1.7, 9]} pad={pad} grid={false} axes={false} ariaLabel="Points on a number line with their squared deviations">
          {showSquares && <DeviationSquares pts={pts} m={m} />}
          {({ x, y }) =>
            showSquares ? (
              <rect x={x(m) - (s * (x(1) - x(0))) / 2} y={y(0) - s * (x(1) - x(0))} width={s * (x(1) - x(0))} height={s * (x(1) - x(0))} fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="5 4" />
            ) : null
          }
          {({ x, y }) => (
            <g>
              {[0, 2, 4, 6, 8, 10, 12, 14, 16].map((t) => (
                <g key={t}>
                  <line x1={x(t)} x2={x(t)} y1={y(0) - 3} y2={y(0) + 3} stroke="var(--c-axis)" />
                  <text x={x(t)} y={y(0) + 44} textAnchor="middle" fontSize={11} fill="var(--text-faint)">{t}</text>
                </g>
              ))}
              <polygon points={`${x(m)},${y(0) + 9} ${x(m) - 8},${y(0) + 22} ${x(m) + 8},${y(0) + 22}`} fill="var(--accent)" />
              <text x={x(m)} y={y(0) + 33} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="var(--accent-text)">mean</text>
              <line x1={x(med)} x2={x(med)} y1={y(0) - 22} y2={y(0) - 11} stroke="var(--c-good)" strokeWidth={3} />
              <text x={x(med)} y={y(0) - 26} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="var(--c-good)">median</text>
            </g>
          )}
          {pts.map((p, i) => (
            <Handle
              key={i}
              x={p}
              y={0}
              r={7}
              color="var(--c-a)"
              snap={0.25}
              label={`point ${i + 1}`}
              onDrag={(nx) => setPts((cur) => cur.map((q, k) => (k === i ? nx : q)))}
            />
          ))}
        </Plot>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">mean (balance point) </span>
            <span style={{ color: 'var(--accent-text)' }}>{m.toFixed(2)}</span>
            {'\n'}
            <span className="k">median (middle value) </span>
            <span style={{ color: 'var(--c-good)' }}>{med.toFixed(2)}</span>
            {'\n'}
            <span className="k">average distance from mean </span>
            {mad.toFixed(2)}
            {'\n'}
            <span className="k">variance (average square) </span>
            {v.toFixed(2)}
            {'\n'}
            <span className="k">standard deviation (its side) </span>
            {s.toFixed(2)}
          </div>
          <div className="live-note">
            {Math.max(...pts) >= 12 || Math.min(...pts) <= 0.5 ? (
              <>
                One far-away point drags the mean towards it and dominates the variance: its square is bigger than all the others put together. The median barely notices. That’s the trade-off from the loss-functions lesson: squaring makes big misses count for a lot.
              </>
            ) : (
              <>
                The squares are the deviations, squared. Points far from the mean get big squares, so the variance pays most attention to the extremes. Taking the square root gets back to the original units: an SD of {s.toFixed(2)} means “a typical point sits about {s.toFixed(1)} away from the mean”.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Bell-curve factory

export function BellFactory() {
  const [src, setSrc] = useState<SourceId>('die');
  const [n, setN] = useState(1);
  const [view, setView] = useState<'sum' | 'avg'>('sum');
  const one = SOURCES[src].dist;
  const mu1 = distMean(one);
  const s1 = distSd(one);
  const d = useMemo(() => sumOf(one, n), [one, n]);
  const gap = bellGap(d);
  const muN = distMean(d);
  const sN = distSd(d);
  const maxP = Math.max(...d.p);
  const scaleX = view === 'avg' ? 1 / n : 1;
  const lo = view === 'avg' ? Math.min(...[0, 1].map((k) => (one.min + k * (one.p.length - 1)))) - 0.5 : d.min - 0.5;
  const hi = view === 'avg' ? one.min + one.p.length - 1 + 0.5 : d.min + d.p.length - 0.5;
  // in the average view, bar heights stay as probabilities; the curve is the bell density × bar spacing
  const curveScale = view === 'avg' ? 1 / n : 1;

  return (
    <Widget
      title="The bell-curve factory"
      wide
      caption={
        <>
          Pick a source of randomness and add up <em>n</em> independent copies. The bars are the <strong>exact</strong> distribution of the total (computed, not sampled); the orange curve is a bell curve with the same mean and SD. Slide <em>n</em> from 1 to 30 for each source. Then switch to <strong>average</strong> and watch the spread shrink.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'end', marginBottom: 10 }}>
        <Segmented value={src} onChange={setSrc} options={(Object.keys(SOURCES) as SourceId[]).map((k) => ({ value: k, label: SOURCES[k].label }))} />
        <Segmented value={view} onChange={setView} options={[{ value: 'sum', label: 'sum' }, { value: 'avg', label: 'average' }]} />
      </div>
      <Slider label="how many copies, n" value={n} min={1} max={30} step={1} onChange={setN} format={(v) => String(v)} color="accent" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 6 }}>
        <Plot width={480} height={250} x={[lo, hi]} y={[0, maxP * 1.15]} xLabel={view === 'avg' ? 'average' : 'total'} ariaLabel="Exact distribution of a sum with a bell curve overlaid">
          <Bars d={d} scaleX={scaleX} />
          <Bell mu={view === 'avg' ? muN / n : muN} sigma={view === 'avg' ? sN / n : sN} scale={curveScale} />
        </Plot>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">one copy: mean </span>
            {mu1.toFixed(2)}
            <span className="k">, SD </span>
            {s1.toFixed(3)}
            {'\n'}
            <span className="k">sum of {n}: mean </span>
            {muN.toFixed(2)}
            <span className="k">, SD </span>
            {sN.toFixed(3)}
            <span className="k"> (= √{n} × {s1.toFixed(3)})</span>
            {'\n'}
            <span className="k">average of {n}: SD </span>
            {(sN / n).toFixed(3)}
            <span className="k"> (= {s1.toFixed(3)} / √{n})</span>
            {'\n'}
            <span className="k">probability in the wrong place vs the bell </span>
            {pct(gap)}
          </div>
          <div className="live-note">
            {n === 1 ? (
              <>One copy is just the source itself: {SOURCES[src].note}. Now add a second.</>
            ) : src === 'ticket' && gap > 0.05 ? (
              <>
                Still lopsided after {n} copies: {pct(gap)} of the probability is in the wrong place. Sums of lopsided things do become bells, but slowly. Even 30 long shots leave {pct(bellGap(sumOf(one, 30)))} out of place, while 5 dice get within {pct(bellGap(sumOf(SOURCES.die.dist, 5)))}.
              </>
            ) : (
              <>
                {n} copies: only {pct(gap)} of the probability is out of place. Nothing about the source was bell-shaped; adding independent pieces made it so, because extreme totals need everything to go the same way at once, and middling totals can happen in many ways. The SD of the sum grew only √{n} ≈ {Math.sqrt(n).toFixed(2)}×, not {n}×: highs and lows partly cancel.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Test-set noise

const NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000];

export function TestSetNoise() {
  const [ni, setNi] = useState(2);
  const n = NS[ni];
  const p = (OPENING.accA + OPENING.accB) / 2;
  const se = accuracySE(p, n);
  const look = probLooksBetter(p, n, OPENING.accA - OPENING.accB);
  // bucket the exact binomial into 0.25-point bins between 85% and 97%
  const bins = useMemo(() => {
    const b = binomial(n, p);
    // bins of a whole number of examples, so every bin covers the same number of possible scores
    const per = Math.ceil(0.0025 * n);
    const width = per / n;
    const k0 = Math.floor((0.85 * n) / per) * per;
    const lo = k0 / n;
    const out = new Array<number>(Math.ceil((0.97 - lo) / width)).fill(0);
    b.p.forEach((q, k) => {
      const idx = Math.floor((k - k0) / per);
      if (idx >= 0 && idx < out.length) out[idx] += q;
    });
    return { lo, width, out };
  }, [n, p]);
  const maxQ = Math.max(...bins.out);

  return (
    <Widget
      title="How much does a test score wobble?"
      wide
      caption={
        <>
          Suppose a model is truly right 90.9% of the time (halfway between the two models in the opening question). Score it on a fresh test set of <em>n</em> examples: the bars show the <strong>exact</strong> probability of each measured accuracy. Change the test-set size and watch the wobble shrink like 1/√n.
        </>
      }
    >
      <Slider label="test-set size n" value={ni} min={0} max={NS.length - 1} step={1} onChange={setNi} format={(i) => NS[i].toLocaleString('en-GB')} color="accent" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 6 }}>
        <Plot width={480} height={240} x={[85, 97]} y={[0, maxQ * 1.2]} xLabel="measured accuracy %" tickLabels ariaLabel="Distribution of measured accuracy">
          {({ x, y }) => (
            <g>
              <rect x={x((p - 2 * se) * 100)} y={y(maxQ * 1.2)} width={x((p + 2 * se) * 100) - x((p - 2 * se) * 100)} height={y(0) - y(maxQ * 1.2)} fill="color-mix(in srgb, var(--accent) 10%, transparent)" />
              {bins.out.map((q, i) =>
                q < 1e-7 ? null : (
                  <rect
                    key={i}
                    x={x((bins.lo + i * bins.width) * 100) + 0.5}
                    y={y(q)}
                    width={Math.max(1, x(bins.width * 100) - x(0) - 1)}
                    height={y(0) - y(q)}
                    fill="var(--c-a)"
                    opacity={0.75}
                  />
                ),
              )}
              {[OPENING.accA, OPENING.accB].map((a, k) => (
                <g key={a}>
                  <line x1={x(a * 100)} x2={x(a * 100)} y1={y(0)} y2={y(maxQ * 1.15)} stroke={k ? 'var(--c-b)' : 'var(--c-good)'} strokeWidth={2} />
                  <text x={x(a * 100) + (k ? -4 : 4)} y={y(maxQ * 1.1)} textAnchor={k ? 'end' : 'start'} fontSize={11.5} fill={k ? 'var(--c-b)' : 'var(--c-good)'}>
                    {k ? 'B 90.6' : 'A 91.2'}
                  </text>
                </g>
              ))}
            </g>
          )}
        </Plot>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">standard error √(p(1−p)/n) </span>
            {(se * 100).toFixed(2)} points
            {'\n'}
            <span className="k">95% of scores land within </span>±{(2 * se * 100).toFixed(2)} points
            {'\n'}
            <span className="k">two equally good models: a 0.6-point gap or more </span>
            {pct(2 * look, 0)} of the time
          </div>
          <div className="live-note">
            {n <= 1000 ? (
              <>
                With {n.toLocaleString('en-GB')} examples, a single score wobbles by about ±{(se * 100).toFixed(1)} points just from which examples happened to be in the test set. Two <em>identical</em> models would differ by 0.6 points or more {pct(2 * look, 0)} of the time. The opening gap tells you nothing.
              </>
            ) : look * 2 > 0.1 ? (
              <>
                {n.toLocaleString('en-GB')} examples: the wobble is down to ±{(se * 100).toFixed(2)} points, but equal models still show a 0.6-point gap {pct(2 * look, 0)} of the time. Quadrupling the test set only halves the wobble.
              </>
            ) : (
              <>
                With {n.toLocaleString('en-GB')} examples a 0.6-point gap between equal models happens only {pct(2 * look, 1)} of the time, so seeing one would start to mean something. That took a test set {Math.round(n / 500)}× bigger than the original.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: the dice workshop

const LAB_KEY = 'nl-stats-dice-lab-v1';
const PROGRESS_KEY = 'nl-stats-dice-progress-v1';
const ROLLS = 1000;
const ROLL_FRAMES = 20;
type Saved = Record<string, { r: Recipe; solved: boolean; optimal: boolean }>;
const emptyRecipe = (): Recipe => ({ counts: { d2: 0, d4: 0, d6: 0, d8: 0, d12: 0, d20: 0 }, plus: 0 });

function loadSaved(): Saved {
  const d: Saved = Object.fromEntries(WORKSHOP.map((b) => [b.id, { r: emptyRecipe(), solved: false, optimal: false }]));
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return d;
    const parsed = JSON.parse(raw) as Saved;
    for (const b of WORKSHOP) {
      const v = parsed[b.id];
      if (!v?.r?.counts) continue;
      const r = emptyRecipe();
      let ok = Number.isInteger(v.r.plus) && Math.abs(v.r.plus) <= 99;
      for (const die of DICE) {
        const c = v.r.counts[die.id];
        if (Number.isInteger(c) && c >= 0 && c <= MAX_EACH) r.counts[die.id] = c;
        else ok = false;
      }
      r.plus = v.r.plus;
      if (ok) d[b.id] = { r, solved: !!v.solved, optimal: !!v.optimal };
    }
  } catch {
    // ignore storage errors
  }
  return d;
}

const describeRecipe = (r: Recipe) => {
  const parts = DICE.filter((d) => r.counts[d.id]).map((d) => `${r.counts[d.id]}${d.id}`);
  if (r.plus) parts.push(`${r.plus > 0 ? '' : '−'}${Math.abs(r.plus)}`);
  return parts.join(' + ') || 'nothing';
};

export function DiceWorkshop() {
  const [saved, setSaved] = useState<Saved>(() => loadSaved());
  const [bi, setBi] = useState(0);
  const progress = useLabProgress(PROGRESS_KEY, () => ({
    solved: Object.fromEntries(WORKSHOP.map((b) => [b.id, saved[b.id].solved])),
    optimal: Object.fromEntries(WORKSHOP.map((b) => [b.id, saved[b.id].optimal])),
  }));
  const log = useMachineLog();
  const brief = WORKSHOP[bi];
  const entry = saved[brief.id];
  const r = entry.r;
  const check = checkRecipe(brief, r);
  const d = useMemo(() => (check.dice > 0 ? recipeDist(r) : null), [r, check.dice]);
  const solution = useMemo(() => (progress.solved[brief.id] || check.pass ? solveWorkshop(brief) : null), [brief, progress.solved, check.pass]);
  const optimal = !!solution && check.pass && check.dice === solution.fewest && Math.abs(check.gap - solution.best.gap) < 1e-12;

  // Roll: throw the recipe 1,000 times, 50 at a time, and watch the histogram grow towards the exact bars.
  const [rolls, setRolls] = useState<{ key: string; totals: number[]; shown: number } | null>(null);
  const recipeKey = `${brief.id}:${describeRecipe(r)}`;
  const rollSeed = useRef(1);
  const player = useFramePlayer(ROLL_FRAMES, 45, (i) => setRolls((cur) => (cur ? { ...cur, shown: Math.round((cur.totals.length * i) / ROLL_FRAMES) } : cur)));
  const roll = () => {
    const totals = rollRecipe(r, ROLLS, createRng(1000 + rollSeed.current++));
    setRolls({ key: recipeKey, totals, shown: 0 });
    player.start();
    const m = totals.reduce((a, b) => a + b, 0) / totals.length;
    const sd = Math.sqrt(totals.reduce((a, b) => a + (b - m) ** 2, 0) / totals.length);
    log.push(`rolled ${describeRecipe(r)} ${ROLLS.toLocaleString()} times: sample mean ${m.toFixed(2)}, SD ${sd.toFixed(2)} (exact ${check.mean.toFixed(2)}, ${check.sd.toFixed(2)}; a 1,000-throw mean typically wobbles by about ±${(check.sd / Math.sqrt(ROLLS)).toFixed(2)})`);
  };
  const live = rolls && rolls.key === recipeKey ? rolls : null;
  const sample = useMemo(() => {
    if (!live || !d) return null;
    const p = new Array<number>(d.p.length).fill(0);
    for (let i = 0; i < live.shown; i++) p[live.totals[i] - d.min] += 1;
    return { p: p.map((c) => c / Math.max(1, live.shown)), n: live.shown };
  }, [live, d]);

  const persist = (next: Saved) => {
    setSaved(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };
  const update = (nr: Recipe) => {
    const c = checkRecipe(brief, nr);
    log.push(recipeChangeLine(r, nr) + (c.dice ? ` · out of place ${pct(c.gap)}` : '') + (c.pass ? ' · meets the brief' : ''), c.pass ? 'good' : 'info');
    persist({ ...saved, [brief.id]: { ...entry, r: nr } });
  };
  const setCount = (id: DieId, delta: number) => {
    const c = Math.min(MAX_EACH, Math.max(0, r.counts[id] + delta));
    update({ ...r, counts: { ...r.counts, [id]: c } });
  };
  const setPlus = (delta: number) => update({ ...r, plus: Math.max(-99, Math.min(99, r.plus + delta)) });

  const tick = (ok: boolean) => <span style={{ color: ok ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>{ok ? '✓' : '✗'}</span>;
  const yMax = d ? Math.max(...d.p, ...(sample ? sample.p : [0])) * 1.15 : 1;

  return (
    <LabShell
      title="Lab: The Dice Workshop"
      caption={
        <>
          Build a random number generator out of dice (d2 is a coin that shows 1 or 2). Pick how many of each to roll and add up; in some briefs you may also add a constant. The distribution, its mean, its SD and how bell-shaped it is are all computed exactly as you go; <strong>Roll</strong> throws your recipe 1,000 times so you can watch real throws pile up towards the exact bars. Your recipes are saved in this browser.
        </>
      }
      briefs={WORKSHOP.map((b) => ({
        id: b.id,
        tab: b.title.split(' · ')[0],
        title: b.title,
        text: (
          <>
            {b.story}
            <div style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 6 }}>
              Mean exactly {b.mean} · SD from {b.sdLo} to {b.sdHi} · at most {pct(b.maxGap, 1)} of the probability out of place compared with a bell
            </div>
          </>
        ),
      }))}
      active={bi}
      onSelect={setBi}
      progress={progress}
      passing={check.pass}
      optimal={optimal}
      status={
        check.dice === 0
          ? 'Means add and variances add: that’s the whole trick. Each die’s mean and variance are listed next to it.'
          : check.pass
            ? optimal
              ? 'Meets the brief with the fewest possible dice and, among those, the most bell-like shape. That’s the exhaustive optimum.'
              : 'Meets the brief. Optional: can you do it with fewer dice — and among recipes with that few, find the most bell-shaped?'
            : !check.meanOk
              ? `The mean is ${check.mean.toFixed(2)}. Means add: each die contributes (sides + 1)/2.`
              : !check.sdOk
                ? `The SD is ${check.sd.toFixed(3)}. SDs don’t add — variances do. Add up the variances, then take the square root.`
                : `Right mean and spread, but ${pct(check.gap)} of the probability is out of place: too few dice to make a bell, or one die dominating the rest.`
      }
      reveal={
        solution
          ? {
              label: 'the exhaustive search',
              content: (
                <>
                  Every recipe with up to {MAX_EACH} of each die was tried (9⁶ = 531,441 combinations, with the constant set to hit the mean where allowed): <strong>{solution.passing.length}</strong> meet the brief. The fewest dice that work is <strong>{solution.fewest}</strong> ({solution.fewestRecipes.length} recipe{solution.fewestRecipes.length === 1 ? '' : 's'}), and the most bell-shaped of those is <strong>{describeRecipe(solution.best.r)}</strong> ({pct(solution.best.gap)} out of place).
                </>
              ),
            }
          : undefined
      }
      log={{ lines: log.lines, onClear: log.clear, empty: 'Change the recipe or press Roll: each change is logged here.' }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 8 }}>
          {DICE.map((die) => (
            <div key={die.id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="mono" style={{ width: 44, fontWeight: 600 }}>{die.id}</span>
              <button type="button" className="btn small" onClick={() => setCount(die.id, -1)} disabled={r.counts[die.id] === 0} aria-label={`fewer ${die.id}`}>−</button>
              <span className="mono" style={{ width: 22, textAlign: 'center' }}>{r.counts[die.id]}</span>
              <button type="button" className="btn small" onClick={() => setCount(die.id, 1)} disabled={r.counts[die.id] === MAX_EACH} aria-label={`more ${die.id}`}>+</button>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                mean {((die.sides + 1) / 2).toFixed(1)}, variance {((die.sides * die.sides - 1) / 12).toFixed(2)}
              </span>
            </div>
          ))}
          {brief.allowPlus && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="mono" style={{ width: 44, fontWeight: 600 }}>+ k</span>
              <button type="button" className="btn small" onClick={() => setPlus(-10)}>−10</button>
              <button type="button" className="btn small" onClick={() => setPlus(-1)}>−1</button>
              <span className="mono" style={{ width: 30, textAlign: 'center' }}>{r.plus}</span>
              <button type="button" className="btn small" onClick={() => setPlus(1)}>+1</button>
              <button type="button" className="btn small" onClick={() => setPlus(10)}>+10</button>
            </div>
          )}
          <div className="btn-row">
            <button type="button" className="btn" onClick={() => update(emptyRecipe())}>Clear</button>
            <button type="button" className="btn primary" onClick={roll} disabled={check.dice === 0 || player.running}>
              {player.running ? 'Rolling…' : `Roll ${ROLLS.toLocaleString()} times`}
            </button>
          </div>
          <div className="readout">
            <span className="k">recipe </span>
            {describeRecipe(r)}
            {'\n'}
            {tick(check.meanOk)} <span className="k">mean </span>
            {check.mean.toFixed(2)}
            <span className="k"> (want {brief.mean}{!brief.allowPlus && r.plus !== 0 ? ', no constants' : ''})</span>
            {'\n'}
            {tick(check.sdOk)} <span className="k">SD </span>
            {check.sd.toFixed(3)}
            <span className="k">
              {' '}
              (want {brief.sdLo}–{brief.sdHi})
            </span>
            {'\n'}
            {tick(check.bellOk && check.dice > 0)} <span className="k">out of place </span>
            {check.dice ? pct(check.gap) : '—'}
            <span className="k"> (want ≤ {pct(brief.maxGap, 1)})</span>
            {'\n'}
            <span className="k">dice </span>
            {check.dice}
          </div>
        </div>
        <div style={{ display: 'grid', gap: 10 }}>
          {d ? (
            <Plot width={440} height={230} x={[d.min - 0.5, d.min + d.p.length - 0.5]} y={[0, yMax]} xLabel="total" ariaLabel="Exact distribution of the recipe">
              <Bars d={d} color={check.pass ? 'var(--c-good)' : 'var(--c-a)'} />
              <Bell mu={check.mean} sigma={check.sd} />
              {sample && <SampleMarks d={d} p={sample.p} />}
            </Plot>
          ) : (
            <div className="live-note">Add some dice to see the distribution.</div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
            {check.pass ? <span className="status good">Meets the brief</span> : <span className="status neutral">Not yet</span>}
            {sample && (
              <span className="muted" style={{ fontSize: 12.5 }}>
                Dashes: the share of {sample.n.toLocaleString()} real throws landing on each total. Bars: the exact probabilities.
              </span>
            )}
          </div>
        </div>
      </div>
    </LabShell>
  );
}

/** Sampled shares as short horizontal dashes on top of the exact bars. */
function SampleMarks({ d, p }: { d: Dist; p: number[] }) {
  const { x, y } = usePlot();
  const w = Math.max(3, Math.abs(x(1) - x(0)) * 0.7);
  return (
    <g>
      {p.map((q, i) =>
        q <= 0 ? null : <rect key={i} x={x(d.min + i) - w / 2} y={y(q) - 1.5} width={w} height={3} rx={1.5} fill="var(--text)" />,
      )}
    </g>
  );
}
