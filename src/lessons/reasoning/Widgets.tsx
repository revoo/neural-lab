// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { createRng } from '../../lib/rng';
import { bestOfN, grpoStep, initPolicy, majority, passAtN, policyStats, pSingle, TIERS, tokensPerSample, type GrpoStats, type Tier } from './reasoning';

const NS = [1, 2, 4, 8, 16, 32, 64, 128];
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const TIER_OPTS = (['small', 'medium', 'large'] as Tier[]).map((t) => ({ value: t, label: TIERS[t].label }));

// ------------------------------------------------------------------ parallel search

export function SearchCurves() {
  const [tier, setTier] = useState<Tier>('medium');
  const [steps, setSteps] = useState(7);
  const [thinking, setThinking] = useState(400);
  const [trap, setTrap] = useState(0.2);
  const p = pSingle(tier, steps, thinking);
  const series = [
    { key: 'pass', label: 'exact checker (pass@N)', colour: 'var(--c-c)', f: (n: number) => passAtN(p, n) },
    { key: 'maj', label: 'majority vote', colour: 'var(--c-a)', f: (n: number) => majority(p, trap, n) },
    { key: 'rm', label: 'best-of-N, noisy judge', colour: 'var(--c-b)', f: (n: number) => bestOfN(p, n, 'noisy') },
  ];
  const W = 440;
  const H = 220;
  const P = { l: 40, r: 10, t: 10, b: 30 };
  const x = (n: number) => P.l + ((W - P.l - P.r) * Math.log2(n)) / 7;
  const y = (a: number) => H - P.b - (H - P.t - P.b) * a;
  const vals = series.map((s) => NS.map((n) => s.f(n)));
  const rmPeak = NS[vals[2].indexOf(Math.max(...vals[2]))];
  return (
    <Widget
      title="More samples: three ways to pick an answer"
      wide
      caption={
        <>
          One problem, answered N times by a stand-in model (described below), with three ways of choosing the final answer. Everything is computed exactly, not
          sampled. Try a hard problem with little room to think, then raise the trap: the share of wrong answers that agree on the same tempting mistake.
        </>
      }
    >
      <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fit, minmax(min(220px, 100%), 1fr))', marginBottom: 8 }}>
        <Segmented label="Model" options={TIER_OPTS} value={tier} onChange={(v) => setTier(v as Tier)} />
        <Slider label="Steps the problem needs" value={steps} min={1} max={10} step={1} onChange={(v) => setSteps(Math.round(v))} format={(v) => String(Math.round(v))} />
        <Slider label="Thinking tokens per answer" value={thinking} min={0} max={400} step={40} onChange={(v) => setThinking(Math.round(v / 40) * 40)} format={(v) => String(Math.round(v))} />
        <Slider label="Wrong answers that agree on the trap" value={trap} min={0} max={0.9} step={0.1} onChange={setTrap} format={(v) => `${Math.round(100 * v)}%`} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, minWidth: 300, display: 'block' }} role="img" aria-label="Accuracy against number of samples">
          {[0, 0.25, 0.5, 0.75, 1].map((a) => (
            <g key={a}>
              <line className="plot-grid" x1={P.l} x2={W - P.r} y1={y(a)} y2={y(a)} />
              <text className="plot-tick" x={P.l - 5} y={y(a) + 3.5} textAnchor="end">
                {Math.round(100 * a)}%
              </text>
            </g>
          ))}
          {NS.map((n) => (
            <text key={n} className="plot-tick" x={x(n)} y={H - P.b + 15} textAnchor="middle">
              {n}
            </text>
          ))}
          <text className="plot-tick" x={(W + P.l) / 2} y={H - 2} textAnchor="middle">
            samples per problem, N
          </text>
          <line x1={P.l} x2={W - P.r} y1={y(p)} y2={y(p)} stroke="var(--text-faint)" strokeDasharray="3 3" />
          <text className="plot-tick" x={W - P.r} y={y(p) - 4} textAnchor="end">
            one answer: {pct(p)}
          </text>
          {series.map((s, i) => (
            <g key={s.key}>
              <path d={NS.map((n, j) => `${j ? 'L' : 'M'}${x(n).toFixed(1)},${y(vals[i][j]).toFixed(1)}`).join('')} fill="none" stroke={s.colour} strokeWidth={2.4} />
              {NS.map((n, j) => (
                <circle key={n} cx={x(n)} cy={y(vals[i][j])} r={2.6} fill={s.colour} />
              ))}
            </g>
          ))}
        </svg>
      </div>
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12.5 }}>
        {series.map((s, i) => (
          <span key={s.key}>
            <span style={{ display: 'inline-block', width: 14, height: 3, background: s.colour, verticalAlign: 'middle', marginRight: 5 }} />
            {s.label}: <span className="mono">{pct(vals[i][3])}</span> at N = 8
          </span>
        ))}
      </div>
      <p className="live-note">
        One answer is right {pct(p)} of the time. With an exact checker, 128 tries find a right one {pct(vals[0][7])} of the time. Majority vote ends at{' '}
        {pct(vals[1][7])}
        {vals[1][7] < p ? ': worse than a single answer, because the trap is more common than the truth.' : '.'} The noisy judge peaks at N = {rmPeak} (
        {pct(Math.max(...vals[2]))}) and then declines to {pct(vals[2][7])}: the more samples it sees, the more likely one of them is a wrong answer it loves.
      </p>
    </Widget>
  );
}

// ------------------------------------------------------------------ GRPO, live

const STEPS = 600;
export function GrpoTrainer() {
  const [group, setGroup] = useState(8);
  const [seed, setSeed] = useState(1);
  const r = TIERS.medium.r;
  const run = useMemo(() => {
    const pol = initPolicy();
    const rng = createRng(seed);
    const hist: GrpoStats[] = [policyStats(pol, r, 0)];
    for (let s = 1; s <= STEPS; s++) {
      grpoStep(pol, r, rng.next, group);
      if (s % 10 === 0) hist.push(policyStats(pol, r, s));
    }
    return hist;
  }, [group, seed, r]);
  const [frame, setFrame] = useState(run.length);
  const raf = useRef<number | null>(null);
  useEffect(() => setFrame(run.length), [run]);
  useEffect(() => () => void (raf.current && cancelAnimationFrame(raf.current)), []);
  const play = () => {
    let f = 1;
    const tick = () => {
      setFrame(f);
      if (f++ < run.length) raf.current = requestAnimationFrame(tick);
    };
    tick();
  };
  const h = run[Math.max(0, frame - 1)];
  const W = 420;
  const H = 170;
  const P = { l: 36, r: 36, t: 10, b: 26 };
  const x = (s: number) => P.l + ((W - P.l - P.r) * s) / STEPS;
  const yA = (a: number) => H - P.b - (H - P.t - P.b) * a;
  const yL = (l: number) => H - P.b - ((H - P.t - P.b) * l) / 10;
  const shown = run.slice(0, frame);
  const ceiling = Array.from({ length: 10 }, (_, i) => r ** (i + 1) / 10).reduce((a, b) => a + b);
  return (
    <Widget
      title="GRPO, live: rewarded only for right answers, the policy learns to think"
      wide
      caption={
        <>
          A policy chooses how many reasoning steps to write before answering, separately for problems of each difficulty, starting out keen to answer at once. For
          each difficulty it samples a group of attempts; a checker scores each 1 or 0; each attempt’s advantage is its score minus the group’s mean, divided by the
          group’s spread. Watch <span className="c-a">accuracy</span> and <span className="c-b">steps written</span> climb together. Press Replay; change the group
          size and the seed.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <Segmented label="Group size" options={[4, 8, 16]} value={group} onChange={(v) => setGroup(v as number)} />
        <Segmented label="Seed" options={[1, 2, 3]} value={seed} onChange={(v) => setSeed(v as number)} />
        <button className="btn" onClick={play}>
          ▶ Replay
        </button>
      </div>
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ flex: '1 1 300px', maxWidth: W, display: 'block' }} role="img" aria-label="Accuracy and steps written over training">
          {[0, 0.5, 1].map((a) => (
            <g key={a}>
              <line className="plot-grid" x1={P.l} x2={W - P.r} y1={yA(a)} y2={yA(a)} />
              <text className="plot-tick" x={P.l - 4} y={yA(a) + 3.5} textAnchor="end">
                {Math.round(100 * a)}%
              </text>
              <text className="plot-tick" x={W - P.r + 4} y={yA(a) + 3.5}>
                {10 * a}
              </text>
            </g>
          ))}
          <line x1={P.l} x2={W - P.r} y1={yA(ceiling)} y2={yA(ceiling)} stroke="var(--c-a)" strokeDasharray="3 3" opacity={0.6} />
          <text className="plot-tick" x={P.l + 4} y={yA(ceiling) - 4}>
            best possible {pct(ceiling)}
          </text>
          <path d={shown.map((s, i) => `${i ? 'L' : 'M'}${x(s.step).toFixed(1)},${yA(s.acc).toFixed(1)}`).join('')} fill="none" stroke="var(--c-a)" strokeWidth={2.4} />
          <path d={shown.map((s, i) => `${i ? 'L' : 'M'}${x(s.step).toFixed(1)},${yL(s.meanLen).toFixed(1)}`).join('')} fill="none" stroke="var(--c-b)" strokeWidth={2.4} />
          <text className="plot-tick" x={(W + P.l - P.r) / 2} y={H - 4} textAnchor="middle">
            training steps (0 to {STEPS})
          </text>
        </svg>
        <div style={{ flex: '1 1 200px', maxWidth: 260 }}>
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginBottom: 4 }}>Average steps written, by how many the problem needs</div>
          {h.lenByDiff.map((l, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, marginBottom: 2 }}>
              <span className="mono" style={{ width: 18, textAlign: 'right' }}>
                {i + 1}
              </span>
              <div style={{ flex: 1, position: 'relative', height: 10, background: 'var(--bg-sunken)', borderRadius: 3 }}>
                <div style={{ width: `${(100 * l) / 12}%`, height: 10, background: 'var(--c-b)', borderRadius: 3 }} />
                <div style={{ position: 'absolute', left: `${(100 * (i + 1)) / 12}%`, top: -2, width: 2, height: 14, background: 'var(--text)' }} />
              </div>
            </div>
          ))}
          <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>tick: the steps the problem needs</div>
        </div>
      </div>
      <p className="live-note">
        Step {h.step}: accuracy {pct(h.acc)} (best possible {pct(ceiling)}), {h.meanLen.toFixed(2)} steps written on average. Easy problems learn first; the hardest
        lag because a group where every attempt fails has zero spread and teaches nothing: there is no better attempt to push towards.
      </p>
    </Widget>
  );
}

// ------------------------------------------------------------------ iso-compute

export function IsoCompute() {
  const [mode, setMode] = useState<'majority' | 'exact'>('majority');
  const large = (d: number) => pSingle('large', d, 400);
  const smallN = Math.floor((TIERS.large.params * tokensPerSample(400)) / (TIERS.small.params * tokensPerSample(400)));
  const N = 64;
  const small = (d: number) => {
    const p = pSingle('small', d, 400);
    return mode === 'majority' ? majority(p, 0.4, N) : passAtN(p, N);
  };
  const rows = Array.from({ length: 10 }, (_, i) => ({ d: i + 1, L: large(i + 1), S: small(i + 1) }));
  const cross = rows.find((r) => r.L > r.S)?.d;
  return (
    <Widget
      title="Same compute, two ways to spend it"
      caption={
        <>
          The large model answers once; the small model, 70 times smaller, answers {N} times (about the same arithmetic: {smallN} small answers would cost exactly one
          large one). Both think for 400 tokens. Bars show the chance of solving a problem of each difficulty. Switch between voting and an exact checker.
        </>
      }
    >
      <Segmented label="How the small model picks" options={[{ value: 'majority', label: 'majority vote' }, { value: 'exact', label: 'exact checker' }]} value={mode} onChange={(v) => setMode(v as 'majority' | 'exact')} />
      <div style={{ marginTop: 10 }}>
        {rows.map((r) => (
          <div key={r.d} style={{ display: 'grid', gridTemplateColumns: '58px 1fr', gap: 6, alignItems: 'center', marginBottom: 4, fontSize: 12 }}>
            <span className="mono">{r.d} step{r.d > 1 ? 's' : ''}</span>
            <div>
              <div style={{ height: 8, width: `${100 * r.L}%`, background: 'var(--c-b)', borderRadius: 2, marginBottom: 2 }} />
              <div style={{ height: 8, width: `${100 * r.S}%`, background: 'var(--c-a)', borderRadius: 2 }} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 14, fontSize: 12.5, flexWrap: 'wrap' }}>
        <span>
          <span style={{ display: 'inline-block', width: 12, height: 8, background: 'var(--c-b)', marginRight: 4 }} />
          large model, one answer
        </span>
        <span>
          <span style={{ display: 'inline-block', width: 12, height: 8, background: 'var(--c-a)', marginRight: 4 }} />
          small model, {N} answers
        </span>
      </div>
      <p className="live-note">
        {mode === 'majority'
          ? `Voting, the small model wins up to ${(cross ?? 11) - 1} steps and the large model from ${cross} steps on. On hard problems the small model is right less often than the tempting trap answer appears, so more votes entrench the mistake.`
          : 'With an exact checker the small model wins at every difficulty here, because even on 10-step problems about one of its answers in ten is right, and 64 tries almost always include one. Real hard problems can have a success rate so close to zero that no number of tries helps, and real checkers only exist for some tasks.'}
      </p>
    </Widget>
  );
}
