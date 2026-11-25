// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { AnswerReveal } from '../../components/AnswerReveal';
import { LiveCode } from '../../components/LabJob';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
import {
  GRADERS,
  GRADER_BY_ID,
  LAB_ALPHAS,
  LAB_BRIEFS,
  LAB_MODES,
  LAB_NS,
  briefPasses,
  evalLabHint,
  evalPowerCurve,
  evaluateEvalDesign,
  judgeModel,
  labExhaustive,
  pairedComparison,
  winnersCurse,
  type EvalBrief,
  type EvalDesign,
  type EvalMode,
  type GraderId,
  type JudgeProtocol,
} from './eval';
import { evalSuiteRealCode } from './realCode';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;
const pct2 = (v: number) => `${(100 * v).toFixed(2)}%`;

// ------------------------------------------------------------------ 1. Paired vs Unpaired

export function PairedComparisonWidget() {
  const [n, setN] = useState(100);
  const [kA, setKA] = useState(84);
  const [kB, setKB] = useState(78);
  const [n11, setN11] = useState(78);

  const clampedKA = Math.min(kA, n);
  const clampedKB = Math.min(kB, n);
  const min11 = Math.max(0, clampedKA + clampedKB - n);
  const max11 = Math.min(clampedKA, clampedKB);
  const clamped11 = Math.max(min11, Math.min(max11, n11));

  const res = useMemo(() => pairedComparison(n, clampedKA, clampedKB, clamped11), [n, clampedKA, clampedKB, clamped11]);

  const applyPreset = (pn: number, pa: number, pb: number, p11: number) => {
    setN(pn);
    setKA(pa);
    setKB(pb);
    setN11(p11);
  };

  const maxBar = Math.max(...res.coinPmf, 0.05);

  return (
    <Widget
      title="Paired vs unpaired: the same top-line scores, totally different evidence"
      wide
      caption={
        <>
          Keep Model A at <strong>84%</strong> and Model B at <strong>78%</strong> on 100 questions, and drag the <strong>both-right overlap</strong> slider. On two separate test sets (unpaired), p = 0.321 — indistinguishable from luck. On the <em>same</em> 100 questions, only the disagreements count: 6 pure wins and 0 regressions gives McNemar p = 0.031, whereas 18 wins and 12 regressions gives p = 0.362.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button type="button" className="btn small" onClick={() => applyPreset(100, 84, 78, 78)}>
          Pure improvement (6–0)
        </button>
        <button type="button" className="btn small" onClick={() => applyPreset(100, 84, 78, 77)}>
          Almost clean (7–1)
        </button>
        <button type="button" className="btn small" onClick={() => applyPreset(100, 84, 78, 66)}>
          Trading wins (18–12)
        </button>
        <button type="button" className="btn small" onClick={() => applyPreset(50, 44, 41, 41)}>
          Small eval (44/50 vs 41/50)
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 8 }}>
          <Segmented
            label="Eval size"
            value={n}
            onChange={(nextN) => {
              const scale = nextN / n;
              const na = Math.round(clampedKA * scale);
              const nb = Math.round(clampedKB * scale);
              const n1 = Math.round(clamped11 * scale);
              setN(nextN);
              setKA(na);
              setKB(nb);
              setN11(n1);
            }}
            options={[50, 100, 200].map((v) => ({ value: v, label: `${v} items` }))}
          />
          <Slider label="Model A right" value={clampedKA} min={Math.round(n * 0.5)} max={n} step={1} onChange={setKA} format={(v) => `${v}/${n} (${pct1(v / n)})`} color="c-a" />
          <Slider label="Model B right" value={clampedKB} min={Math.round(n * 0.5)} max={n} step={1} onChange={setKB} format={(v) => `${v}/${n} (${pct1(v / n)})`} color="c-b" />
          <Slider
            label="Both right (overlap)"
            value={clamped11}
            min={min11}
            max={max11}
            step={1}
            onChange={setN11}
            format={(v) => `${v} (${res.disagreements} disagreements)`}
            color="c-c"
          />

          <div style={{ overflowX: 'auto' }}>
            <table className="num-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Paired 2×2</th>
                  <th>B right</th>
                  <th>B wrong</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: 600 }}>A right</td>
                  <td style={{ color: 'var(--text-muted)' }}>{res.n11} (agree)</td>
                  <td style={{ color: 'var(--c-a)', fontWeight: 700 }}>{res.n10} (A wins)</td>
                </tr>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: 600 }}>A wrong</td>
                  <td style={{ color: 'var(--c-b)', fontWeight: 700 }}>{res.n01} (B wins)</td>
                  <td style={{ color: 'var(--text-muted)' }}>{res.n00} (agree)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">headline gap (A − B) </span>
            {res.gap >= 0 ? '+' : ''}
            {(res.gap * 100).toFixed(1)} pp ({res.kA} vs {res.kB} of {res.n})
            {'\n'}
            <span className="k">disagreements (A only / B only) </span>
            <span style={{ color: 'var(--c-a)', fontWeight: 650 }}>{res.n10}</span> vs{' '}
            <span style={{ color: 'var(--c-b)', fontWeight: 650 }}>{res.n01}</span> (total {res.disagreements})
            {'\n'}
            <span className="k">unpaired exact p-value </span>
            <span style={{ color: res.unpairedP < 0.05 ? 'var(--c-good)' : 'var(--text-muted)', fontWeight: 650 }}>{res.unpairedP.toFixed(4)}</span>
            {'\n'}
            <span className="k">paired McNemar exact p-value </span>
            <span style={{ color: res.mcnemarP < 0.05 ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>{res.mcnemarP.toFixed(4)}</span>
          </div>

          <Plot
            width={420}
            height={195}
            x={[-0.6, Math.max(res.disagreements, 1) + 0.6]}
            y={[0, maxBar * 1.18]}
            pad={32}
            xLabel="A-only wins (out of disagreements)"
            yLabel="P(under tie)"
            ariaLabel="Fair-coin null distribution of A-only wins among disagreements"
          >
            {({ x, y }) => {
              const m = res.disagreements;
              if (m === 0) {
                return (
                  <text x={x(0.5)} y={y(maxBar * 0.5)} textAnchor="middle" fontSize={12} fill="var(--text-muted)">
                    Zero disagreements (models agreed on every item)
                  </text>
                );
              }
              const bw = Math.max(3, Math.min(26, (x(1) - x(0)) * 0.72));
              const hi = Math.max(res.n10, res.n01);
              const lo = Math.min(res.n10, res.n01);
              return (
                <>
                  {res.coinPmf.map((prob, k) => {
                    const inTail = k >= hi || k <= lo;
                    const isObs = k === res.n10;
                    const fill = isObs ? 'var(--c-a)' : inTail ? 'var(--c-b)' : 'var(--text-muted)';
                    return (
                      <rect
                        key={k}
                        x={x(k) - bw / 2}
                        y={y(prob)}
                        width={bw}
                        height={Math.max(0, y(0) - y(prob))}
                        fill={fill}
                        opacity={isObs ? 1 : inTail ? 0.65 : 0.35}
                        rx={2}
                      />
                    );
                  })}
                </>
              );
            }}
          </Plot>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Bars show the exact fair-coin distribution across the {res.disagreements} disagreements if A and B were equally good. Highlighted tails sum to McNemar p = {res.mcnemarP.toFixed(4)}.
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {res.disagreements === 0
          ? 'Both models gave the exact same pass/fail verdict on every question — zero evidence of a difference.'
          : res.mcnemarP < 0.05 && res.unpairedP >= 0.05
            ? `Same ${res.kA}% vs ${res.kB}% headline — yet unpaired says noise (p = ${res.unpairedP.toFixed(3)}) while paired McNemar flags a real signal (p = ${res.mcnemarP.toFixed(3)}) because ${res.n10} vs ${res.n01} disagreements is lopsided.`
            : res.mcnemarP >= 0.05 && res.disagreements > Math.abs(res.kA - res.kB) * 2
              ? `With ${res.n10} wins and ${res.n01} losses, the two models are mostly trading mistakes on different questions — paired p = ${res.mcnemarP.toFixed(3)} warns you this gap could easily be luck.`
              : `Unpaired p = ${res.unpairedP.toFixed(3)}, paired McNemar p = ${res.mcnemarP.toFixed(3)}. Slide “Both right” up and down to watch how question overlap changes your statistical sharpness without moving either top-line accuracy.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Winner's Curse (Eval Overfitting)

export function EvalOverfitWidget() {
  const [n, setN] = useState(50);
  const [K, setK] = useState(20);
  const [p, setP] = useState(0.8);

  const res = useMemo(() => winnersCurse(n, p, K), [n, p, K]);

  // Visible x-range around [p - 4 SE, 1.0]
  const se = Math.sqrt((p * (1 - p)) / n);
  const xLo = Math.max(0, Math.floor((p - 3.5 * se) * 20) / 20);
  const xHi = Math.min(1, Math.ceil((Math.min(1, res.expectedMax + 3 * se)) * 20) / 20);
  const yMax = Math.max(...res.single, ...res.maxPmf, 0.05);

  return (
    <Widget
      title="The winner’s curse: tuning K identical prompts on the same eval set"
      wide
      caption={
        <>
          Suppose you test <strong>K</strong> prompt tweaks on an <strong>n</strong>-question eval set and keep the highest scorer — when in truth <em>none</em> of them improved accuracy above <strong>p</strong>. The curves are exact binomial distributions (not simulations). Try <code>n = 50, K = 20, p = 80%</code>, then raise <code>n</code> to 500.
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 10 }}>
          <Segmented
            label="Eval set size"
            value={n}
            onChange={setN}
            options={[25, 50, 100, 200, 500].map((v) => ({ value: v, label: String(v) }))}
          />
          <Slider label="prompts tried (K)" value={K} min={1} max={100} step={1} onChange={setK} format={(v) => `${v} prompt${v === 1 ? '' : 's'}`} color="c-b" />
          <Slider label="true accuracy of every prompt" value={p} min={0.6} max={0.95} step={0.05} onChange={setP} format={pct1} color="c-a" />

          <div className="readout">
            <span className="k">true accuracy (held-out score) </span>
            <span style={{ color: 'var(--c-a)', fontWeight: 650 }}>{pct1(p)}</span>
            {'\n'}
            <span className="k">expected score of best of {K} </span>
            <span style={{ color: 'var(--c-b)', fontWeight: 700 }}>{pct1(res.expectedMax)}</span>
            {'\n'}
            <span className="k">average mirage (overfit gap) </span>
            <strong>+{(res.expectedMirage * 100).toFixed(1)} pp</strong>
            {'\n'}
            <span className="k">chance the “winner” looks ≥ +5 pp better </span>
            {pct1(res.probPlus5)}
          </div>
        </div>

        <div>
          <Plot
            width={440}
            height={250}
            x={[xLo, Math.max(xLo + 0.1, xHi)]}
            y={[0, yMax * 1.15]}
            pad={36}
            xLabel="measured eval accuracy"
            yLabel="probability"
            ariaLabel="Exact distribution of single prompt score vs maximum of K prompts"
          >
            {({ x, y }) => {
              const pathOf = (pmf: number[]) => {
                const pts: string[] = [];
                for (let k = 0; k <= n; k++) {
                  const acc = k / n;
                  if (acc < xLo - 1 / n || acc > xHi + 1 / n) continue;
                  pts.push(`${pts.length === 0 ? 'M' : 'L'}${x(acc).toFixed(1)},${y(pmf[k]).toFixed(1)}`);
                }
                return pts.join(' ');
              };
              return (
                <>
                  <line x1={x(p)} x2={x(p)} y1={y(0)} y2={y(yMax * 1.1)} stroke="var(--text-muted)" strokeWidth={1.3} strokeDasharray="4 4" />
                  <path d={pathOf(res.single)} fill="none" stroke="var(--c-a)" strokeWidth={2.2} />
                  {K > 1 && <path d={pathOf(res.maxPmf)} fill="none" stroke="var(--c-b)" strokeWidth={2.6} />}
                  <line x1={x(res.expectedMax)} x2={x(res.expectedMax)} y1={y(0)} y2={y(yMax * 1.1)} stroke="var(--c-b)" strokeWidth={1.3} strokeDasharray="2 3" />
                  <text x={x(p) + 4} y={y(yMax * 1.06)} fontSize={10.5} fill="var(--text-muted)">
                    truth {pct1(p)}
                  </text>
                </>
              );
            }}
          </Plot>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            <span>
              <span style={{ color: 'var(--c-a)', fontWeight: 700 }}>━</span> Single prompt
            </span>
            {K > 1 && (
              <span>
                <span style={{ color: 'var(--c-b)', fontWeight: 700 }}>━</span> Best of {K} prompts (avg {pct1(res.expectedMax)})
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {K === 1
          ? `With K = 1 (one untouched evaluation), there is no selection bias: the score averages ${pct1(p)}. Now drag K up to 20 to simulate an afternoon of prompt tweaking on the same ${n} questions.`
          : `Even though every one of the ${K} prompts is truly ${pct1(p)}, picking the highest scorer on ${n} questions hands you an average “improvement” of +${(res.expectedMirage * 100).toFixed(1)} percentage points — and a ${pct1(res.probPlus5)} chance of looking at least +5 points better. On a locked held-out test set, it immediately drops back to ${pct1(p)}.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. LLM-as-a-Judge & Position Bias

export function JudgeBiasWidget() {
  const [trueEdge, setTrueEdge] = useState(0); // trueWinA - trueWinB in [-0.2, 0.2]
  const [q, setQ] = useState(0.85);
  const [bias, setBias] = useState(0.2);
  const [protocol, setProtocol] = useState<JudgeProtocol>('a-first');

  const trueWinA = 0.25 + trueEdge / 2;
  const trueWinB = 0.25 - trueEdge / 2;
  const model = useMemo(() => judgeModel(trueWinA, trueWinB, q, bias), [trueWinA, trueWinB, q, bias]);

  const active = protocol === 'a-first' ? model.aFirst : protocol === 'b-first' ? model.bFirst : model.swapped;

  const Bar = ({ label, a, b, tieOrFlip, tieLabel }: { label: string; a: number; b: number; tieOrFlip: number; tieLabel: string }) => (
    <div style={{ display: 'grid', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5 }}>
        <span style={{ fontWeight: 600 }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>
          A {pct1(a)} · {tieLabel} {pct1(tieOrFlip)} · B {pct1(b)}
        </span>
      </div>
      <div style={{ display: 'flex', height: 22, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
        <div style={{ width: `${a * 100}%`, background: 'var(--c-a)', transition: 'width 150ms' }} title={`A wins: ${pct1(a)}`} />
        <div style={{ width: `${tieOrFlip * 100}%`, background: 'var(--border)', transition: 'width 150ms' }} title={`${tieLabel}: ${pct1(tieOrFlip)}`} />
        <div style={{ width: `${b * 100}%`, background: 'var(--c-b)', transition: 'width 150ms' }} title={`B wins: ${pct1(b)}`} />
      </div>
    </div>
  );

  return (
    <Widget
      title="LLM-as-a-judge: how position bias invents victories (and how swapping cancels it)"
      wide
      caption={
        <>
          No LLM runs here: this is an explicit probabilistic judge model (accuracy <strong>q</strong> on clear wins, Slot-1 bias <strong>b</strong> favouring whichever answer is shown first: on ties Slot 1 wins with 50% + b, on clear wins Slot 1 accuracy is q + b/2 and Slot 2 is q − b/2). Leave true advantage at 0 (two identical models), set bias to 0.20, and switch between A-first, B-first and Position-swapped.
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 10 }}>
          <Slider
            label="true advantage of A over B"
            value={trueEdge}
            min={-0.2}
            max={0.2}
            step={0.05}
            onChange={setTrueEdge}
            format={(v) => (Math.abs(v) < 1e-9 ? '0 pp (equal models)' : `${v > 0 ? '+' : ''}${(v * 100).toFixed(0)} pp`)}
            color="c-a"
          />
          <Slider label="judge accuracy on clear wins" value={q} min={0.6} max={0.95} step={0.05} onChange={setQ} format={pct1} color="c-c" />
          <Slider label="Slot-1 position bias" value={bias} min={0} max={0.35} step={0.05} onChange={setBias} format={(v) => `+${(v * 100).toFixed(0)} pp to Slot 1`} color="c-b" />
          <Segmented
            label="Judging protocol"
            value={protocol}
            onChange={setProtocol}
            options={[
              { value: 'a-first', label: 'A always first' },
              { value: 'b-first', label: 'B always first' },
              { value: 'swapped', label: 'Position-swapped (2×)' },
            ]}
          />
        </div>

        <div style={{ display: 'grid', gap: 12 }}>
          <Bar label="Ground truth (what a perfect oracle sees)" a={model.trueWinA} b={model.trueWinB} tieOrFlip={model.tie} tieLabel="Tie" />
          <Bar
            label={
              protocol === 'a-first'
                ? 'Reported by judge (A in Slot 1)'
                : protocol === 'b-first'
                  ? 'Reported by judge (B in Slot 1)'
                  : 'Reported by judge (both orders must agree)'
            }
            a={active.winA}
            b={active.winB}
            tieOrFlip={active.flip}
            tieLabel="Flip/tie"
          />
          <div className="readout">
            <span className="k">true net margin (A − B) </span>
            {trueEdge >= 0 ? '+' : ''}
            {(trueEdge * 100).toFixed(1)} pp
            {'\n'}
            <span className="k">reported net margin (A − B) </span>
            <span style={{ color: Math.abs(active.netA - trueEdge) > 0.08 ? 'var(--c-bad)' : 'var(--c-good)', fontWeight: 700 }}>
              {active.netA >= 0 ? '+' : ''}
              {(active.netA * 100).toFixed(1)} pp
            </span>
            {'\n'}
            <span className="k">swapped inconsistency (flip rate) </span>
            {pct1(model.swapped.flip)} (calls: {protocol === 'swapped' ? '2×' : '1×'})
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {protocol === 'swapped'
          ? Math.abs(trueEdge) < 1e-9
            ? `Position bias cancelled to zero: A and B both win exactly ${pct1(model.swapped.winA)}, regardless of the +${(bias * 100).toFixed(0)} pp Slot-1 bias. Better yet, the ${pct1(model.swapped.flip)} flip rate warns you how often the judge contradicted itself when the order changed.`
            : `Swapping recovers the right direction (A ${pct1(model.swapped.winA)} vs B ${pct1(model.swapped.winB)}) cleanly without bias, at the cost of 2× judge calls. Notice the ${pct1(model.swapped.flip)} flip rate diagnostic.`
          : Math.abs(trueEdge) < 1e-9 && bias > 0
            ? `A and B are literally identical — yet putting ${protocol === 'a-first' ? 'A' : 'B'} in Slot 1 hands it a fake ${pct1(protocol === 'a-first' ? active.winA : active.winB)} vs ${pct1(protocol === 'a-first' ? active.winB : active.winA)} victory! Switch to “Position-swapped” to watch the bias vanish.`
            : trueEdge > 0 && active.netA < 0
              ? `Position bias flipped the winner! Model A is truly +${(trueEdge * 100).toFixed(0)} pp better, yet placing B in Slot 1 makes the judge award B ${pct1(active.winB)} to ${pct1(active.winA)}.`
              : `Single-order judging mixes true quality with Slot-1 bias (+${(bias * 100).toFixed(0)} pp). Flip between “A always first” and “B always first” to see how much the scoreboard swings just from swapping the prompt order.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Size Your Eval Suite

const LAB_KEY = 'nl-eval-lab-v1';

type LabEntry = {
  design: EvalDesign;
  solved: boolean;
  bestCost: number | null;
};

type LabState = Record<EvalBrief['id'], LabEntry>;

const START_DESIGN: EvalDesign = { mode: 'unpaired', grader: 'human', n: 100, alpha: 0.05 };

function loadLab(): LabState {
  const fresh: LabState = {
    release: { design: { ...START_DESIGN }, solved: false, bestCost: null },
    'tight-set': { design: { ...START_DESIGN }, solved: false, bestCost: null },
    'fine-gain': { design: { ...START_DESIGN }, solved: false, bestCost: null },
  };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return fresh;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of LAB_BRIEFS) {
      const e = parsed[b.id];
      if (!e || !e.design) continue;
      const d = e.design;
      if (
        !LAB_MODES.includes(d.mode) ||
        !(d.grader in GRADER_BY_ID) ||
        !(LAB_NS as readonly number[]).includes(d.n) ||
        !(LAB_ALPHAS as readonly number[]).includes(d.alpha)
      )
        continue;
      const clampedN = b.maxN !== undefined ? Math.min(d.n, b.maxN) : d.n;
      fresh[b.id] = {
        design: { ...d, n: clampedN },
        solved: !!e.solved,
        bestCost: typeof e.bestCost === 'number' ? e.bestCost : null,
      };
    }
  } catch {
    // ignore storage errors
  }
  return fresh;
}

const GRADER_COLOR: Record<GraderId, string> = {
  human: 'var(--c-a)',
  judge2: 'var(--c-b)',
  judge1: 'var(--c-c)',
};

export function EvalSizingLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [showHint, setShowHint] = useState(false);
  const log = useMachineLog(12);

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const allowedNs: number[] = useMemo(
    () => (brief.maxN !== undefined ? LAB_NS.filter((x) => x <= brief.maxN!) : [...LAB_NS]),
    [brief.maxN],
  );
  const design: EvalDesign = allowedNs.includes(entry.design.n)
    ? entry.design
    : { ...entry.design, n: allowedNs[allowedNs.length - 1] };
  const ev = useMemo(() => evaluateEvalDesign(design, brief), [design, brief]);
  const pass = briefPasses(brief, design, ev);
  const curve = useMemo(() => evalPowerCurve(design, brief), [design, brief]);

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
  const update = (patch: Partial<EvalDesign>) => {
    const nextDesign = { ...design, ...patch };
    persist({ ...lab, [brief.id]: { ...entry, design: nextDesign } });
    const nextEv = evaluateEvalDesign(nextDesign, brief);
    const nextPass = briefPasses(brief, nextDesign, nextEv);
    log.push(
      `${brief.id}: ${nextDesign.mode}, ${GRADER_BY_ID[nextDesign.grader].shortLabel}, n=${nextDesign.n}, cap=${(nextDesign.alpha * 100).toFixed(0)}% -> power ${pct1(nextEv.power)}, FPR ${pct2(nextEv.fpr)}, ${nextEv.cost.toLocaleString()} cr (${nextPass ? 'PASS' : 'FAIL'})`,
      nextPass ? 'good' : 'bad',
    );
  };

  useEffect(() => {
    if (!pass) return;
    const cur = saved.current[brief.id];
    if (cur.solved && cur.bestCost !== null && cur.bestCost <= ev.cost) return;
    persist({
      ...saved.current,
      [brief.id]: {
        ...cur,
        solved: true,
        bestCost: cur.bestCost === null ? ev.cost : Math.min(cur.bestCost, ev.cost),
      },
    });
  }, [pass, brief.id, ev.cost]); // eslint-disable-line react-hooks/exhaustive-deps

  const okPow = ev.power >= brief.minPower - 1e-9;
  const okFpr = ev.fpr <= brief.maxFpr + 1e-9;
  const g = GRADER_BY_ID[design.grader];
  const atten = 2 * g.accuracy - 1;
  const maxN = allowedNs[allowedNs.length - 1];
  const firstCross = curve.find((pt) => pt.power >= brief.minPower - 1e-9);
  const peakPower = curve[curve.length - 1].power;
  const trueDiscord = 2 * brief.baseDiscord + brief.delta;

  const animBars = useTweenedArray([ev.power * 100, Math.min(12, ev.fpr * 100), atten * 100], 220);
  const animPow = animBars[0] ?? ev.power * 100;
  const animFpr = animBars[1] ?? Math.min(12, ev.fpr * 100);
  const animAtten = animBars[2] ?? atten * 100;

  const hintText = useMemo(() => evalLabHint(brief, design, ev), [brief, design, ev]);
  const realCode = useMemo(() => evalSuiteRealCode(design, brief, ev), [design, brief, ev]);

  return (
    <Widget
      title="Lab: Size Your Eval Suite"
      wide
      caption={
        <>
          Design an A/B evaluation suite for each brief: choose <strong>Unpaired</strong> (1.0× grading cost) or <strong>Paired</strong> (1.5× cost, cancels shared question difficulty), a grader (Exact 100% at 5 cr, Dual LLM 93% at 1 cr, or Fast LLM 85% at 0.4 cr), sample size <strong>n</strong>, and your false-positive cap. Watch the live power curve and grader signal attenuation respond; click any point on the curve to jump to that sample size. Meet the brief; optionally minimise total grading cost.
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setShowHint(false);
          log.push(`Switched to ${LAB_BRIEFS[i].title}`, 'info');
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

      {/* Live "into the machine" instrument panel: Power(n) curve + Grader attenuation & gate meters */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(290px, 100%), 1fr))',
          gap: 14,
          marginBottom: 14,
          alignItems: 'stretch',
        }}
      >
        <div
          style={{
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: '10px 12px',
            background: 'var(--bg-subtle)',
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 11.5, fontWeight: 650, letterSpacing: '0.02em', color: 'var(--text-muted)' }}>
              EXACT POWER CURVE · {design.mode.toUpperCase()} · {g.shortLabel}
            </span>
            <span
              style={{
                fontFamily: 'var(--mono)',
                fontSize: 11.5,
                color: firstCross ? 'var(--c-good)' : 'var(--c-bad)',
                fontWeight: 600,
              }}
            >
              {firstCross
                ? `Reaches ${pct1(brief.minPower)} at n = ${firstCross.n.toLocaleString()}`
                : `Peaks at ${pct1(peakPower)} (n = ${maxN.toLocaleString()})`}
            </span>
          </div>

          <Plot
            width={380}
            height={195}
            x={[0, maxN * 1.06]}
            y={[0, 100]}
            xLabel="sample size n (click any dot)"
            yLabel="power (%)"
          >
            {({ x: sx, y: sy }) => {
              const targetY = sy(brief.minPower * 100);
              const curX = sx(design.n);
              const curY = sy(ev.power * 100);
              const pathD = curve
                .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${sx(pt.n).toFixed(1)} ${sy(pt.power * 100).toFixed(1)}`)
                .join(' ');
              const labelRight = design.n < maxN * 0.62;
              const labelY = !labelRight && ev.power > 0.72
                ? curY + 18
                : Math.max(sy(100) + 12, curY - 9);
              return (
                <>
                  {/* Passing power zone above target */}
                  <rect
                    x={sx(0)}
                    y={sy(100)}
                    width={sx(maxN * 1.06) - sx(0)}
                    height={targetY - sy(100)}
                    fill="var(--c-good)"
                    fillOpacity={0.08}
                  />
                  {/* Target power line */}
                  <line
                    x1={sx(0)}
                    x2={sx(maxN * 1.06)}
                    y1={targetY}
                    y2={targetY}
                    stroke="var(--c-good)"
                    strokeWidth={1.4}
                    strokeDasharray="4 3"
                  />
                  <text
                    x={sx(0) + 6}
                    y={targetY - 4}
                    textAnchor="start"
                    fontSize={10}
                    fontWeight={600}
                    fill="var(--c-good)"
                  >
                    {(brief.minPower * 100).toFixed(0)}% target
                  </text>
                  {/* Power(n) curve */}
                  <path d={pathD} fill="none" stroke="var(--c-a)" strokeWidth={2.2} strokeLinejoin="round" />
                  {/* Clickable discrete n points */}
                  {curve.map((pt) => {
                    const isCur = pt.n === design.n;
                    return (
                      <circle
                        key={pt.n}
                        cx={sx(pt.n)}
                        cy={sy(pt.power * 100)}
                        r={isCur ? 5.5 : 3.5}
                        fill={isCur ? (okPow ? 'var(--c-good)' : 'var(--c-bad)') : 'var(--bg)'}
                        stroke={isCur ? 'var(--text)' : 'var(--c-a)'}
                        strokeWidth={isCur ? 2 : 1.6}
                        style={{ cursor: 'pointer' }}
                        onClick={() => update({ n: pt.n })}
                      >
                        <title>{`n = ${pt.n}: power ${pct1(pt.power)}, cost ${pt.cost.toLocaleString()} cr`}</title>
                      </circle>
                    );
                  })}
                  {/* Vertical drop guide for current n */}
                  <line
                    x1={curX}
                    x2={curX}
                    y1={curY + 6}
                    y2={sy(0)}
                    stroke={okPow ? 'var(--c-good)' : 'var(--c-bad)'}
                    strokeWidth={1}
                    strokeDasharray="2 3"
                    opacity={0.75}
                  />
                  <text
                    x={labelRight ? curX + 8 : curX - 8}
                    y={labelY}
                    textAnchor={labelRight ? 'start' : 'end'}
                    fontSize={10.5}
                    fontWeight={700}
                    fontFamily="var(--mono)"
                    fill={okPow ? 'var(--c-good)' : 'var(--c-bad)'}
                  >
                    n={design.n}: {pct1(ev.power)}
                  </text>
                </>
              );
            }}
          </Plot>
        </div>

        <div
          style={{
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: '10px 12px',
            background: 'var(--bg-subtle)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: 10,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
            <span style={{ fontSize: 11.5, fontWeight: 650, letterSpacing: '0.02em', color: 'var(--text-muted)' }}>
              SIGNAL &amp; GATE METERS
            </span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 12.5, fontWeight: 700, color: 'var(--text)' }}>
              {ev.cost.toLocaleString()} cr
            </span>
          </div>

          {/* Grader signal attenuation meter */}
          <div style={{ display: 'grid', gap: 5 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>Grader signal retained (2q − 1)</span>
              <span style={{ fontFamily: 'var(--mono)', fontWeight: 600 }}>
                +{(ev.observedGap * 100).toFixed(2)} pp / +{(brief.delta * 100).toFixed(1)} pp (×{atten.toFixed(2)})
              </span>
            </div>
            <div style={{ position: 'relative', height: 10, borderRadius: 99, background: 'var(--bg-elev)', border: '1px solid var(--border)', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.max(0, Math.min(100, animAtten))}%`,
                  height: '100%',
                  background: g.accuracy >= 1 ? 'var(--c-good)' : 'var(--c-b)',
                  borderRadius: 99,
                }}
              />
            </div>
            {design.mode === 'paired' && ev.effectiveDiscord !== null && (
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                Paired disagreements: true <span style={{ fontFamily: 'var(--mono)' }}>{pct1(trueDiscord)}</span> → observed{' '}
                <span
                  style={{
                    fontFamily: 'var(--mono)',
                    color: ev.effectiveDiscord > trueDiscord + 0.01 ? 'var(--c-b)' : 'var(--c-good)',
                    fontWeight: 600,
                  }}
                >
                  {pct1(ev.effectiveDiscord)}
                </span>{' '}
                {ev.effectiveDiscord > trueDiscord + 0.01 ? '(grader noise injects fake flips)' : '(clean pairs)'}
              </div>
            )}
          </div>

          {/* Detection power meter */}
          <div style={{ display: 'grid', gap: 5 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>Detection power (need ≥ {pct1(brief.minPower)})</span>
              <span style={{ fontFamily: 'var(--mono)', fontWeight: 700, color: okPow ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {pct1(ev.power)}
              </span>
            </div>
            <div style={{ position: 'relative', height: 11, borderRadius: 99, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
              <div
                style={{
                  width: `${Math.max(0, Math.min(100, animPow))}%`,
                  height: '100%',
                  background: okPow ? 'var(--c-good)' : 'var(--c-bad)',
                  borderRadius: 99,
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  left: `${brief.minPower * 100}%`,
                  top: -2,
                  bottom: -2,
                  width: 2,
                  background: 'var(--text)',
                }}
                title={`Target ${pct1(brief.minPower)}`}
              />
            </div>
          </div>

          {/* False-positive rate meter (0%..12% scale) */}
          <div style={{ display: 'grid', gap: 5 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>Exact false-positive rate (need ≤ {pct1(brief.maxFpr)})</span>
              <span style={{ fontFamily: 'var(--mono)', fontWeight: 700, color: okFpr ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {pct2(ev.fpr)}
              </span>
            </div>
            <div style={{ position: 'relative', height: 11, borderRadius: 99, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
              <div
                style={{
                  width: `${Math.max(0, Math.min(100, (animFpr / 12) * 100))}%`,
                  height: '100%',
                  background: okFpr ? 'var(--c-good)' : 'var(--c-bad)',
                  borderRadius: 99,
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  left: `${Math.min(100, ((brief.maxFpr * 100) / 12) * 100)}%`,
                  top: -2,
                  bottom: -2,
                  width: 2,
                  background: 'var(--text)',
                }}
                title={`Limit ${pct1(brief.maxFpr)}`}
              />
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gap: 10 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
          <Segmented
            label="Eval protocol"
            value={design.mode}
            onChange={(mode) => update({ mode: mode as EvalMode })}
            options={[
              { value: 'unpaired', label: 'Unpaired (1.0× cost)' },
              { value: 'paired', label: 'Paired McNemar (1.5× cost)' },
            ]}
          />
          <Segmented
            label="False-positive cap"
            value={design.alpha}
            onChange={(alpha) => update({ alpha })}
            options={LAB_ALPHAS.map((a) => ({ value: a, label: `${(a * 100).toFixed(0)}%` }))}
          />
        </div>

        <Segmented
          label="Grader"
          value={design.grader}
          onChange={(grader) => update({ grader: grader as GraderId })}
          options={GRADERS.map((gr) => ({ value: gr.id, label: gr.shortLabel }))}
        />

        <Segmented
          label="Number of test questions"
          value={design.n}
          onChange={(n) => update({ n })}
          options={allowedNs.map((v) => ({ value: v, label: String(v) }))}
        />
      </div>

      <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
        <div className="btn-row">
          <span className={`status ${pass ? 'good' : 'bad'}`}>{pass ? 'Meets the brief' : 'Not yet'}</span>
          <button type="button" className="btn" onClick={() => setShowHint((h) => !h)}>
            {showHint ? 'Hide hint' : 'Hint'}
          </button>
          {entry.solved && (
            <span className="status good">
              Solved{entry.bestCost !== null ? ` · lowest cost so far: ${entry.bestCost.toLocaleString()} cr` : ''}
            </span>
          )}
        </div>

        {showHint && <div className="live-note">{hintText}</div>}

        <div className="readout">
          <span style={{ color: okPow ? 'var(--c-good)' : 'var(--c-bad)' }}>{okPow ? '✓' : '✗'}</span>{' '}
          <span className="k">detection power (for +{(brief.delta * 100).toFixed(1)} pp gain) </span>
          {pct1(ev.power)} (need ≥ {pct1(brief.minPower)})
          {'\n'}
          <span style={{ color: okFpr ? 'var(--c-good)' : 'var(--c-bad)' }}>{okFpr ? '✓' : '✗'}</span>{' '}
          <span className="k">exact false-positive rate (if no real gain) </span>
          {pct2(ev.fpr)} (need ≤ {pct1(brief.maxFpr)})
          {'\n'}
          <span className="k">observed gap through {g.shortLabel} </span>
          +{(ev.observedGap * 100).toFixed(2)} pp (true +{(brief.delta * 100).toFixed(1)} pp × {atten.toFixed(2)})
          {'\n'}
          <span className="k">total grading cost </span>
          <strong>{ev.cost.toLocaleString()} credits</strong>
        </div>

        <LiveCode code={realCode} note="scipy.stats exact test" />

        <div>
          <AnswerReveal
            solved={entry.solved}
            stuckKey={`eval-sizing:${brief.id}`}
            label="the exhaustive search & cost–power map"
          >
            <EvalParetoFrontier brief={brief} design={design} pass={pass} />
          </AnswerReveal>
        </div>

        <MachineLog
          title="EVAL SUITE LOG"
          lines={log.lines}
          onClear={log.clear}
          empty="Toggle protocol, grader, sample size, or cap to log evaluations."
        />
      </div>
    </Widget>
  );
}

function EvalParetoFrontier({ brief, design, pass }: { brief: EvalBrief; design: EvalDesign; pass: boolean }) {
  const ex = useMemo(() => labExhaustive(brief), [brief]);
  const bestPt = ex.best[0];
  const maxCost = Math.max(...ex.passingPts.map((p) => p.eval.cost), ex.bestCost * 1.2);
  const costDomainMax = Math.ceil(maxCost / 1000) * 1000;

  return (
    <div style={{ display: 'grid', gap: 10, marginTop: 8 }}>
      <div className="live-note">
        All {ex.total} combinations were computed exactly: <strong>{ex.passing}</strong> meet this brief. The cheapest costs{' '}
        <strong>{ex.bestCost.toLocaleString()} credits</strong>:{' '}
        {ex.best
          .map(
            (w) =>
              `${w.design.mode}, ${GRADER_BY_ID[w.design.grader].shortLabel}, n = ${w.design.n.toLocaleString()}, cap ${(w.design.alpha * 100).toFixed(0)}% (power ${pct1(w.eval.power)}, FPR ${pct2(w.eval.fpr)})`,
          )
          .join('; or ')}
        .
      </div>

      {ex.passingPts.length > 0 && (
        <div
          style={{
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: '10px 12px',
            background: 'var(--bg-subtle)',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 8,
              marginBottom: 6,
              fontSize: 12,
            }}
          >
            <span style={{ fontWeight: 650, color: 'var(--text-muted)' }}>
              ALL {ex.passing} PASSING SUITES · COST VS POWER
            </span>
            <span style={{ display: 'flex', gap: 10, flexWrap: 'wrap', color: 'var(--text-muted)' }}>
              {GRADERS.map((gr) => (
                <span key={gr.id} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 99,
                      background: GRADER_COLOR[gr.id],
                      display: 'inline-block',
                    }}
                  />
                  {gr.shortLabel}
                </span>
              ))}
            </span>
          </div>
          <Plot
            width={680}
            height={185}
            x={[0, costDomainMax * 1.04]}
            y={[81, 101]}
            xLabel="total grading cost (credits)"
            yLabel="power (%)"
          >
            {({ x: sx, y: sy }) => (
              <>
                <line
                  x1={sx(0)}
                  x2={sx(costDomainMax * 1.04)}
                  y1={sy(brief.minPower * 100)}
                  y2={sy(brief.minPower * 100)}
                  stroke="var(--c-good)"
                  strokeDasharray="4 3"
                  strokeWidth={1.2}
                />
                {ex.passingPts.map((pt, idx) => {
                  const isBest =
                    pt.design.mode === bestPt.design.mode &&
                    pt.design.grader === bestPt.design.grader &&
                    pt.design.n === bestPt.design.n &&
                    pt.design.alpha === bestPt.design.alpha;
                  const isCur =
                    pass &&
                    pt.design.mode === design.mode &&
                    pt.design.grader === design.grader &&
                    pt.design.n === design.n &&
                    pt.design.alpha === design.alpha;
                  return (
                    <circle
                      key={idx}
                      cx={sx(pt.eval.cost)}
                      cy={sy(pt.eval.power * 100)}
                      r={isBest ? 6 : 4}
                      fill={GRADER_COLOR[pt.design.grader]}
                      fillOpacity={isBest ? 1 : 0.78}
                      stroke={isCur ? 'var(--text)' : isBest ? 'var(--c-good)' : 'none'}
                      strokeWidth={isCur ? 2.2 : isBest ? 2 : 0}
                    >
                      <title>{`${pt.design.mode}, ${GRADER_BY_ID[pt.design.grader].shortLabel}, n=${pt.design.n}, cap ${(pt.design.alpha * 100).toFixed(0)}%: ${pct1(pt.eval.power)} power, ${pt.eval.cost.toLocaleString()} cr`}</title>
                    </circle>
                  );
                })}
                <text
                  x={sx(bestPt.eval.cost) + 8}
                  y={sy(bestPt.eval.power * 100) - 9}
                  fontSize={10.5}
                  fontWeight={700}
                  fontFamily="var(--mono)"
                  fill="var(--c-good)"
                >
                  Cheapest: {ex.bestCost.toLocaleString()} cr
                </text>
              </>
            )}
          </Plot>
        </div>
      )}
    </div>
  );
}
