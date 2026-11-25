// authored-by: Claude Opus 5.5
import { useMemo, useState, type ReactElement } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Label, Plot, usePlot } from '../../components/plot';
import {
  ACTION_ARROWS,
  BOAT_TRACK,
  BRIEFS,
  CAMERA_ROOM,
  FIELD,
  QL_EPISODES,
  VASE_ROOM,
  boatVerdict,
  boatWorld,
  changedSquares,
  complexity,
  exhaustiveLab,
  findPlan,
  judge,
  shapingRace,
  solve,
  solveField,
  type BoatVerdict,
  type BriefId,
  type PlanStep,
  type ShapingMode,
  type Weights,
} from './hack';

const signedNum = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${Math.abs(v)}` : '0');

const EVENT_MARK: Record<string, { mark: string; color: string; label: string }> = {
  pad: { mark: '★', color: 'var(--c-b)', label: 'pad fires' },
  finish: { mark: '⚑', color: 'var(--c-good)', label: 'finish' },
  deliver: { mark: '⚑', color: 'var(--c-good)', label: 'delivered' },
  break: { mark: '✕', color: 'var(--c-bad)', label: 'vase breaks' },
  clean: { mark: '✓', color: 'var(--c-good)', label: 'cleans a mess' },
  spill: { mark: '!', color: 'var(--c-bad)', label: 'spills the bin' },
  tamper: { mark: '⏻', color: 'var(--c-bad)', label: 'camera off' },
};

const TILE_LABEL: Record<string, string> = { S: 'start', F: 'finish', P: 'pad', G: 'goal', V: 'vase', C: 'camera', M: 'mess' };

function cellOf(map: readonly string[], ch: string): [number, number] {
  for (let i = 0; i < map.length; i++) {
    const j = map[i].indexOf(ch);
    if (j >= 0) return [i, j];
  }
  return [0, 0];
}

/** A grid world with the agent's plan drawn over it. */
function GridView({ map, steps, endChar, highlight, arrows, size = 56 }: {
  map: readonly string[];
  steps?: PlanStep[];
  endChar?: string;
  highlight?: (i: number, j: number) => string | undefined;
  arrows?: (i: number, j: number) => number[];
  size?: number;
}) {
  const rows = map.length;
  const cols = map[0].length;
  const W = cols * size + 4;
  const H = rows * size + 4;
  const pos = (s: string): [number, number] => {
    const [r, c] = s.split(',').map(Number);
    return [r, c];
  };
  const pts: [number, number][] = steps ? steps.map((s) => pos(s.state)) : [];
  if (steps && steps.length) {
    const lastEv = steps[steps.length - 1].events;
    if ((lastEv.includes('finish') || lastEv.includes('deliver')) && endChar) pts.push(cellOf(map, endChar));
  }
  const cx = (j: number) => 2 + j * size + size / 2;
  const cy = (i: number) => 2 + i * size + size / 2;
  // events by cell (where the agent ends up after the step)
  const marks: { i: number; j: number; ev: string; n: number }[] = [];
  if (steps) {
    steps.forEach((s, k) => {
      const where = pts[k + 1] ?? pts[k];
      for (const e of s.events) if (EVENT_MARK[e]) marks.push({ i: where[0], j: where[1], ev: e, n: k + 1 });
    });
  }
  // de-duplicate repeated marks per cell
  const markCells = new Map<string, { ev: string; count: number }>();
  for (const m of marks) {
    const key = `${m.i},${m.j},${m.ev}`;
    const cur = markCells.get(key);
    markCells.set(key, { ev: m.ev, count: (cur?.count ?? 0) + 1 });
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ margin: '0 auto' }} role="img" aria-label="Grid world with the agent's plan">
      {map.map((row, i) =>
        row.split('').map((ch, j) => {
          const hl = highlight?.(i, j);
          const fill =
            ch === '#'
              ? 'var(--border-strong)'
              : hl ?? (ch === 'F' || ch === 'G' ? 'color-mix(in srgb, var(--c-good) 20%, transparent)' : ch === 'P' ? 'color-mix(in srgb, var(--c-b) 18%, transparent)' : ch === 'V' || ch === 'C' ? 'color-mix(in srgb, var(--c-bad) 12%, transparent)' : ch === 'M' ? 'color-mix(in srgb, var(--c-b) 12%, transparent)' : 'var(--bg-sunken)');
          return (
            <g key={`${i}-${j}`}>
              <rect x={2 + j * size + 1.5} y={2 + i * size + 1.5} width={size - 3} height={size - 3} rx={6} fill={fill} stroke="var(--border)" />
              {TILE_LABEL[ch] && (
                <text x={cx(j)} y={2 + i * size + 13} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
                  {TILE_LABEL[ch]}
                </text>
              )}
              {arrows &&
                ch !== '#' &&
                ch !== 'G' &&
                arrows(i, j).map((a) => {
                  const dx = [0, 1, 0, -1][a];
                  const dy = [-1, 0, 1, 0][a];
                  const L = size * 0.28;
                  return (
                    <line
                      key={a}
                      x1={cx(j)}
                      y1={cy(i) + 4}
                      x2={cx(j) + dx * L}
                      y2={cy(i) + 4 + dy * L}
                      stroke="var(--accent)"
                      strokeWidth={2.2}
                      markerEnd="url(#rh-arrow)"
                    />
                  );
                })}
            </g>
          );
        }),
      )}
      <defs>
        <marker id="rh-arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>
      {pts.length > 1 && (
        <polyline
          points={pts.map(([i, j], k) => `${cx(j) + ((k % 3) - 1) * 2},${cy(i) + 6 + ((k % 3) - 1) * 2}`).join(' ')}
          fill="none"
          stroke="var(--c-grad)"
          strokeWidth={2.5}
          strokeLinejoin="round"
          opacity={0.8}
        />
      )}
      {pts.length > 0 && <circle cx={cx(pts[0][1])} cy={cy(pts[0][0]) + 6} r={5} fill="var(--c-grad)" />}
      {[...markCells.entries()].map(([key, m], k) => {
        const [i, j] = key.split(',').map(Number);
        const em = EVENT_MARK[m.ev];
        return (
          <text key={k} x={2 + j * size + size - 8} y={2 + i * size + size - 7} textAnchor="end" fontSize={13} fontWeight={700} fill={em.color}>
            {em.mark}
            {m.count > 1 ? `×${m.count}` : ''}
          </text>
        );
      })}
    </svg>
  );
}

function PlanText({ steps, max = 24 }: { steps: PlanStep[]; max?: number }) {
  return (
    <div className="readout" style={{ lineHeight: 1.9 }}>
      <span className="k">plan </span>
      {steps.slice(0, max).map((s, k) => {
        const evs = s.events.filter((e) => EVENT_MARK[e]);
        return (
          <span key={k}>
            {ACTION_ARROWS[s.action]}
            {evs.map((e) => (
              <span key={e} style={{ color: EVENT_MARK[e].color, fontWeight: 700 }}>
                {EVENT_MARK[e].mark}
              </span>
            ))}{' '}
          </span>
        );
      })}
      {steps.length > max ? '…' : ''}
    </div>
  );
}

// ------------------------------------------------------------------ 1. The boat race

const VERDICT_COLOR: Record<BoatVerdict, string> = {
  direct: 'var(--c-a)',
  'pad-then-finish': 'var(--c-good)',
  loops: 'var(--c-bad)',
  mixed: 'var(--text-muted)',
  'never-finishes': 'var(--c-bad)',
};
const VERDICT_TEXT: Record<BoatVerdict, string> = {
  direct: 'drives straight to the finish',
  'pad-then-finish': 'takes the pad once, then finishes',
  loops: 'circles the pad for ever',
  mixed: 'is torn between plans',
  'never-finishes': 'never finishes',
};

function Segments({ pts }: { pts: { p: number; v: number; verdict: BoatVerdict }[] }) {
  const { x, y } = usePlot();
  const out: ReactElement[] = [];
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1];
    const b = pts[k];
    if (a.verdict !== b.verdict) continue;
    out.push(<line key={k} x1={x(a.p)} y1={y(a.v)} x2={x(b.p)} y2={y(b.v)} stroke={VERDICT_COLOR[a.verdict]} strokeWidth={3} strokeLinecap="round" />);
  }
  return <g>{out}</g>;
}

export const BOAT_T1 = 1.7622;
export const BOAT_T2 = 4.5396;

export function BoatRace() {
  const [pad, setPad] = useState(0);
  const world = useMemo(() => boatWorld(), []);
  const sol = useMemo(() => solve(world, { finish: 10, pad, step: 0 }), [world, pad]);
  const verdict = boatVerdict(sol);
  const plan = findPlan(sol, 20);
  const curve = useMemo(
    () =>
      Array.from({ length: 121 }, (_, k) => {
        const p = (k * 6) / 120;
        const s = solve(world, { finish: 10, pad: p, step: 0 });
        return { p, v: s.V[0], verdict: boatVerdict(s) };
      }),
    [world],
  );
  const hits = plan.steps.filter((s) => s.events.includes('pad')).length;

  return (
    <Widget
      title="The boat race: when does a bonus become the goal?"
      wide
      caption={
        <>
          The finish is worth +10 and ends the race (γ = 0.95). The orange pad pays your bonus every time the boat drives onto it, then needs 8 steps to recharge. Value Iteration finds the best possible behaviour for each bonus, exactly. Drag the bonus up slowly from 0 and watch the plan change twice.
        </>
      }
    >
      <Slider label="Turbo-pad bonus" value={pad} min={0} max={6} step={0.05} onChange={setPad} format={(v) => `+${v.toFixed(2)}`} color="c-b" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 8 }}>
        <div style={{ display: 'grid', gap: 10 }}>
          <GridView map={BOAT_TRACK} steps={plan.steps} endChar="F" />
          <PlanText steps={plan.steps} />
          <div className="live-note">
            At +{pad.toFixed(2)} the best plan <strong style={{ color: VERDICT_COLOR[verdict] }}>{VERDICT_TEXT[verdict]}</strong>
            {verdict === 'loops' ? ` (pad hits in 20 steps: ${hits})` : ''}.{' '}
            {verdict === 'direct'
              ? `A detour over the pad costs 4 extra steps, and the bonus isn’t worth them until it passes ${BOAT_T1.toFixed(3)}.`
              : verdict === 'pad-then-finish'
                ? `The bonus pays for the detour, but not for giving up the finish: that happens above ${BOAT_T2.toFixed(3)}.`
                : 'A recharging +bonus every 9 steps, for ever, is now worth more than +10 once. The boat is not broken: this is the best answer to the question you asked.'}
          </div>
        </div>
        <Plot width={420} height={250} x={[0, 6]} y={[0, 30]} xLabel="pad bonus" yLabel="best total reward" ariaLabel="Best achievable reward against pad bonus">
          {({ x, y }) => (
            <g>
              {[BOAT_T1, BOAT_T2].map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={y(0)} y2={y(30)} stroke="var(--text-faint)" strokeDasharray="4 4" />
              ))}
              <circle cx={x(pad)} cy={y(sol.V[0])} r={6} fill={VERDICT_COLOR[verdict]} stroke="var(--bg)" strokeWidth={1.5} />
            </g>
          )}
          <Segments pts={curve} />
          <Label at={[0.1, 11.5]} color="var(--c-a)" size={11.5}>straight to the finish</Label>
          <Label at={[1.85, 16]} color="var(--c-good)" size={11.5}>pad once</Label>
          <Label at={[4.6, 26]} color="var(--c-bad)" size={11.5}>loop for ever</Label>
        </Plot>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Shaping explorer

export function ShapingExplorer() {
  const [mode, setMode] = useState<ShapingMode>('naive');
  const [k, setK] = useState(0.25);
  const [race, setRace] = useState<null | { label: string; alpha: number; mean: number; sd: number; never: number; first: number }[]>(null);
  const [racing, setRacing] = useState(false);

  const sol = useMemo(() => solveField(mode, k), [mode, k]);
  const base = useMemo(() => solveField('none', 0), []);
  const changed = mode === 'none' ? 0 : changedSquares(mode, k);
  const plan = findPlan(sol, 20, (o) => !o.done);
  const finishes = plan.outcome.done;

  const optAt = (i: number, j: number) => {
    const id = sol.ids.get(`${i},${j}`);
    return id === undefined ? [] : sol.opt[id];
  };
  const isChanged = (i: number, j: number) => {
    const a = base.ids.get(`${i},${j}`);
    const b = sol.ids.get(`${i},${j}`);
    if (a === undefined || b === undefined) return false;
    return base.opt[a].join(',') !== sol.opt[b].join(',');
  };

  const runRace = () => {
    setRacing(true);
    setTimeout(() => {
      const rows = [
        { label: 'no shaping', mode: 'none' as ShapingMode, k: 0 },
        ...(mode === 'none' ? [] : [{ label: `${mode === 'naive' ? 'naive' : 'potential-based'}, k = ${k.toFixed(2)}`, mode, k }]),
      ].map((r) => {
        const res = shapingRace(r.mode, r.k);
        const sd = Math.sqrt(res.perSeed.reduce((s, v) => s + (v - res.mean) ** 2, 0) / (res.perSeed.length - 1));
        return { label: r.label, alpha: res.alpha, mean: res.mean, sd, never: res.perSeed.filter((v) => v > QL_EPISODES).length, first: res.curve[0] };
      });
      setRace(rows);
      setRacing(false);
    }, 20);
  };

  return (
    <Widget
      title="Shaping: helpful hints that don’t change the answer"
      wide
      caption={
        <>
          An open field: start bottom-left, +10 at the goal top-right, nothing else (γ = 0.95). That’s a sparse reward: a learner wanders a long time before it ever sees the +10. A shaping bonus rewards getting closer. Arrows show every best move in every square; squares where the best moves differ from the unshaped field are tinted red. Try the <strong>naive</strong> bonus past k ≈ 0.57, then switch to <strong>potential-based</strong> and push k as high as you like.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'end', marginBottom: 10 }}>
        <Segmented
          value={mode}
          onChange={(m) => {
            setMode(m);
            setRace(null);
          }}
          options={[
            { value: 'none', label: 'no shaping' },
            { value: 'naive', label: 'naive: +k·closeness(s′)' },
            { value: 'potential', label: 'potential: k·(γΦ(s′) − Φ(s))' },
          ]}
        />
        <div style={{ minWidth: 200, flex: '1 1 200px' }}>
          <Slider label="bonus strength k" value={k} min={0} max={4} step={0.05} onChange={(v) => { setK(v); setRace(null); }} color="accent" />
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <GridView
          map={FIELD}
          steps={plan.steps}
          endChar="G"
          size={58}
          arrows={optAt}
          highlight={(i, j) => (isChanged(i, j) ? 'color-mix(in srgb, var(--c-bad) 22%, transparent)' : undefined)}
        />
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">squares whose best moves changed </span>
            {changed} / 24
            {'\n'}
            <span className="k">best plan from the start </span>
            {finishes ? `reaches the goal in ${plan.steps.length} steps` : 'never reaches the goal'}
          </div>
          <div className="live-note">
            {mode === 'none' ? (
              <>The true problem: only the goal pays. Every square’s best moves point up or right — any shortest route works.</>
            ) : mode === 'naive' ? (
              finishes ? (
                <>A small bonus for being close leaves the best moves unchanged (for now). But reaching the goal <em>ends</em> the stream of bonuses. Push k past 4/7 ≈ 0.571 and see what that does.</>
              ) : (
                <>
                  The agent now parks next to the goal and bumps the wall for ever: {k.toFixed(2)} × ⅞ per step, for ever, is worth {((k * 0.875) / 0.05).toFixed(1)} — more than the +10 that would end it. You paid for “closer”, not for “there”.
                </>
              )
            ) : (
              <>
                Not one square changed{k > 0 ? `, even at k = ${k.toFixed(2)}` : ''}. Every bonus is a <em>difference</em> of potentials, so around any loop they cancel, and along any route they add up to Φ(end) − Φ(start), the same for every route. They change how good things look on the way, never which way is best.
              </>
            )}
          </div>
          <div className="btn-row">
            <button type="button" className="btn primary" onClick={runRace} disabled={racing}>
              {racing ? 'Racing…' : 'Race Q-learners (30 seeds)'}
            </button>
          </div>
          {race && (
            <div style={{ overflowX: 'auto' }}>
              <table className="num-table" style={{ width: '100%' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left' }}>reward</th>
                    <th>best step</th>
                    <th>episodes to learn</th>
                    <th>never learned</th>
                    <th>steps in ep. 1</th>
                  </tr>
                </thead>
                <tbody>
                  {race.map((r) => (
                    <tr key={r.label}>
                      <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)' }}>{r.label}</td>
                      <td>{r.alpha}</td>
                      <td>
                        {r.mean.toFixed(1)} ± {r.sd.toFixed(1)}
                      </td>
                      <td>{r.never} / 30</td>
                      <td>{r.first.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 4 }}>
                Q-learning, ε = 0.1, each reward at its own best step size from {`{0.05 … 0.9}`}. “Learned” = the greedy policy first walks the 8-step route ({QL_EPISODES + 1} if it never does in {QL_EPISODES} episodes).
              </div>
            </div>
          )}
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. The lab

const LAB_KEY = 'nl-reward-hack-lab-v1';
type LabState = Record<BriefId, { wts: Weights; solved: boolean; solvedSimplest: boolean }>;

const zeroWeights = (id: BriefId): Weights => Object.fromEntries(BRIEFS.find((b) => b.id === id)!.knobs.map((k) => [k.event, k.values[0]]));
const STARTERS: Record<BriefId, Weights> = {
  boat: { finish: 10, pad: 0, step: 0 },
  vase: { deliver: 10, break: 0, step: 0 },
  camera: { clean: 0, 'looks-clean': 1, tamper: 0, spill: 0, step: 0 },
};

function loadLab(): LabState {
  const d: LabState = {
    boat: { wts: { ...STARTERS.boat }, solved: false, solvedSimplest: false },
    vase: { wts: { ...STARTERS.vase }, solved: false, solvedSimplest: false },
    camera: { wts: { ...STARTERS.camera }, solved: false, solvedSimplest: false },
  };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return d;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of BRIEFS) {
      const v = parsed[b.id];
      if (!v || typeof v !== 'object') continue;
      const wts = { ...zeroWeights(b.id) };
      let valid = true;
      for (const k of b.knobs) {
        const x = v.wts?.[k.event];
        if (typeof x === 'number' && k.values.includes(x)) wts[k.event] = x;
        else valid = false;
      }
      if (valid) d[b.id] = { wts, solved: !!v.solved, solvedSimplest: !!v.solvedSimplest };
    }
  } catch {
    // ignore storage errors
  }
  return d;
}

const HINTS: Record<BriefId, string> = {
  boat: 'Three plans compete: straight to the finish, one detour over the pad, or circling the pad for ever. The bonus must beat the detour’s cost in delay but lose to the finish. What if finishing were the way to stop paying something?',
  vase: 'The agent doesn’t know the vase matters unless the reward says so. And every reward term is a price: whatever you charge for the vase has to beat what the shortcut saves.',
  camera: 'Paying for what the camera reports invites the agent to fix the camera instead of the room. Paying per clean invites it to make new messes to clean. And once the job is done, every remaining action ties — the checker assumes the agent might pick any of them.',
};

export function RewardLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [result, setResult] = useState<ReturnType<typeof judge> | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [showOpt, setShowOpt] = useState(false);
  const brief = BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const ex = useMemo(() => (entry.solved ? exhaustiveLab(brief.id) : null), [brief.id, entry.solved]);

  const persist = (next: LabState) => {
    setLab(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };

  const setKnob = (event: string, v: number) => {
    persist({ ...lab, [brief.id]: { ...entry, wts: { ...entry.wts, [event]: v } } });
    setResult(null);
  };

  const run = () => {
    const r = judge(brief.id, entry.wts);
    setResult(r);
    if (r.pass) {
      const exh = exhaustiveLab(brief.id);
      const mine = complexity(entry.wts);
      const best = complexity(exh.simplest!.wts);
      const simplest = mine.terms === best.terms && mine.size === best.size;
      persist({ ...lab, [brief.id]: { ...entry, solved: true, solvedSimplest: entry.solvedSimplest || simplest } });
    }
  };

  const plan = result ? findPlan(result.sol, brief.id === 'camera' ? 40 : brief.horizon, (o) => !brief.ok(o) || (brief.id === 'camera' && (o.events.has('tamper') || o.events.has('spill')))) : null;
  const map = brief.id === 'boat' ? BOAT_TRACK : brief.id === 'vase' ? VASE_ROOM : CAMERA_ROOM;
  const endChar = brief.id === 'boat' ? 'F' : 'G';
  const mine = complexity(entry.wts);

  return (
    <Widget
      title="Lab: The Reward-Hacking Sandbox"
      wide
      caption={
        <>
          You write the reward; an exact optimiser (Value Iteration, γ = 0.95) finds the behaviour that earns the most of it — instantly, with no mercy. The checker then follows <strong>every</strong> plan the optimiser considers best (when two plans tie, the agent might take either) and tests it against what you actually wanted. Your rewards are saved in this browser.
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setResult(null);
          setShowHint(false);
          setShowOpt(false);
        }}
        options={BRIEFS.map((b, i) => ({ value: i, label: `${lab[b.id].solved ? '✓ ' : ''}${b.title.split(' · ')[0]}` }))}
      />
      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
        <div style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 6 }}>Wanted: {brief.spec}.</div>
      </div>
      <div style={{ display: 'grid', gap: 10, marginBottom: 12 }}>
        {brief.knobs.map((k) => (
          <div key={k.event} style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
            <div style={{ minWidth: 190, fontSize: 14 }}>
              <div style={{ fontWeight: 600 }}>{k.label}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{k.hint}</div>
            </div>
            <Segmented value={entry.wts[k.event] ?? 0} onChange={(v) => setKnob(k.event, v)} options={k.values.map((v) => ({ value: v, label: signedNum(v) }))} />
          </div>
        ))}
        <div className="btn-row">
          <button type="button" className="btn primary" onClick={run}>Unleash the optimiser</button>
          <button type="button" className="btn" onClick={() => setShowHint((h) => !h)}>{showHint ? 'Hide hint' : 'Hint'}</button>
          {entry.solved && <span className="status good">Solved{entry.solvedSimplest ? ' · simplest reward found' : ''}</span>}
        </div>
        {showHint && <div className="live-note">{HINTS[brief.id]}</div>}
      </div>

      {result && plan && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
          <div style={{ display: 'grid', gap: 8 }}>
            <GridView map={map} steps={plan.steps} endChar={endChar} size={brief.id === 'vase' ? 60 : 56} />
            <PlanText steps={plan.steps} max={brief.id === 'camera' ? 30 : 20} />
          </div>
          <div style={{ display: 'grid', gap: 10 }}>
            <div>
              <span className={`status ${result.pass ? 'good' : 'bad'}`}>{result.pass ? 'Every best plan does what you wanted' : 'Loophole found'}</span>
            </div>
            {!result.pass && (
              <div className="readout">
                {result.failures.map((f) => (
                  <div key={f}>✗ the agent {f}</div>
                ))}
              </div>
            )}
            <div className="live-note">
              {result.pass ? (
                <>
                  Watertight, within this little world: every tied best plan does the job. Your reward uses {mine.terms} non-zero term{mine.terms === 1 ? '' : 's'} with total size {mine.size}.{' '}
                  {ex && !entry.solvedSimplest ? 'Optional: can you find a simpler reward that still works (fewer terms, then smaller numbers)?' : ''}
                </>
              ) : (
                <>The plan shown is one the optimiser rates as best (or tied for best) under your reward. It isn’t cheating — it’s answering exactly the question your numbers asked. Change the numbers, not the agent.</>
              )}
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              {Object.entries(EVENT_MARK)
                .filter(([e]) => brief.knobs.some((k) => k.event === e) || (brief.id === 'boat' && e === 'finish') || (brief.id === 'vase' && e === 'deliver'))
                .map(([e, m]) => (
                  <span key={e} style={{ marginRight: 12 }}>
                    <span style={{ color: m.color, fontWeight: 700 }}>{m.mark}</span> {m.label}
                  </span>
                ))}
            </div>
          </div>
        </div>
      )}

      {entry.solved && ex && (
        <div style={{ marginTop: 14 }}>
          <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>{showOpt ? 'Hide' : 'Show'} the exhaustive search</button>
          {showOpt && (
            <div className="live-note" style={{ marginTop: 10 }}>
              All {ex.total} reward settings on these dials were tried: <strong>{ex.passing.length}</strong> work. The simplest (fewest non-zero terms, then smallest numbers) is{' '}
              <strong>
                {Object.entries(ex.simplest!.wts)
                  .filter(([, v]) => v !== 0)
                  .map(([e, v]) => `${brief.knobs.find((k) => k.event === e)!.label} ${signedNum(v)}`)
                  .join(', ')}
              </strong>
              {ex.simplestCount === 1 ? ', and it is the only one that simple.' : `, one of ${ex.simplestCount} equally simple.`}
            </div>
          )}
        </div>
      )}
    </Widget>
  );
}
