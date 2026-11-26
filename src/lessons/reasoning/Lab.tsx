// authored-by: Claude Opus 5.5
import { useEffect, useRef, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Segmented } from '../../components/controls';
import { STRATEGIES, TIERS, VERIFIERS, type Config, type Strategy, type Tier, type Verifier } from './reasoning';
import { applyLocks, B2_BAR, B3_BAR, B3_TOKENS, BRIEFS, briefGrid, LAB_N, LAB_STRATEGIES, LAB_THINKING, LAB_TIERS, LAB_VERIFIERS, labHint, LOCKS, passes, refScore, score, type BriefId } from './lab';

const KEY = 'reasoning-lab-config-v1';
const INIT: Config = { tier: 'small', strategy: 'single', n: 1, verifier: 'none', thinking: 0 };
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const tf = (f: number) => `${(f / 1e12).toFixed(2)} TFLOP`;

function load(): Config {
  try {
    const c = JSON.parse(localStorage.getItem(KEY) ?? 'null');
    if (c && LAB_TIERS.includes(c.tier) && LAB_STRATEGIES.includes(c.strategy) && LAB_VERIFIERS.includes(c.verifier) && LAB_THINKING.includes(c.thinking)) return c;
  } catch {
    /* ignore */
  }
  return INIT;
}

export function buildReasoningCode(c: Config) {
  const lines = [`model = load("${c.tier}")   # ${TIERS[c.tier].label}`];
  if (c.strategy === 'single') lines.push(`answer = model.generate(problem, max_thinking_tokens=${c.thinking})`);
  else {
    lines.push(`answers = [model.generate(problem, max_thinking_tokens=${c.thinking}, temperature=0.8)`, `           for _ in range(${c.n})]`);
    if (c.strategy === 'majority') lines.push('answer = Counter(a.final for a in answers).most_common(1)[0][0]');
    else if (c.verifier === 'exact') lines.push('answer = next((a for a in answers if run_tests(a)), answers[0])');
    else if (c.verifier === 'noisy') lines.push('answer = max(answers, key=reward_model.score)');
    else lines.push('answer = random.choice(answers)   # no judge: best-of-N is just a random pick');
  }
  return lines.join('\n');
}

export function BudgetLab() {
  const progress = useLabProgress('reasoning-lab-v1');
  const log = useMachineLog();
  const [active, setActive] = useState(0);
  const brief = BRIEFS[active].id as BriefId;
  const [raw, setRaw] = useState<Config>(load);
  const c = applyLocks(brief, raw);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(raw));
    } catch {
      /* ignore */
    }
  }, [raw]);
  const s = score(brief, c);
  const ok = passes(brief, s);
  const ref = refScore();
  const grid = briefGrid(brief).map((g) => ({ g, s: score(brief, g) }));
  const sols = grid.filter((x) => passes(brief, x.s));
  const optimal =
    ok &&
    (brief === 'cheap'
      ? s.flops <= Math.min(...sols.map((x) => x.s.flops))
      : brief === 'noisy'
        ? s.acc >= Math.max(...grid.map((x) => x.s.acc)) - 1e-12
        : s.tokens <= Math.min(...sols.map((x) => x.s.tokens)));

  const lastLogged = useRef('');
  useEffect(() => {
    const k = `${brief}:${JSON.stringify(c)}`;
    if (k === lastLogged.current) return;
    lastLogged.current = k;
    log.push(`${TIERS[c.tier].label}, ${STRATEGIES[c.strategy]}${c.strategy === 'single' ? '' : ` × ${c.n}`}${c.strategy === 'best' ? `, ${VERIFIERS[c.verifier]}` : ''}, think ${c.thinking} → ${pct(s.acc)}, ${tf(s.flops)}, ${s.tokens} tokens`, ok ? 'good' : undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brief, c.tier, c.strategy, c.n, c.verifier, c.thinking]);

  const L = LOCKS[brief];
  const set = (patch: Partial<Config>) => setRaw((r) => ({ ...r, ...patch, ...(patch.strategy && patch.strategy !== 'single' && r.n < 2 ? { n: 2 } : {}) }));

  const briefs: LabBrief[] = [
    {
      id: 'cheap',
      tab: BRIEFS[0].tab,
      title: 'Brief 1 — No checker, less compute',
      text: `Open-ended questions: nothing can check the answers. The large model answering once after 100 thinking tokens scores ${pct(ref.acc)} for ${tf(ref.flops)} per problem. Match or beat its accuracy with the small or medium model, for less compute. Optional: the least compute.`,
    },
    {
      id: 'noisy',
      tab: BRIEFS[1].tab,
      title: 'Brief 2 — A judge you can’t fully trust',
      text: `A reward model scores each answer and you keep the best: the medium model, best-of-N. Reach ${pct(B2_BAR)} on the benchmark. Optional: the highest accuracy this setup can reach.`,
    },
    {
      id: 'hard',
      tab: BRIEFS[2].tab,
      title: 'Brief 3 — Hard problems, exact checker, fixed budget',
      text: `Only the 20 problems needing 7 to 10 steps, with unit tests that recognise a right answer for sure. The medium model may generate at most ${B3_TOKENS.toLocaleString()} tokens per problem in all. Solve ${pct(B3_BAR)}. Optional: the fewest tokens.`,
    },
  ];

  const status = `${pct(s.acc)} solved · ${tf(s.flops)} per problem · ${s.tokens.toLocaleString()} tokens generated${ok ? (optimal ? ' · ✓ passed, and optimal' : ' · ✓ passed') : ''}.`;
  const best = brief === 'cheap' ? sols.reduce((a, b) => (b.s.flops < a.s.flops ? b : a)) : brief === 'noisy' ? grid.reduce((a, b) => (b.s.acc > a.s.acc ? b : a)) : sols.reduce((a, b) => (b.s.tokens < a.s.tokens ? b : a));
  const reveal = (
    <div>
      <p style={{ margin: '0 0 8px' }}>
        Every allowed setting of this brief was computed exactly ({grid.length} of them); {sols.length} pass. The optimum: {TIERS[best.g.tier].label}, {STRATEGIES[best.g.strategy]}
        {best.g.strategy === 'single' ? '' : ` with N = ${best.g.n}`}, {best.g.thinking} thinking tokens: {pct(best.s.acc)}, {tf(best.s.flops)}, {best.s.tokens.toLocaleString()}{' '}
        tokens.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="num-table" style={{ fontSize: 12.5 }}>
          <thead>
            <tr>
              <th>model</th>
              <th>strategy</th>
              <th>samples</th>
              <th>thinking</th>
              <th>solved</th>
              <th>compute</th>
              <th>tokens</th>
            </tr>
          </thead>
          <tbody>
            {sols.map((x, i) => (
              <tr key={i}>
                <td>{TIERS[x.g.tier].label}</td>
                <td>{STRATEGIES[x.g.strategy]}</td>
                <td>{x.g.n}</td>
                <td>{x.g.thinking}</td>
                <td className="mono">{pct(x.s.acc)}</td>
                <td className="mono">{tf(x.s.flops)}</td>
                <td className="mono">{x.s.tokens}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const tierOpts = (L.tier ? [L.tier] : (L.tiers ?? LAB_TIERS)) as Tier[];
  const stratOpts = (L.strategy ? [L.strategy] : (L.strategies ?? LAB_STRATEGIES)) as Strategy[];
  const verOpts = (L.verifier ? [L.verifier] : LAB_VERIFIERS) as Verifier[];
  return (
    <LabShell
      title="Lab: Spend the Test-Time Budget"
      caption="The stand-in model from this lesson on its 50-problem benchmark (problems needing 1 to 10 steps, with traps of different strength). Every number is computed exactly; there is no sampling luck."
      briefs={briefs}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={ok}
      optimal={optimal}
      status={status}
      hint={labHint(brief, c, s)}
      reveal={{ label: 'every setting of this brief, computed', content: reveal }}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Change a setting to log it.' }}
      actions={
        <button type="button" className="btn" onClick={() => setRaw(INIT)}>
          Reset
        </button>
      }
    >
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div style={{ flex: '1 1 280px', display: 'grid', gap: 10 }}>
          <Segmented label="Model" options={tierOpts.map((t) => ({ value: t, label: TIERS[t].label }))} value={c.tier} onChange={(v) => set({ tier: v as Tier })} />
          <Segmented label="Strategy" options={stratOpts.map((t) => ({ value: t, label: STRATEGIES[t] }))} value={c.strategy} onChange={(v) => set({ strategy: v as Strategy })} />
          {c.strategy !== 'single' && <Segmented label="Samples per problem" options={LAB_N} value={c.n} onChange={(v) => set({ n: v as number })} />}
          {c.strategy === 'best' && <Segmented label="Judge" options={verOpts.map((t) => ({ value: t, label: VERIFIERS[t] }))} value={c.verifier} onChange={(v) => set({ verifier: v as Verifier })} />}
          <Segmented label="Thinking tokens per answer" options={LAB_THINKING} value={c.thinking} onChange={(v) => set({ thinking: v as number })} />
          {(L.tier || L.strategy) && <div style={{ fontSize: 12, color: 'var(--text-faint)' }}>Settings with a single option are fixed by this brief.</div>}
        </div>
        <div style={{ flex: '1 1 280px', minWidth: 0 }}>
          <LiveCode note="The same choice in code (sketch)" code={buildReasoningCode(c)} />
        </div>
      </div>
    </LabShell>
  );
}
