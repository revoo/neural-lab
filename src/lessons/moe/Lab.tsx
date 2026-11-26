// authored-by: Claude Opus 5.5
import { useEffect, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Segmented } from '../../components/controls';
import { ExpertKey, EXPERT_COLOURS } from './Widgets';
import {
  ACC_BAR,
  BRIEFS,
  DROP_BAR,
  LAB_ALPHA,
  LAB_CAPACITY,
  LAB_K,
  LAB_NOISE,
  labHint,
  MAX_LOAD_BAR,
  passesBrief1,
  passesBrief2,
  passesBrief3,
  scoreSetting,
  TIGHT_C,
  type LabScore,
  type LabSetting,
} from './lab';
import { LAB_RESULTS } from './results';

const KEY = 'moe-lab-setting-v1';
const INIT: LabSetting = { k: 1, alpha: 0, capacity: 1, noise: 0 };
const same = (a: LabSetting, b: LabSetting) => a.k === b.k && a.alpha === b.alpha && a.capacity === b.capacity && a.noise === b.noise;

function loadSetting(): LabSetting {
  try {
    const s = JSON.parse(localStorage.getItem(KEY) ?? 'null');
    if (s && LAB_K.includes(s.k) && (LAB_ALPHA as readonly number[]).includes(s.alpha) && (LAB_CAPACITY as readonly number[]).includes(s.capacity) && (LAB_NOISE as readonly number[]).includes(s.noise)) return s;
  } catch {
    /* ignore */
  }
  return INIT;
}

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

export function buildMoeCode(s: LabSetting) {
  return [
    `scores = router(x)                          # [tokens, 4]`,
    ...(s.noise ? [`scores = scores + ${s.noise} * torch.randn_like(scores)   # training only`] : []),
    `probs = scores.softmax(-1)`,
    `top = scores.topk(${s.k}, dim=-1).indices      # each token's ${s.k === 1 ? 'expert' : 'two experts'}`,
    `capacity = math.ceil(${s.capacity} * ${s.k} * tokens / 4)`,
    `f = one_hot(top, 4).float().mean((0, 1))    # share of slots per expert`,
    `aux = ${s.alpha} * 4 * (f * probs.mean(0)).sum()`,
    `loss = task_loss + aux`,
  ].join('\n');
}

export function BalanceLab() {
  const progress = useLabProgress('moe-lab-v1');
  const log = useMachineLog();
  const [active, setActive] = useState(0);
  const [setting, setSetting] = useState<LabSetting>(loadSetting);
  const [score, setScore] = useState<LabScore | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(setting));
    } catch {
      /* ignore */
    }
  }, [setting]);

  const tested = score && same(score, setting) ? score : null;
  const briefId = BRIEFS[active].id;
  const passing = tested !== null && (briefId === 'balance' ? passesBrief1(tested) : briefId === 'tight' ? passesBrief2(tested) : passesBrief3(tested));
  const optimal = briefId === 'cheap' && passing && tested!.cost <= 1.25;

  const run = () => {
    setBusy(true);
    // let the button repaint before 5 seeds × 600 steps run on the main thread
    setTimeout(() => {
      const r = scoreSetting(setting);
      setScore(r);
      setBusy(false);
      const ok = passesBrief1(r) || passesBrief3(r);
      log.push(
        `top-${r.k}, α=${r.alpha}, C=${r.capacity}, noise=${r.noise} → accuracy ${pct(r.acc)}, busiest expert ${pct(r.maxLoad)}, dropped ${pct(r.drop)}, ${r.cost.toFixed(2)} slots/token`,
        ok ? 'good' : undefined,
      );
    }, 30);
  };

  const briefs: LabBrief[] = [
    {
      id: 'balance',
      tab: BRIEFS[0].tab,
      title: 'Brief 1 — No idle experts',
      text: `You are paying for four experts' memory; make all four work. On every one of the five seeds, no expert may take more than ${100 * MAX_LOAD_BAR}% of the token slots, and mean test accuracy must stay at ${100 * ACC_BAR}% or more.`,
    },
    {
      id: 'tight',
      tab: BRIEFS[1].tab,
      title: 'Brief 2 — Tight buffers',
      text: `Each expert's buffer is reserved in advance. Use a capacity factor of at most ${TIGHT_C}, drop at most ${100 * DROP_BAR}% of token slots (averaged over the seeds, last 100 steps), and keep accuracy at ${100 * ACC_BAR}% or more.`,
    },
    {
      id: 'cheap',
      tab: BRIEFS[2].tab,
      title: 'Brief 3 — The cheapest layer',
      text: (
        <>
          The hardware reserves <strong>k × C</strong> expert slots per token, used or not. Meet Brief 2’s drop and accuracy bars at any capacity, then (optional goal)
          make k × C as small as possible. Best possible over the lab’s 120 settings, found by exhaustive search: 1.25.
        </>
      ),
    },
  ];

  let status = 'Choose the four settings, then press “Train 5 seeds”. Each run trains the layer from scratch on seeds 1 to 5 (about a second in all).';
  if (busy) status = 'Training 5 seeds…';
  else if (tested) {
    const base = `Mean accuracy ${pct(tested.acc)} · busiest expert on the worst seed ${pct(tested.maxLoad)} · quietest ${pct(tested.minLoad)} · dropped ${pct(tested.drop)} · ${tested.cost.toFixed(2)} expert slots per token.`;
    status = passing ? `✓ Brief passed${optimal ? ', at the lowest cost found' : ''}. ${base}` : `Not yet. ${base}`;
  } else if (score) status = 'The settings changed since the last run: train again to score them.';

  const b1 = LAB_RESULTS.filter(passesBrief1).length;
  const withCost = LAB_RESULTS.map((r) => ({ ...r, cost: r.k * r.capacity }));
  const b2 = withCost.filter(passesBrief2).length;
  const winners = withCost.filter((r) => passesBrief3(r) && r.cost === 1.25);
  const reveal = (
    <div>
      <p style={{ margin: '0 0 8px' }}>
        <strong>Exhaustive search over all 120 settings</strong> (2 × 5 × 4 × 3, each on seeds 1–5): {b1} pass Brief 1, {b2} pass Brief 2, and the cheapest layer that
        meets Brief 3’s bars reserves 1.25 slots per token. {winners.length} settings reach it, all with top-1 routing, C = 1.25 and α of 0.1 or 0.3, with any router
        noise. The cheapest top-2 layer that passes needs 2.5.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="num-table" style={{ fontSize: 12.5 }}>
          <thead>
            <tr>
              <th>top-k</th>
              <th>α</th>
              <th>C</th>
              <th>noise</th>
              <th>accuracy</th>
              <th>busiest</th>
              <th>dropped</th>
            </tr>
          </thead>
          <tbody>
            {winners.map((r, i) => (
              <tr key={i}>
                <td>{r.k}</td>
                <td>{r.alpha}</td>
                <td>{r.capacity}</td>
                <td>{r.noise}</td>
                <td className="mono">{pct(r.acc)}</td>
                <td className="mono">{pct(r.maxLoad)}</td>
                <td className="mono">{pct(r.drop)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <LabShell
      title="Lab: Balance the Experts"
      caption="A 4-expert layer on the four-cluster task. Every setting is trained from scratch on the same five seeds and scored on what they do together."
      briefs={briefs}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={status}
      hint={tested ? labHint(briefId, tested) : 'Train a setting first: the hint reads its measurements.'}
      reveal={{ label: 'the exhaustive search of all 120 settings', content: reveal }}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Press “Train 5 seeds” to log a setting.' }}
      actions={
        <button type="button" className="btn" onClick={() => setSetting(INIT)}>
          Reset
        </button>
      }
    >
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 260px', display: 'grid', gap: 10 }}>
          <Segmented label="Experts per token" options={LAB_K.map((k) => ({ value: k, label: `top-${k}` }))} value={setting.k} onChange={(v) => setSetting((s) => ({ ...s, k: v as 1 | 2 }))} />
          <Segmented label="Balancing weight" options={LAB_ALPHA.map((a) => ({ value: a, label: `α = ${a}` }))} value={setting.alpha} onChange={(v) => setSetting((s) => ({ ...s, alpha: v as number }))} />
          <Segmented label="Capacity factor" options={LAB_CAPACITY.map((c) => ({ value: c, label: `C = ${c}` }))} value={setting.capacity} onChange={(v) => setSetting((s) => ({ ...s, capacity: v as number }))} />
          <Segmented label="Router noise while training" options={LAB_NOISE.map((n) => ({ value: n, label: `σ = ${n}` }))} value={setting.noise} onChange={(v) => setSetting((s) => ({ ...s, noise: v as number }))} />
          <div className="btn-row">
            <button type="button" className="btn primary" disabled={busy} onClick={run}>
              {busy ? 'Training…' : 'Train 5 seeds'}
            </button>
          </div>
        </div>
        <div style={{ flex: '1 1 300px', minWidth: 0 }}>
          {tested ? (
            <div>
              <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginBottom: 4 }}>Each seed’s share of token slots per expert</div>
              {tested.perSeed.map((p, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, fontSize: 12.5 }}>
                  <span className="mono" style={{ flex: '0 0 auto', whiteSpace: 'nowrap' }}>
                    seed {i + 1}
                  </span>
                  <div style={{ flex: 1, minWidth: 60, display: 'flex', height: 14, borderRadius: 3, overflow: 'hidden', position: 'relative', background: 'var(--bg-sunken)' }}>
                    {p.load.map((l, e) => (
                      <div key={e} style={{ width: `${100 * l}%`, background: EXPERT_COLOURS[e] }} title={`expert ${e + 1}: ${pct(l)}`} />
                    ))}
                  </div>
                  <span className="mono" style={{ flex: '0 0 auto', whiteSpace: 'nowrap', fontSize: 11.5, textAlign: 'right' }}>
                    {pct(p.acc)} · {pct(p.drop)}
                  </span>
                </div>
              ))}
              <div style={{ fontSize: 12, color: 'var(--text-faint)', textAlign: 'right' }}>accuracy · dropped</div>
              <ExpertKey />
            </div>
          ) : (
            <div style={{ fontSize: 13, color: 'var(--text-muted)', padding: '20px 0' }}>Results for each seed appear here after training.</div>
          )}
          <LiveCode note="The same layer in PyTorch (sketch)" code={buildMoeCode(setting)} />
        </div>
      </div>
    </LabShell>
  );
}
