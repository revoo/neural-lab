import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Toggle } from '../../components/controls';
import { Plot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import { LogChart } from '../../components/LogChart';
import { Clinic, DOMAIN, classBalance, duplicateFraction, inputRanges, neighbourAgreement, paramCount, type Config, type Pt } from './clinic';
import { CASES, CAUSES, KNOBS, singleChanges, type Case, type CauseId } from './cases';
import { FIXES } from './fixes';

const STORE = 'nl-clinic-v1';
type Saved = { diagnosed: Record<string, boolean>; wrong: Record<string, CauseId[]>; fixes: Record<string, string[]>; fixed: Record<string, boolean> };
const load = (): Saved => {
  try {
    return { diagnosed: {}, wrong: {}, fixes: {}, fixed: {}, ...JSON.parse(localStorage.getItem(STORE) ?? '{}') };
  } catch {
    return { diagnosed: {}, wrong: {}, fixes: {}, fixed: {} };
  }
};

const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(0)}%` : '—');
const num = (v: number) => (!Number.isFinite(v) ? 'NaN' : v === 0 ? '0' : Math.abs(v) < 0.001 ? v.toExponential(0) : v.toFixed(3));
const hiddenLabel = (h: number[]) => (h.length ? h.join(' × ') : 'none');

/** One-change fixes recorded by the exhaustive search for a case. */
export const fixesFor = (c: Case) => (FIXES[c.id] ?? []).filter((r) => r.fresh >= c.target);

// ------------------------------------------------------------------ evidence

function Boundary({ run, size = 300 }: { run: Clinic; size?: number }) {
  const f = useMemo(() => (x: number, y: number) => {
    const p = run.predict(x, y);
    return Number.isFinite(p) ? p : 0.5;
  }, [run]);
  const cmap = useMemo(() => (t: number) => diverging(1 - t), []);
  const dot = (p: Pt, i: number, hollow: boolean) => ({ p, i, hollow });
  const pts = [...run.data.train.slice(0, 300).map((p, i) => dot(p, i, false)), ...run.data.test.slice(0, 300).map((p, i) => dot(p, i, true))];
  return (
    <Plot x={[-DOMAIN, DOMAIN]} y={[-DOMAIN, DOMAIN]} width={size} height={size} pad={10} grid={false} axes={false} ariaLabel="The model's prediction everywhere, with training and test points">
      <Heatmap f={f} range={[0, 1]} colormap={cmap} resolution={60} opacity={0.75} />
      {({ x, y }) => (
        <g>
          {pts.map(({ p, i, hollow }) => (
            <circle
              key={(hollow ? 't' : 'p') + i}
              cx={x(p.x)}
              cy={y(p.y)}
              r={hollow ? 3.2 : 3}
              fill={hollow ? 'none' : p.label ? 'var(--c-b)' : 'var(--c-a)'}
              stroke={hollow ? (p.label ? 'var(--c-b)' : 'var(--c-a)') : 'var(--bg-elev)'}
              strokeWidth={hollow ? 1.4 : 0.8}
            />
          ))}
        </g>
      )}
    </Plot>
  );
}

function AccuracyChart({ run }: { run: Clinic }) {
  const H = run.history;
  const max = run.cfg.steps;
  return (
    <Plot x={[0, max]} y={[0, 1]} width={380} height={170} grid={false} axes={false} pad={36} ariaLabel="Accuracy during training">
      {({ x, y }) => (
        <g>
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line className="plot-grid" x1={x(0)} x2={x(max)} y1={y(v)} y2={y(v)} />
              <text className="plot-tick" x={x(0) - 5} y={y(v) + 3.5} textAnchor="end">
                {v * 100}%
              </text>
            </g>
          ))}
          <text className="plot-tick" x={x(max)} y={y(0) + 14} textAnchor="end">
            {max} steps
          </text>
          {(
            [
              ['trainAcc', 'var(--c-a)', undefined],
              ['testAcc', 'var(--c-b)', '6 4'],
              ['freshAcc', 'var(--c-c)', undefined],
            ] as const
          ).map(([k, color, dash]) => (
            <polyline
              key={k}
              points={H.map((h) => `${x(h.step)},${y(Number.isFinite(h[k]) ? h[k] : 0)}`).join(' ')}
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeDasharray={dash}
            />
          ))}
        </g>
      )}
    </Plot>
  );
}

function BatchStrip({ mix }: { mix: number[] }) {
  return (
    <svg viewBox={`0 0 ${mix.length * 3} 16`} width={mix.length * 3} height={16} role="img" aria-label="Fraction of class-1 examples in each of the last batches">
      {mix.map((m, i) => (
        <rect key={i} x={i * 3} y={0} width={2.4} height={16} fill="var(--c-b)" opacity={0.12 + 0.88 * m} />
      ))}
    </svg>
  );
}

function Evidence({ run }: { run: Clinic }) {
  const stats = run.layerStats();
  const L = run.last;
  const ranges = inputRanges(run);
  const dup = duplicateFraction(run);
  const agree = neighbourAgreement(run);
  const cfg = run.cfg;
  const trouble = cfg.activation === 'relu' ? 'dead' : 'saturated';
  return (
    <div className="clinic-evidence">
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '0 1 300px' }}>
          <div className="field-label">the model's predictions</div>
          <Boundary run={run} />
          <div className="muted" style={{ fontSize: 12 }}>
            Filled points: training data. Hollow: test data. Orange and blue are the two classes; the background is the model's prediction.
          </div>
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <div className="field-label">loss during training (log scale)</div>
          <LogChart
            series={[
              { points: run.history.filter((h) => Number.isFinite(h.trainLoss)).map((h) => [h.step, Math.max(h.trainLoss, 1e-4)]), color: 'var(--c-a)' },
              { points: run.history.filter((h) => Number.isFinite(h.testLoss)).map((h) => [h.step, Math.max(h.testLoss, 1e-4)]), color: 'var(--c-b)', dashed: true },
            ]}
            xDomain={[0, cfg.steps]}
            yDomain={[0.001, 100]}
            xLabel="steps"
            xTicks={[0, cfg.steps]}
            hlines={[{ y: Math.LN2, label: 'ln 2: a 50–50 shrug', color: 'var(--text-faint)' }]}
            width={380}
            height={170}
            ariaLabel="Training and test loss during training"
          />
          <div className="field-label">accuracy during training</div>
          <AccuracyChart run={run} />
          <div className="muted" style={{ fontSize: 12 }}>
            <span className="c-a">training</span>, <span className="c-b">test</span> (dashed) and <span className="c-c">fresh data from the real world</span>.
          </div>
        </div>
      </div>
      <div className="widget-row" style={{ marginTop: 10 }}>
        <div className="widget-col" style={{ flex: '1 1 200px' }}>
          <div className="field-label">final scores</div>
          <div className="readout">
            <span className="k">training   </span> {pct(L.trainAcc)} <span className="k">loss</span> {num(L.trainLoss)}
            {'\n'}
            <span className="k">test       </span> {pct(L.testAcc)} <span className="k">loss</span> {num(L.testLoss)}
            {'\n'}
            <span className="k">fresh data </span> <span className="c-c">{pct(L.freshAcc)}</span>
          </div>
        </div>
        <div className="widget-col" style={{ flex: '1 1 240px' }}>
          <div className="field-label">inside the network</div>
          <div className="readout">
            {stats.length === 0 ? (
              <span className="k">no hidden layers</span>
            ) : (
              <>
                <span className="k">layer  {trouble.padEnd(10)} weights  grads</span>
                {stats.map((s, l) => (
                  <span key={l}>
                    {'\n'}
                    {String(l + 1).padEnd(7)}
                    {`${s.dead + s.saturated}/${s.size}`.padEnd(11)}
                    {num(s.weightRms).padEnd(9)}
                    {num(s.gradRms)}
                  </span>
                ))}
                {'\n'}
                <span className="k">{trouble} at step 0: </span>
                {pct(run.initialTrouble)}
              </>
            )}
          </div>
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="field-label">data checks</div>
          <div className="readout">
            <span className="k">input 1     </span> {ranges[0].min.toFixed(1)} … {ranges[0].max.toFixed(1)}
            {'\n'}
            <span className="k">input 2     </span> {ranges[1].min.toFixed(1)} … {ranges[1].max.toFixed(1)}
            {'\n'}
            <span className="k">test copies </span> {pct(dup)} <span className="k">of test = training points</span>
            {'\n'}
            <span className="k">labels      </span> {pct(agree)} <span className="k">agree with neighbours</span>
            {'\n'}
            <span className="k">balance     </span> {pct(classBalance(run))} / {pct(1 - classBalance(run))}
            {'\n'}
            <span className="k">last 120 batches (share of orange):</span>
          </div>
          <BatchStrip mix={run.batchMix} />
        </div>
      </div>
    </div>
  );
}

function ConfigCard({ cfg }: { cfg: Config }) {
  return (
    <div className="readout" style={{ fontSize: 12.5 }}>
      <span className="k">data      </span> {cfg.dataset}, {cfg.n} points ({cfg.n / 2} train / {cfg.n / 2} test), noise {cfg.noise}
      {'\n'}
      <span className="k">network   </span> 2 inputs → hidden {hiddenLabel(cfg.hidden)} {cfg.hidden.length ? cfg.activation : ''} → 1 output ({paramCount(cfg)} parameters)
      {'\n'}
      <span className="k">training  </span> {cfg.optimizer}, step size {cfg.lr}, batches of {cfg.batch}, {cfg.steps} steps{cfg.l2 ? `, weight decay ${cfg.l2}` : ''}
    </div>
  );
}

// ------------------------------------------------------------------ the fix bench

function sameCfg(a: Config, b: Config) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function FixBench({ c, saved, onFix }: { c: Case; saved: Saved; onFix: (label: string | null, ok: boolean) => void }) {
  const [cfg, setCfg] = useState<Config>(c.cfg);
  const [result, setResult] = useState<{ cfg: Config; run: Clinic } | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setCfg(c.cfg);
    setResult(null);
  }, [c]);
  const changes = singleChanges(c.cfg);
  const match = changes.find((ch) => sameCfg(ch.apply(c.cfg), cfg));
  const nChanged = (Object.keys(cfg) as (keyof Config)[]).filter((k) => JSON.stringify(cfg[k]) !== JSON.stringify(c.cfg[k])).length;
  const all = fixesFor(c);
  const found = saved.fixes[c.id] ?? [];
  const set = <K extends keyof Config>(k: K, v: Config[K]) => setCfg({ ...cfg, [k]: v });

  const run = () => {
    setBusy(true);
    setTimeout(() => {
      const r = new Clinic(cfg).run(25);
      setResult({ cfg, run: r });
      setBusy(false);
      onFix(match && nChanged === 1 ? match.label : null, r.last.freshAcc >= c.target);
    }, 20);
  };
  const ok = result && result.run.last.freshAcc >= c.target;

  return (
    <div className="clinic-fix">
      <h4 style={{ margin: '0 0 6px' }}>Fix it</h4>
      <p className="muted" style={{ margin: '0 0 10px', fontSize: 13.5 }}>
        Change anything you like, then train. The goal: at least <strong>{pct(c.target)}</strong> accuracy on <strong>fresh data</strong>. Optional: how many different{' '}
        <em>single</em> changes can you find that fix it on their own?
      </p>
      <div className="controls inline" style={{ gap: '10px 16px' }}>
        <label className="clinic-select">
          <span className="field-label">step size</span>
          <select value={cfg.lr} onChange={(e) => set('lr', Number(e.target.value))}>
            {[...new Set([...KNOBS.lr, c.cfg.lr])].sort((a, b) => a - b).map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <Segmented label="optimizer" value={cfg.optimizer} onChange={(v) => set('optimizer', v)} options={[...KNOBS.optimizer]} />
        <Segmented label="activation" value={cfg.activation} onChange={(v) => set('activation', v)} options={[...KNOBS.activation]} />
        <label className="clinic-select">
          <span className="field-label">hidden layers</span>
          <select value={JSON.stringify(cfg.hidden)} onChange={(e) => set('hidden', JSON.parse(e.target.value))}>
            {[...new Set([...KNOBS.hidden, c.cfg.hidden].map((h) => JSON.stringify(h)))].map((h) => (
              <option key={h} value={h}>
                {hiddenLabel(JSON.parse(h))}
              </option>
            ))}
          </select>
        </label>
        <Segmented label="weight decay" value={cfg.l2} onChange={(v) => set('l2', v)} options={KNOBS.l2} />
        <Segmented label="batch size" value={cfg.batch} onChange={(v) => set('batch', v)} options={KNOBS.batch} />
        <Segmented label="steps" value={cfg.steps} onChange={(v) => set('steps', v)} options={[...new Set([...KNOBS.steps, c.cfg.steps])].sort((a, b) => a - b)} />
        {c.cfg.init !== 'standard' && (
          <Segmented
            label="initialisation"
            value={cfg.init}
            onChange={(v) => set('init', v)}
            options={[
              { value: c.cfg.init, label: 'as it was' },
              { value: 'standard', label: 'standard' },
            ]}
          />
        )}
      </div>
      <div className="controls inline" style={{ marginTop: 10, gap: '8px 18px' }}>
        <Toggle label="collect 4× more data" checked={cfg.n !== c.cfg.n} onChange={(v) => set('n', v ? c.cfg.n * 4 : c.cfg.n)} />
        <Toggle label="normalise the inputs" checked={cfg.normalise} onChange={(v) => set('normalise', v)} />
        <Toggle label="shuffle the training data" checked={cfg.shuffle} onChange={(v) => set('shuffle', v)} />
        {c.cfg.labelNoise > 0 && <Toggle label="fix the labels" checked={cfg.labelNoise === 0} onChange={(v) => set('labelNoise', v ? 0 : c.cfg.labelNoise)} />}
        {c.cfg.leak > 0 && <Toggle label="remove duplicates from the test set" checked={cfg.leak === 0} onChange={(v) => set('leak', v ? 0 : c.cfg.leak)} />}
      </div>
      <div className="btn-row" style={{ marginTop: 12 }}>
        <button className="btn primary" onClick={run} disabled={busy || nChanged === 0}>
          {busy ? 'Training…' : '▶ Train with these settings'}
        </button>
        <button className="btn" onClick={() => setCfg(c.cfg)} disabled={nChanged === 0}>
          Undo all changes
        </button>
        <span className="muted" style={{ fontSize: 13 }}>
          {nChanged === 0 ? 'No changes yet.' : nChanged === 1 ? `One change: ${match?.label ?? '—'}` : `${nChanged} changes`}
        </span>
      </div>
      {result && (
        <div className="widget-row" style={{ marginTop: 12 }}>
          <div className="widget-col" style={{ flex: '0 1 220px' }}>
            <Boundary run={result.run} size={210} />
          </div>
          <div className="widget-col">
            <div className="readout">
              <span className="k">training   </span> {pct(result.run.last.trainAcc)}
              {'\n'}
              <span className="k">test       </span> {pct(result.run.last.testAcc)}
              {'\n'}
              <span className="k">fresh data </span> <strong>{pct(result.run.last.freshAcc)}</strong> <span className="k">(target {pct(c.target)})</span>
            </div>
            <div className={`live-note`} style={{ marginTop: 8 }}>
              {ok ? (
                <>
                  <strong>Fixed.</strong> {pct(result.run.last.freshAcc)} on fresh data.{' '}
                  {sameCfg(result.cfg, cfg) && nChanged === 1 && match ? `With a single change: ${match.label}.` : 'Can you do it by changing just one thing?'}
                </>
              ) : (
                <>
                  Not yet: {pct(result.run.last.freshAcc)} on fresh data, short of {pct(c.target)}.
                  {c.id === 'too-good' && result.cfg.leak === 0 && result.cfg.n === c.cfg.n
                    ? ' Removing the duplicates made the test score honest, but it can’t make the model itself any better. Now treat what the honest score reveals.'
                    : ''}
                </>
              )}
            </div>
          </div>
        </div>
      )}
      <div style={{ marginTop: 12, fontSize: 13.5 }}>
        <strong>
          One-change fixes found: {found.length} of {all.length}
        </strong>
        <span className="muted">
          {' '}
          (counted by training every single change in the bench's menu — {changes.length} of them for this case — and checking each against the target)
        </span>
        {found.length > 0 && <div className="muted">{found.join(' · ')}</div>}
        {found.length === all.length && all.length > 0 && <div className="c-good">You've found every one.</div>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ the clinic

export function ModelClinic() {
  const [idx, setIdx] = useState(0);
  const [saved, setSaved] = useState<Saved>(load);
  const [run, setRun] = useState<Clinic | null>(null);
  const c = CASES[idx];
  useEffect(() => {
    setRun(null);
    const id = setTimeout(() => setRun(new Clinic(c.cfg).run(10)), 20);
    return () => clearTimeout(id);
  }, [c]);
  const save = (s: Saved) => {
    setSaved(s);
    try {
      localStorage.setItem(STORE, JSON.stringify(s));
    } catch {
      /* ignore */
    }
  };
  const wrong = saved.wrong[c.id] ?? [];
  const diagnosed = !!saved.diagnosed[c.id];
  const [lastPick, setLastPick] = useState<CauseId | null>(null);
  useEffect(() => setLastPick(null), [c]);

  const pick = (id: CauseId) => {
    setLastPick(id);
    if (id === c.cause) save({ ...saved, diagnosed: { ...saved.diagnosed, [c.id]: true } });
    else if (!wrong.includes(id)) save({ ...saved, wrong: { ...saved.wrong, [c.id]: [...wrong, id] } });
  };
  const onFix = (label: string | null, ok: boolean) => {
    const found = saved.fixes[c.id] ?? [];
    save({
      ...saved,
      fixed: ok ? { ...saved.fixed, [c.id]: true } : saved.fixed,
      fixes: ok && label && !found.includes(label) ? { ...saved.fixes, [c.id]: [...found, label] } : saved.fixes,
    });
  };

  let feedback: ReactNode = null;
  if (diagnosed && (lastPick === null || lastPick === c.cause))
    feedback = (
      <div className="feedback good">
        <div className="feedback-title">✓ {CAUSES.find((k) => k.id === c.cause)!.name}</div>
        <p>{c.explain}</p>
      </div>
    );
  else if (lastPick && lastPick !== c.cause)
    feedback = (
      <div className="feedback bad">
        <div className="feedback-title">Not quite: {CAUSES.find((k) => k.id === lastPick)!.name.toLowerCase()}</div>
        <p>{c.ruleOut[lastPick] ?? 'Look at the evidence again: is there anything that specifically points to this? Every cause leaves its own fingerprints.'}</p>
      </div>
    );

  const nDiag = CASES.filter((k) => saved.diagnosed[k.id]).length;
  return (
    <Widget
      title="The model clinic"
      wide
      caption={
        <>
          Every case is a real training run, recomputed in your browser from the configuration shown. Study the evidence, choose the root cause, then try to fix it. The
          “fresh data” score is the model's accuracy on new data from the real world: the number that actually matters. Your progress is saved in this browser.
        </>
      }
    >
      <div className="clinic-tabs" role="tablist" aria-label="Case files">
        {CASES.map((k, i) => (
          <button key={k.id} role="tab" aria-selected={i === idx} className={i === idx ? 'on' : ''} onClick={() => setIdx(i)} title={k.title}>
            {i + 1}
            {saved.diagnosed[k.id] ? ' ✓' : ''}
            {saved.fixed[k.id] ? '★' : ''}
          </button>
        ))}
        <span className="muted" style={{ fontSize: 12.5, marginLeft: 6 }}>
          {nDiag}/{CASES.length} diagnosed
        </span>
      </div>
      <div className="clinic-brief">
        <div className="field-label">
          case {idx + 1}: {c.title}
        </div>
        <p style={{ margin: '4px 0 8px' }}>“{c.brief}”</p>
        <ConfigCard cfg={c.cfg} />
      </div>
      {run ? <Evidence run={run} /> : <div className="muted" style={{ padding: 20 }}>Training the case…</div>}
      <div className="clinic-diagnose">
        <h4 style={{ margin: '14px 0 6px' }}>Your diagnosis: what is the root cause?</h4>
        <div className="clinic-causes">
          {CAUSES.map((k) => (
            <button
              key={k.id}
              className={`option${diagnosed && k.id === c.cause ? ' correct' : wrong.includes(k.id) ? ' wrong' : ''}`}
              onClick={() => pick(k.id)}
            >
              {k.name}
            </button>
          ))}
        </div>
        {feedback}
      </div>
      <FixBench c={c} saved={saved} onFix={onFix} />
    </Widget>
  );
}
