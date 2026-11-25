// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Arrow, Handle, Plot, usePlot } from '../../components/plot';
import { AnswerReveal } from '../../components/AnswerReveal';
import { LiveCode } from '../../components/LabJob';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import {
  GUIDE_BRIEFS,
  LAB_STEPS,
  LAB_WS,
  ON_SHAPE,
  TRIBES,
  classProbs,
  ddim,
  guess,
  guided,
  guideExhaustive,
  guideLabHint,
  judge,
  labPass,
  labTry,
  noiseStart,
  stops,
  abarAt,
  type GuideBrief,
  type Pt,
} from './guide';
import { guidanceRealCode } from './realCode';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const VIEW: [number, number] = [-2.6, 2.6];
const CLS_COLOR = ['var(--c-a)', 'var(--c-b)'];
const N = 2000;

type PlotChildren = Parameters<typeof Plot>[0]['children'];
type PlotChild = Exclude<PlotChildren, unknown[] | undefined>;

function Scatter({ children, size = 330, label, onPointerDown }: { children: PlotChildren; size?: number; label: string; onPointerDown?: (p: { x: number; y: number }) => void }) {
  const bg: PlotChild = ({ x, y }) => <rect x={x(VIEW[0])} y={y(VIEW[1])} width={x(VIEW[1]) - x(VIEW[0])} height={y(VIEW[0]) - y(VIEW[1])} fill="var(--bg-sunken)" rx={6} />;
  const kids = [bg, ...((Array.isArray(children) ? children : [children]) as PlotChild[])];
  return (
    <Plot width={size} height={size} x={VIEW} y={VIEW} grid={false} axes={false} ariaLabel={label} onPointerDown={onPointerDown} style={{ maxWidth: '100%', height: 'auto', touchAction: 'none' }} children={kids} />
  );
}

function Blobs({ highlight = null }: { highlight?: number | null }) {
  const { x, y } = usePlot();
  return (
    <g fill="none" strokeWidth={1.6} strokeDasharray="3 3">
      {TRIBES.m.map((m, k) => (
        <g key={k}>
          <circle cx={x(m[0])} cy={y(m[1])} r={x(ON_SHAPE * TRIBES.s[k]) - x(0)} stroke={CLS_COLOR[TRIBES.cls[k]]} opacity={highlight === null || highlight === TRIBES.cls[k] ? 0.9 : 0.3} />
          <text x={x(m[0])} y={y(m[1]) - (x(ON_SHAPE * TRIBES.s[k]) - x(0)) - 4} textAnchor="middle" fontSize={10.5} fill={CLS_COLOR[TRIBES.cls[k]]} stroke="none" fontWeight={600}>
            {TRIBES.cls[k] === 0 ? `B${k + 1}` : `O${k - 2}`}
          </text>
        </g>
      ))}
    </g>
  );
}

function Dots({ pts, color = 'var(--accent)', r = 1.5, opacity = 0.45 }: { pts: Pt[]; color?: string; r?: number; opacity?: number }) {
  const { x, y } = usePlot();
  return (
    <g fill={color} opacity={opacity}>
      {pts.map((p, i) => (p[0] < VIEW[0] || p[0] > VIEW[1] || p[1] < VIEW[0] || p[1] > VIEW[1] ? null : <circle key={i} cx={x(p[0])} cy={y(p[1])} r={r} />))}
    </g>
  );
}

/** Bars: how the requested class's samples split between its three blobs. */
function BlobShares({ within, cls }: { within: number[]; cls: number }) {
  const names = cls === 0 ? ['B1 (outer)', 'B2 (middle)', 'B3 (inner)'] : ['O1 (inner)', 'O2 (middle)', 'O3 (outer)'];
  return (
    <div style={{ display: 'grid', gap: 4 }}>
      {within.map((v, i) => (
        <div key={i} style={{ display: 'grid', gridTemplateColumns: '92px 1fr 52px', gap: 8, alignItems: 'center', fontSize: 12.5 }}>
          <span style={{ color: 'var(--text-muted)' }}>{names[i]}</span>
          <div className="progress-bar" style={{ height: 8, position: 'relative' }}>
            <div style={{ width: `${100 * v}%`, background: CLS_COLOR[cls] }} />
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', textAlign: 'right' }}>{pct(v)}</span>
        </div>
      ))}
      <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>fair share: 33.3% each</div>
    </div>
  );
}

// ------------------------------------------------------------------ 1. Ask for a class

export function AskForClass() {
  const [cls, setCls] = useState<number | null>(0);
  const [showPaths, setShowPaths] = useState(false);
  const run = useMemo(() => ddim(guided(TRIBES, cls, 1), stops(50), noiseStart(N, 5), true), [cls]);
  const j = useMemo(() => judge(TRIBES, run.xs, cls ?? 0), [run, cls]);
  const j1 = useMemo(() => (cls === null ? judge(TRIBES, run.xs, 1) : null), [run, cls]);
  const paths = useMemo(() => (showPaths ? Array.from({ length: 36 }, (_, i) => run.path.map((xs) => xs[i * 11])) : []), [showPaths, run]);

  return (
    <Widget
      title="Ask for a class"
      wide
      caption={
        <>
          Six blobs in two classes: three <span style={{ color: 'var(--c-a)' }}>blue</span> on the left, three <span style={{ color: 'var(--c-b)' }}>orange</span> on the right. The innermost blob of each class sits close to the other class. Choose what to ask for and the sampler runs 50 DDIM steps from 2,000 points of pure noise, with the <strong>exact</strong> denoiser: unconditional, or conditioned on the label (its best guess only considers that class’s blobs).
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <Segmented
          value={cls === null ? -1 : cls}
          onChange={(v) => setCls(v === -1 ? null : v)}
          options={[
            { value: -1, label: 'No label' },
            { value: 0, label: 'Blue' },
            { value: 1, label: 'Orange' },
          ]}
        />
        <button type="button" className="btn small" onClick={() => setShowPaths((s) => !s)}>{showPaths ? 'Hide paths' : 'Show 36 paths'}</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <Scatter label="Samples, conditioned or not">
          <Blobs highlight={cls} />
          <Dots pts={run.xs} color={cls === null ? 'var(--text-muted)' : CLS_COLOR[cls]} />
          {({ x, y }) => (
            <g fill="none" stroke="var(--accent)" strokeWidth={1} opacity={0.75}>
              {paths.map((p, i) => (
                <polyline key={i} points={p.map((q) => `${x(Math.max(VIEW[0], Math.min(VIEW[1], q[0])))},${y(Math.max(VIEW[0], Math.min(VIEW[1], q[1])))}`).join(' ')} />
              ))}
            </g>
          )}
        </Scatter>
        <div style={{ display: 'grid', gap: 10 }}>
          {cls === null ? (
            <div className="readout">
              <span className="k">on blue blobs </span>
              {pct(j.onClass)}
              {'\n'}
              <span className="k">on orange blobs </span>
              {pct(j1!.onClass)}
              {'\n'}
              <span className="k">on no blob </span>
              {pct(j.offShape)}
            </div>
          ) : (
            <>
              <div className="readout">
                <span className="k">on a {TRIBES.classNames[cls]} blob </span>
                {pct(j.onClass)}
                {'\n'}
                <span className="k">on the other class </span>
                {pct(j.wrongClass)}
                {'\n'}
                <span className="k">on no blob </span>
                {pct(j.offShape)}
              </div>
              <BlobShares within={j.withinClass} cls={cls} />
            </>
          )}
          <div className="live-note">
            {cls === null ? (
              <>Without a label you get a fair sample of everything: about half blue, half orange. The sampler has no way to be asked for one or the other.</>
            ) : (
              <>
                Conditioning is just a different best guess: averaging only over the {TRIBES.classNames[cls]} blobs. Every sample lands on a {TRIBES.classNames[cls]} blob ({pct(j.onClass)} on one, none on the other class), split fairly between its three blobs, including the inner one next to the other class. With a perfect conditional denoiser, there is nothing left to fix.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Two pushes

export function TwoPushes() {
  const [t, setT] = useState(500);
  const [probe, setProbe] = useState<Pt>([0.1, 0.3]);
  const a = abarAt(t);
  const u = useMemo(() => guess(TRIBES, a, probe, null).x0, [a, probe]);
  const c = useMemo(() => guess(TRIBES, a, probe, 0).x0, [a, probe]);
  const p = useMemo(() => classProbs(TRIBES, a, probe), [a, probe]);
  const push = Math.hypot(c[0] - u[0], c[1] - u[1]);

  return (
    <Widget
      title="Two pushes"
      wide
      caption={
        <>
          Drag the noisy point and choose a noise level. The grey arrow is the <strong>unconditional</strong> best guess; the blue arrow is the best guess <strong>given the label “blue”</strong>. The dashed arrow between their tips is the difference: the extra push that the label adds, which comes entirely from how sure a classifier would be that this noisy point is blue. Try points near the middle at different noise levels, and points deep inside the blue side.
        </>
      }
    >
      <Slider label="noise level: forward step t" value={t} min={0} max={1000} step={10} onChange={setT} format={(v) => `${Math.round(v)} (ᾱ = ${abarAt(v).toFixed(3)})`} color="c-p" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 8 }}>
        <Scatter label="Unconditional and conditional best guesses" onPointerDown={(q) => setProbe([q.x, q.y])}>
          <Blobs />
          <Arrow from={probe} to={u} color="var(--text-muted)" width={2.2} />
          <Arrow from={probe} to={c} color="var(--c-a)" width={2.6} />
          <Arrow from={u} to={c} color="var(--accent)" width={1.8} dashed />
          <Handle x={probe[0]} y={probe[1]} onDrag={(px, py) => setProbe([px, py])} color="var(--accent)" label="noisy point" />
        </Scatter>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">classifier: chance this noisy point is blue </span>
            {pct(p[0])}
            {'\n'}
            <span className="k">unconditional guess </span>({u[0].toFixed(2)}, {u[1].toFixed(2)}){'\n'}
            <span className="k">guess given “blue” </span>({c[0].toFixed(2)}, {c[1].toFixed(2)}){'\n'}
            <span className="k">the label’s extra push (length) </span>
            {push.toFixed(2)}
          </div>
          <div className="live-note">
            {p[0] > 0.97 ? (
              <>
                The classifier is already {pct(p[0])} sure this point is blue, so the label adds almost nothing: both guesses agree. The label only matters where the classifier has doubts.
              </>
            ) : p[0] < 0.1 ? (
              <>
                This point looks orange ({pct(p[0])} blue). Asking for blue pulls the guess hard across to the blue side: a long extra push.
              </>
            ) : (
              <>
                The classifier is unsure ({pct(p[0])} blue), so the unconditional guess sits between the classes and the label pushes it firmly towards blue. At high noise almost every point is like this, which is why the label’s push is strongest early in sampling.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Turn up the guidance

const DIAL_STEPS = [3, 5, 10, 50];

export function GuidanceDial() {
  const [w, setW] = useState(1);
  const [steps, setSteps] = useState(50);
  const run = useMemo(() => ddim(guided(TRIBES, 0, w), stops(steps), noiseStart(N, 5)).xs, [w, steps]);
  const j = useMemo(() => judge(TRIBES, run, 0), [run]);
  const minBlob = Math.min(...j.withinClass);

  return (
    <Widget
      title="Turn up the guidance"
      wide
      caption={
        <>
          Classifier-free guidance takes the unconditional guess and moves <strong>w times</strong> the label’s extra push from it. w = 0 ignores the label, w = 1 is plain conditioning, and w above 1 exaggerates the label. The denoiser is exact, and we ask for blue. Slide w and watch both where samples land and how they split between the three blue blobs. Then try 3 steps.
        </>
      }
    >
      <Slider label="guidance scale w" value={w} min={0} max={8} step={0.25} onChange={setW} format={(v) => v.toFixed(2)} color="c-p" />
      <div style={{ marginTop: 6 }}>
        <Segmented label="Denoising steps" value={steps} onChange={setSteps} options={DIAL_STEPS.map((s) => ({ value: s, label: String(s) }))} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 10 }}>
        <Scatter label="Guided samples">
          <Blobs highlight={0} />
          <Dots pts={run} color="var(--c-a)" />
        </Scatter>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">on a blue blob </span>
            {pct(j.onClass)}
            {'\n'}
            <span className="k">on an orange blob </span>
            {pct(j.wrongClass)}
            {'\n'}
            <span className="k">on no blob </span>
            {pct(j.offShape)}
          </div>
          <BlobShares within={j.withinClass} cls={0} />
          <div className="live-note">
            {w < 1 ? (
              <>
                Below 1 the label is diluted with the unconditional guess, so some samples wander to orange blobs ({pct(j.wrongClass)}) or get stranded between ({pct(j.offShape)}).
              </>
            ) : w === 1 ? (
              <>
                Plain conditioning: {pct(j.onClass)} on blue, split fairly between the three blobs. {steps <= 5 ? `But with only ${steps} steps, ${pct(j.offShape)} of samples end up between blobs. Try raising w.` : 'With a perfect denoiser and enough steps, this is already the right answer.'}
              </>
            ) : minBlob < 0.05 ? (
              <>
                The guidance has turned a range into a stereotype: {pct(Math.max(...j.withinClass))} of blue samples are on the outer blob B1, the one furthest from orange, and the inner blob B3 is down to {pct(j.withinClass[2])}. Samples are “very blue” ({pct(j.onClass)} on blue){j.offShape > 0.03 ? `, though ${pct(j.offShape)} have been pushed right past the blobs` : ''}. Diversity is what guidance spends.
              </>
            ) : (
              <>
                Above 1, samples are pushed away from anything that looks orange. {steps <= 5 ? `With few steps that helps: ${pct(j.onClass)} land on a blue blob, against fewer at w = 1. ` : ''}But the inner blob B3, the blue that looks a little orange, is losing its share ({pct(j.withinClass[2])} instead of 33.3%).
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tune the guidance

const LAB_KEY = 'nl-guidance-lab-v1';
type Entry = { steps: number; w: number; solved: boolean; fewest: number | null };
type LabState = Record<GuideBrief['id'], Entry>;
const START: Entry = { steps: 3, w: 1, solved: false, fewest: null };

function loadLab(): LabState {
  const d: LabState = { catalogue: { ...START }, poster: { ...START }, both: { ...START } };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return d;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of GUIDE_BRIEFS) {
      const v = parsed[b.id];
      if (!v || !(LAB_STEPS as readonly number[]).includes(v.steps) || !LAB_WS.includes(v.w)) continue;
      d[b.id] = { steps: v.steps, w: v.w, solved: !!v.solved, fewest: typeof v.fewest === 'number' ? v.fewest : null };
    }
  } catch {
    // ignore storage errors
  }
  return d;
}

export function GuidanceLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [hint, setHint] = useState(false);
  const log = useMachineLog(20);

  const brief = GUIDE_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const r = useMemo(() => labTry(entry.steps, entry.w), [entry.steps, entry.w]);
  const runRes = useMemo(() => ddim(guided(TRIBES, 0, entry.w), stops(entry.steps), noiseStart(N, 5), true), [entry.steps, entry.w]);
  const pass = labPass(brief, r);
  const saved = useRef(lab);
  saved.current = lab;

  const maxFrame = Math.max(0, runRes.path.length - 1);
  const replay = useReplay(maxFrame, 420, `${brief.id}:${entry.steps}:${entry.w}`, false);
  const frameIdx = Math.min(replay.i, maxFrame);
  const shownPts = runRes.path[frameIdx] ?? runRes.xs;

  const persist = (next: LabState) => {
    setLab(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };
  const update = (patch: Partial<Entry>) => {
    const nextEntry = { ...entry, ...patch };
    persist({ ...lab, [brief.id]: nextEntry });
    const nr = labTry(nextEntry.steps, nextEntry.w);
    const np = labPass(brief, nr);
    log.push(
      `${brief.id}: steps=${nextEntry.steps}, w=${nextEntry.w.toFixed(2)} → on-class ${pct(nr.onClass)}, min-blob ${pct(nr.minBlob)} (${np ? 'PASS' : 'FAIL'})`,
      np ? 'good' : 'bad',
    );
  };

  useEffect(() => {
    if (!pass) return;
    const cur = saved.current[brief.id];
    if (cur.solved && cur.fewest !== null && cur.fewest <= entry.steps) return;
    persist({ ...saved.current, [brief.id]: { ...cur, solved: true, fewest: cur.fewest === null ? entry.steps : Math.min(cur.fewest, entry.steps) } });
  }, [pass, brief.id, entry.steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const hintText = useMemo(() => guideLabHint(brief, r), [brief, r]);
  const liveCode = useMemo(() => guidanceRealCode(entry.steps, entry.w), [entry.steps, entry.w]);

  return (
    <Widget
      title="Lab: Tune the Guidance"
      wide
      caption={
        <>
          Ask for the target class with the exact denoiser and choose two things: how many DDIM steps (fewer is faster) and the guidance scale w. Scrub or press Play to watch guidance push 2,000 noisy samples towards the target blobs step by step. Meet the brief; optionally, with as few steps as possible. Your settings are saved in this browser.
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setHint(false);
        }}
        options={GUIDE_BRIEFS.map((b, i) => ({ value: i, label: `${lab[b.id].solved ? '✓ ' : ''}${b.title.split(' · ')[0]}` }))}
      />
      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
      </div>
      <Segmented label="Denoising steps" value={entry.steps} onChange={(steps) => update({ steps })} options={LAB_STEPS.map((s) => ({ value: s as number, label: String(s) }))} />
      <div style={{ marginTop: 8 }}>
        <Slider label="guidance scale w" value={entry.w} min={0} max={4} step={0.25} onChange={(w) => update({ w })} format={(v) => v.toFixed(2)} color="c-p" />
      </div>
      <div className="btn-row" style={{ margin: '8px 0 12px' }}>
        <button type="button" className="btn" onClick={() => setHint((h) => !h)}>{hint ? 'Hide hint' : 'Hint'}</button>
        {entry.solved && <span className="status good">Solved{entry.fewest !== null ? ` · fewest so far: ${entry.fewest} steps` : ''}</span>}
      </div>
      {hint && <div className="live-note" style={{ marginBottom: 12 }}>{hintText}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginBottom: 12 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <Scatter label={`Samples at step ${frameIdx}/${maxFrame} (w = ${entry.w.toFixed(2)})`}>
            <Blobs highlight={0} />
            <Dots pts={shownPts} color="var(--c-a)" />
          </Scatter>
          <ReplayControls r={replay} label="step" />
        </div>
        <div style={{ display: 'grid', gap: 10 }}>
          <span className={`status ${pass ? 'good' : 'bad'}`} style={{ justifySelf: 'start' }}>
            {pass ? 'Meets the brief' : 'Not yet'}
          </span>
          <div className="readout">
            <span style={{ color: r.onClass >= brief.minOn ? 'var(--c-good)' : 'var(--c-bad)' }}>{r.onClass >= brief.minOn ? '✓' : '✗'}</span> on a blue blob {pct(r.onClass)} (need ≥ {pct(brief.minOn)})
            {brief.minBlob > 0 && (
              <>
                {'\n'}
                <span style={{ color: r.minBlob >= brief.minBlob ? 'var(--c-good)' : 'var(--c-bad)' }}>{r.minBlob >= brief.minBlob ? '✓' : '✗'}</span> smallest blob share {pct(r.minBlob)} (need ≥ {pct(brief.minBlob)})
              </>
            )}
          </div>
          <BlobShares within={r.withinClass} cls={0} />
          <div className="live-note">
            {pass
              ? `It works with ${entry.steps} steps at w = ${entry.w}. ${entry.fewest !== null && entry.steps > (entry.fewest ?? 99) ? '' : 'Optional: can you do it with fewer steps?'}`
              : r.onClass < brief.minOn && r.offShape > r.wrongClass
                ? `${pct(r.offShape)} of samples are stranded between blobs. More steps, or more guidance, will commit them.`
                : r.onClass < brief.minOn
                  ? `${pct(r.wrongClass)} of samples ended up orange: the label isn’t being followed strongly enough.`
                  : `Accurate, but the guidance has starved a blob (${pct(r.minBlob)}).`}
          </div>
        </div>
      </div>

      <LiveCode note="Hugging Face diffusers CFG call" code={liveCode} />

      <div style={{ marginTop: 12 }}>
        <AnswerReveal solved={entry.solved} stuckKey={`guidance:${brief.id}`} label="the 102-setting (steps × w) map">
          <GuidanceExhaustiveMap brief={brief} curSteps={entry.steps} curW={entry.w} />
        </AnswerReveal>
      </div>

      <div style={{ marginTop: 12 }}>
        <MachineLog
          title="GUIDANCE BENCH LOG"
          lines={log.lines}
          onClear={log.clear}
          empty="Change denoising steps or guidance scale w to log runs."
        />
      </div>
    </Widget>
  );
}

function GuidanceExhaustiveMap({ brief, curSteps, curW }: { brief: GuideBrief; curSteps: number; curW: number }) {
  const ex = useMemo(() => guideExhaustive(brief), [brief]);
  const byKey = useMemo(() => new Map(ex.rows.map((row) => [`${row.steps}:${row.w}`, row])), [ex]);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
      <div className="live-note">
        All {ex.total} settings ({LAB_STEPS.length} step counts × {LAB_WS.length} values of w) were tried: <strong>{ex.passing}</strong> meet this brief. The fewest steps that can is <strong>{ex.fewest}</strong>, with w = {ex.atFewest.join(', ')}.
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 10.5 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '3px 5px', borderBottom: '1px solid var(--border)' }}>s \ w</th>
              {LAB_WS.map((w) => (
                <th key={w} style={{ textAlign: 'center', padding: '3px 2px', borderBottom: '1px solid var(--border)' }}>
                  {Number.isInteger(w) ? w : w.toFixed(2).replace(/^0/, '')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {LAB_STEPS.map((s) => (
              <tr key={s}>
                <td style={{ padding: '3px 5px', fontWeight: 600 }}>{s}</td>
                {LAB_WS.map((w) => {
                  const cell = byKey.get(`${s}:${w}`)!;
                  const ok = labPass(brief, cell);
                  const isMin = ok && s === ex.fewest;
                  const isCur = s === curSteps && Math.abs(w - curW) < 1e-6;
                  return (
                    <td
                      key={w}
                      title={`steps=${s}, w=${w}: on-class ${pct(cell.onClass)}, min-blob ${pct(cell.minBlob)}`}
                      style={{
                        textAlign: 'center',
                        padding: '3px 2px',
                        borderRadius: 3,
                        color: ok ? 'var(--c-good)' : 'var(--text-muted)',
                        fontWeight: isMin ? 700 : 400,
                        background: isMin
                          ? 'color-mix(in srgb, var(--c-good) 24%, transparent)'
                          : ok
                            ? 'color-mix(in srgb, var(--c-good) 10%, transparent)'
                            : 'transparent',
                        outline: isCur ? '1.5px solid var(--accent)' : 'none',
                      }}
                    >
                      {isMin ? '★' : ok ? '✓' : '·'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
          <code>★</code> = fewest-step winner · <code>✓</code> = passes brief · <code>·</code> = fails (hover any cell for exact on-class &amp; min-blob shares).
        </div>
      </div>
    </div>
  );
}
