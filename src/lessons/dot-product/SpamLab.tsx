// authored-by: Claude Opus 5.5
import { useMemo, useRef, useState } from 'react';
import { Slider, Segmented } from '../../components/controls';
import { Arrow, Handle, Plot, usePlot } from '../../components/plot';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { BANK, BRIEFS, LONG, MAILS, bands, gap, scoreOf, sweep, sweep3, unit, type Mail, type Scorer } from './spamLab';

const LAB_KEY = 'neural-lab:dot-product-lab:w';
type Saved = { w: Record<string, number[]>; scorer: Scorer };

function load(): Saved {
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return { w: {}, scorer: 'dot' };
}

const START: Record<string, number[]> = { rank: [2, -1], gap: unit(-26.57), negative: [1, 1, 0], cosine: [2, -1] };

/** The passing band of directions, drawn as a shaded wedge from the origin. */
function Wedge({ from, to, r }: { from: number; to: number; r: number }) {
  const { x, y } = usePlot();
  const pts = [[0, 0]];
  for (let k = 0; k <= 40; k++) {
    const d = ((from + ((to - from) * k) / 40) * Math.PI) / 180;
    pts.push([r * Math.cos(d), r * Math.sin(d)]);
  }
  return <polygon points={pts.map(([a, b]) => `${x(a)},${y(b)}`).join(' ')} fill="var(--c-good)" opacity={0.22} stroke="var(--c-good)" strokeWidth={1} />;
}

/** Emails as dots, plus each one's shadow on the line of w (the shadow's position along the line is its score, up to w's length). */
function Mails({ mails, w, scorer }: { mails: Mail[]; w: number[]; scorer: Scorer }) {
  const { x, y } = usePlot();
  const n = Math.hypot(w[0], w[1]) || 1;
  const u = [w[0] / n, w[1] / n];
  return (
    <g>
      <line x1={x(-12 * u[0])} y1={y(-12 * u[1])} x2={x(12 * u[0])} y2={y(12 * u[1])} stroke="var(--text-faint)" strokeDasharray="2 4" />
      {scorer === 'cosine' && <circle cx={x(0)} cy={y(0)} r={x(1) - x(0)} fill="none" stroke="var(--text-faint)" strokeDasharray="3 3" />}
      {mails.map((m, i) => {
        const p = m.x.slice(0, 2);
        const len = Math.hypot(p[0], p[1]) || 1;
        const q = scorer === 'cosine' ? [p[0] / len, p[1] / len] : p; // cosine scores the direction only
        const t = q[0] * u[0] + q[1] * u[1];
        const foot = [t * u[0], t * u[1]];
        const col = m.spam ? 'var(--c-b)' : 'var(--c-a)';
        return (
          <g key={i}>
            <line x1={x(q[0])} y1={y(q[1])} x2={x(foot[0])} y2={y(foot[1])} stroke={col} strokeOpacity={0.35} strokeDasharray="2 2" />
            <circle cx={x(foot[0])} cy={y(foot[1])} r={3} fill={col} />
            {scorer === 'cosine' && <circle cx={x(q[0])} cy={y(q[1])} r={3.5} fill={col} opacity={0.8} />}
            <circle cx={x(p[0])} cy={y(p[1])} r={6} fill={col} stroke="var(--bg-elev)" strokeWidth={1.5} opacity={scorer === 'cosine' ? 0.45 : 1}>
              <title>{`${m.subject} (${m.spam ? 'spam' : 'real'}): ${m.x.slice(0, 2).join(' links, ')} exclamation marks`}</title>
            </circle>
          </g>
        );
      })}
    </g>
  );
}

/** The junk queue: every email's score as a bar, sorted from highest to lowest. */
function Queue({ mails, w, d, scorer }: { mails: Mail[]; w: number[]; d: number; scorer: Scorer }) {
  const rows = mails.map((m) => ({ m, s: scoreOf(w, m, d, scorer) })).sort((a, b) => b.s - a.s);
  const max = Math.max(1e-9, ...rows.map((r) => Math.abs(r.s)));
  const firstReal = rows.findIndex((r) => !r.m.spam);
  const lastSpam = rows.map((r) => r.m.spam).lastIndexOf(true);
  return (
    <div style={{ display: 'grid', gap: 3, fontSize: 12.5 }} aria-label="The junk queue, highest score first">
      {rows.map((r, i) => {
        const wrong = (r.m.spam && i > firstReal && firstReal >= 0) || (!r.m.spam && i < lastSpam);
        return (
          <div key={r.m.subject} style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 90px 52px', gap: 6, alignItems: 'center' }}>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: wrong ? 'var(--c-bad)' : 'var(--text)' }} title={r.m.subject}>
              {r.m.spam ? '✉︎ spam · ' : 'real · '}
              {r.m.subject}
            </span>
            <span style={{ position: 'relative', height: 10, background: 'var(--bg-sunken)', borderRadius: 3 }}>
              <span style={{ position: 'absolute', left: r.s >= 0 ? '50%' : `${50 - (45 * -r.s) / max}%`, width: `${(45 * Math.abs(r.s)) / max}%`, top: 0, bottom: 0, background: r.m.spam ? 'var(--c-b)' : 'var(--c-a)', borderRadius: 3 }} />
            </span>
            <span className="mono" style={{ textAlign: 'right' }}>{r.s.toFixed(scorer === 'cosine' ? 3 : 2)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function SpamLab() {
  const [saved, setSaved] = useState<Saved>(load);
  const [bi, setBi] = useState(0);
  const progress = useLabProgress('dot-product-lab');
  const log = useMachineLog();
  const brief = BRIEFS[bi];
  const d = brief.id === 'negative' ? 3 : 2;
  const mails = brief.id === 'negative' ? [...MAILS, BANK] : brief.id === 'cosine' ? [...MAILS, LONG] : MAILS;
  const scorer: Scorer = brief.id === 'cosine' ? saved.scorer : 'dot';
  const w = saved.w[brief.id] ?? START[brief.id];
  const g = gap(w, mails, d, scorer);
  const opt = useMemo(() => sweep(MAILS).best.gap, []);
  const passing = brief.id === 'gap' ? g >= 0.95 * opt : brief.id === 'cosine' ? scorer === 'cosine' && g > 0 : g > 0;
  const wasPassing = useRef(passing);

  const persist = (next: Saved) => {
    setSaved(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  const setW = (nw: number[]) => {
    const next = brief.id === 'gap' ? (() => { const n = Math.hypot(nw[0], nw[1]) || 1; return [nw[0] / n, nw[1] / n]; })() : nw;
    const ng = gap(next, mails, d, scorer);
    const np = brief.id === 'gap' ? ng >= 0.95 * opt : brief.id === 'cosine' ? scorer === 'cosine' && ng > 0 : ng > 0;
    if (np !== wasPassing.current) {
      const deg = (Math.atan2(next[1], next[0]) * 180) / Math.PI;
      log.push(`w = (${next.map((v) => v.toFixed(2)).join(', ')})${d === 2 ? `, pointing at ${deg.toFixed(1)}°` : ''}: gap ${ng.toFixed(3)} → ${np ? 'meets the brief' : 'no longer meets the brief'}`, np ? 'good' : 'bad');
      wasPassing.current = np;
    }
    persist({ ...saved, w: { ...saved.w, [brief.id]: next } });
  };
  const setScorer = (s: Scorer) => {
    const ng = gap(w, mails, d, s);
    log.push(`scorer switched to ${s === 'dot' ? 'dot product' : 'cosine similarity'}: gap ${ng.toFixed(3)}`, s === 'cosine' && ng > 0 ? 'good' : 'info');
    wasPassing.current = s === 'cosine' && ng > 0;
    persist({ ...saved, scorer: s });
  };

  const sw = useMemo(() => sweep(mails, scorer), [brief.id, scorer]); // eslint-disable-line react-hooks/exhaustive-deps
  const s3memo = useMemo(() => (brief.id === 'negative' ? sweep3(mails) : null), [brief.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const deg = (Math.atan2(w[1], w[0]) * 180) / Math.PI;
  const longScore = scoreOf(w, LONG, 2, scorer);
  const parcel = scoreOf(w, MAILS[1], 2, scorer);

  const status = passing
    ? brief.id === 'gap'
      ? `Gap ${g.toFixed(3)}: ${((100 * g) / opt).toFixed(1)}% of the widest possible (${opt.toFixed(3)}).`
      : `Every spam email scores above every real one. Gap ${g.toFixed(3)}.`
    : brief.id === 'gap'
      ? `Gap ${g.toFixed(3)}${g > 0 ? `: ${((100 * g) / opt).toFixed(1)}% of the widest possible. Keep turning.` : ': the ranking isn’t even perfect yet.'}`
      : brief.id === 'cosine' && scorer === 'dot'
        ? `With the dot product, the long newsletter scores ${longScore.toFixed(2)}, against ${parcel.toFixed(2)} for the parcel spam. Doubling every count doubled its score.`
        : `Gap ${g.toFixed(3)}: some real email scores at least as high as some spam (red in the queue).`;

  const reveal =
    brief.id === 'negative'
      ? (() => {
          const s3 = s3memo!;
          return {
            label: 'the search over directions',
            content: (
              <>Every direction of w on a 1° grid over the sphere was tried (64,440 directions). <strong>{s3.negative.toLocaleString()}</strong> rank perfectly, and every one of them has a <strong>negative</strong> weight on name mentions; {s3.posOrZero} have a weight of zero or more. Without negative evidence, the bank email can’t be told apart from spam.</>
            ),
          };
        })()
      : {
          label: 'the band of directions that work',
          content: (
            <>
              Sweeping w’s direction over 0–360° in 0.01° steps (exhaustive at that resolution):{' '}
              {sw.pass.length === 0 ? (
                <>no direction ranks perfectly with the {scorer === 'dot' ? 'dot product' : 'cosine'} scorer.</>
              ) : (
                <>
                  directions from {bands(sw.pass).map(([a, b]) => `${a.toFixed(2)}° to ${b.toFixed(2)}°`).join(' and ')} rank perfectly (the green wedge). The widest gap is {sw.best.gap.toFixed(3)}, at {sw.best.deg.toFixed(2)}°.
                </>
              )}
              {brief.id === 'cosine' && <> With the dot product, no direction at all works: the long newsletter always outscores the parcel spam once both point the same way.</>}
            </>
          ),
        };

  return (
    <LabShell
      title="Lab: Tune the Spam Scorer"
      caption={
        <>
          Each email is a point (orange: spam; blue: real mail; hover for its subject). Drag the tip of the weight arrow <strong>w</strong>. Every email casts a shadow on the line through w, and the shadow’s position is its score: the junk queue on the right lists the scores, highest first. Your settings are saved in this browser.
        </>
      }
      briefs={BRIEFS}
      active={bi}
      onSelect={(i) => {
        wasPassing.current = false;
        setBi(i);
      }}
      progress={progress}
      passing={passing}
      status={status}
      hint={
        brief.id === 'negative' ? (
          <>Look at the bank email’s row in the queue. On links and exclamation marks it looks exactly like spam; the only measurement where it differs is the third one. What does that third measurement tell you about whether an email is spam?</>
        ) : brief.id === 'cosine' ? (
          <>Compare the newsletter with its long edition: same direction, twice the length. Which scorer ignores length?</>
        ) : (
          <>Watch the shadows on the dashed line. A perfect ranking means every orange shadow lies further along w than every blue one.</>
        )
      }
      reveal={reveal}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Turn w: every time the ranking starts or stops meeting the brief, it’s logged here.' }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div>
          {d === 2 ? (
            <Plot x={[-4, 9]} y={[-4, 9]} width={400} height={400} xLabel="links" yLabel="exclamation marks" ariaLabel="Emails as points with the weight arrow w">
              {progress.solved[brief.id] && bands(sw.pass).map(([a, b]) => <Wedge key={a} from={a} to={b} r={12} />)}
              <Mails mails={mails} w={w} scorer={scorer} />
              <Arrow from={[0, 0]} to={[w[0] * (brief.id === 'gap' ? 3 : 1), w[1] * (brief.id === 'gap' ? 3 : 1)]} color="var(--c-c)" width={3.5} />
              <Handle
                x={w[0] * (brief.id === 'gap' ? 3 : 1)}
                y={w[1] * (brief.id === 'gap' ? 3 : 1)}
                onDrag={(px, py) => setW(brief.id === 'gap' ? [px / 3, py / 3] : [px, py])}
                color="var(--c-c)"
                snap={brief.id === 'gap' ? undefined : 0.1}
                label="tip of the weight arrow w"
              />
            </Plot>
          ) : (
            <div className="controls">
              <Slider label="weight on links" value={w[0]} min={-2} max={2} step={0.05} onChange={(v) => setW([v, w[1], w[2]])} />
              <Slider label="weight on exclamation marks" value={w[1]} min={-2} max={2} step={0.05} onChange={(v) => setW([w[0], v, w[2]])} />
              <Slider label="weight on mentions of your name" value={w[2]} min={-2} max={2} step={0.05} onChange={(v) => setW([w[0], w[1], v])} />
            </div>
          )}
          {brief.id === 'cosine' && (
            <div style={{ marginTop: 8 }}>
              <Segmented label="scorer" options={[{ value: 'dot' as Scorer, label: 'dot product w · x' }, { value: 'cosine' as Scorer, label: 'cosine similarity' }]} value={scorer} onChange={setScorer} />
            </div>
          )}
          <div className="readout" style={{ marginTop: 8 }}>
            <span className="k">w </span>({w.map((v) => v.toFixed(2)).join(', ')}){d === 2 && <span className="k"> · direction {deg.toFixed(1)}° · length {Math.hypot(w[0], w[1]).toFixed(2)}</span>}
            {'\n'}
            <span className="k">gap </span>
            {g.toFixed(3)} <span className="k">(lowest spam − highest real)</span>
          </div>
        </div>
        <Queue mails={mails} w={w} d={d} scorer={scorer} />
      </div>
    </LabShell>
  );
}
