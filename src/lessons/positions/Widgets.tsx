import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Widget } from '../../components/Widget';
import { Slider, Toggle } from '../../components/controls';
import { Arrow, Label, Plot } from '../../components/plot';
import { LEVELS, SPEEDS, previousWordWeights, rotate, ropeScore, selfAttention, sinusoid, solveLevel, speedPairMap, speedStrip, type Level } from './positions';

const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

function VecBars({ v, max = 2.5, color }: { v: number[]; max?: number; color: string }) {
  return (
    <svg viewBox={`0 0 ${v.length * 16} 40`} width={v.length * 16} height={40} role="img" aria-label={`vector ${v.map(f3).join(', ')}`}>
      <line x1={0} x2={v.length * 16} y1={20} y2={20} className="plot-axis" />
      {v.map((x, i) => {
        const h = (Math.max(-max, Math.min(max, x)) / max) * 19;
        return <rect key={i} x={i * 16 + 2} width={12} y={h > 0 ? 20 - h : 20} height={Math.max(0.8, Math.abs(h))} fill={color} rx={1.5} />;
      })}
    </svg>
  );
}

// ------------------------------------------------------------------ 1. attention can't see order

export function OrderBlind() {
  const [pos, setPos] = useState(false);
  const A = selfAttention(['dog', 'bites', 'man'], pos);
  const B = selfAttention(['man', 'bites', 'dog'], pos);
  const same = (w: string) => {
    const a = A.find((x) => x.word === w)!.out;
    const b = B.find((x) => x.word === w)!.out;
    return Math.max(...a.map((v, i) => Math.abs(v - b[i]))) < 1e-12;
  };
  const allSame = ['dog', 'bites', 'man'].every(same);
  const Row = ({ s }: { s: typeof A }) => (
    <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
      {s.map((x) => (
        <div key={x.word} style={{ textAlign: 'center' }}>
          <div className="mono" style={{ fontWeight: 700, fontSize: 13 }}>
            {x.word}
          </div>
          <VecBars v={x.out} color={x.word === 'dog' ? 'var(--c-a)' : x.word === 'man' ? 'var(--c-b)' : 'var(--text-faint)'} />
        </div>
      ))}
    </div>
  );
  return (
    <Widget
      title="Who bit whom?"
      caption={
        <>
          One real attention layer (random but fixed weights, vectors of length 6) applied to two sentences with the same words in a different order. Each word's output
          vector is drawn as six bars. Compare "dog" in the first sentence with "dog" in the second. Then add position information.
        </>
      }
    >
      <Toggle label="add position vectors to the words first" checked={pos} onChange={setPos} />
      <div style={{ display: 'grid', gap: 10, marginTop: 10 }}>
        <div>
          <div className="field-label">"dog bites man"</div>
          <Row s={A} />
        </div>
        <div>
          <div className="field-label">"man bites dog"</div>
          <Row s={B} />
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        {allSame ? (
          <>
            <strong>Identical.</strong> Every word gets the same output in both sentences — identical to fifteen decimal places, the rest being rounding — just in a different order. Attention compares
            every word with every other word, and a comparison doesn't care where the words are. As far as this layer is concerned, the two sentences mean the same thing.
          </>
        ) : (
          <>
            <strong>Now they differ.</strong> With a position vector added to each word, "dog" at position 0 and "dog" at position 2 are different vectors, so they ask
            different questions and get different answers. The layer can tell the biter from the bitten.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. clocks: sinusoidal positions

const D = 16;
const MAXPOS = 64;

function Dial({ angle, r = 16 }: { angle: number; r?: number }) {
  return (
    <svg width={2 * r + 4} height={2 * r + 4} viewBox={`${-r - 2} ${-r - 2} ${2 * r + 4} ${2 * r + 4}`} aria-hidden>
      <circle r={r} fill="var(--bg-sunken)" stroke="var(--border-strong)" />
      <line x1={0} y1={0} x2={r * Math.sin(angle)} y2={-r * Math.cos(angle)} stroke="var(--accent)" strokeWidth={2.4} strokeLinecap="round" />
    </svg>
  );
}

export function PositionClocks() {
  const [p, setP] = useState(10);
  const [q, setQ] = useState(11);
  const pe = (n: number) => sinusoid(n, D);
  const sim = (a: number, b: number) => pe(a).reduce((s, v, i) => s + v * pe(b)[i], 0);
  const rows = useMemo(() => Array.from({ length: MAXPOS }, (_, n) => pe(n)), []);
  const curve = useMemo(() => Array.from({ length: MAXPOS }, (_, o) => sim(20, 20 + o - 0)), []);
  return (
    <Widget
      title="Positions as clocks"
      wide
      caption={
        <>
          The original transformer's position vectors, here with 16 numbers: 8 pairs, each pair the (sin, cos) of one clock hand. Every hand turns at its own speed per
          position, each about 3.2 times slower than the last. Slide the position and watch the hands. The stripes on the right show all 64 positions at once (rows),
          16 numbers each (columns).
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 8 }}>
        <div style={{ flex: '1 1 200px' }}>
          <Slider label="position" value={p} min={0} max={MAXPOS - 1} step={1} onChange={setP} format={(v) => String(v)} color="accent" />
        </div>
        <div style={{ flex: '1 1 200px' }}>
          <Slider label="compare with position" value={q} min={0} max={MAXPOS - 1} step={1} onChange={setQ} format={(v) => String(v)} />
        </div>
      </div>
      <div className="widget-row">
        <div className="widget-col">
          <div className="field-label">the 8 hands at position {p} (fastest first)</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {Array.from({ length: D / 2 }, (_, i) => (
              <Dial key={i} angle={p / 10000 ** ((2 * i) / D)} />
            ))}
          </div>
          <div className="readout" style={{ marginTop: 10 }}>
            <span className="k">agreement of positions {p} and {q} (dot product) </span> {sim(p, q).toFixed(2)}
            {'\n'}
            <span className="k">largest possible (a position with itself)     </span> {(D / 2).toFixed(2)}
          </div>
          <div className="field-label" style={{ marginTop: 10 }}>
            agreement of position 20 with positions 20 … 83
          </div>
          <svg viewBox="0 0 320 60" width={320} height={60} role="img" aria-label="Dot product of one position with later positions">
            {curve.map((v, o) => (
              <rect key={o} x={o * 5} width={4} y={60 - (Math.max(0, v) / 8) * 58} height={(Math.max(0, v) / 8) * 58} fill="var(--accent)" opacity={0.75} />
            ))}
          </svg>
          <div className="muted" style={{ fontSize: 12 }}>
            Nearby positions agree strongly; agreement falls (unevenly) with distance.
          </div>
        </div>
        <div className="widget-col" style={{ flex: '0 1 200px' }}>
          <svg viewBox={`0 0 ${D * 10} ${MAXPOS * 4}`} width={D * 10} height={MAXPOS * 4} role="img" aria-label="Position vectors for positions 0 to 63">
            {rows.map((row, n) =>
              row.map((v, j) => (
                <rect key={`${n}-${j}`} x={j * 10} y={n * 4} width={10} height={4} fill={v >= 0 ? 'var(--c-a)' : 'var(--c-b)'} opacity={Math.abs(v)} />
              )),
            )}
            <rect x={0} y={p * 4} width={D * 10} height={4} fill="none" stroke="var(--text)" strokeWidth={1.2} />
          </svg>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. RoPE

export function RopeDial() {
  const [m, setM] = useState(3);
  const [n, setN] = useState(1);
  const [theta, setTheta] = useState(30);
  const q = [1.6, 0.6];
  const k = [1.2, 1.2];
  const t = (theta * Math.PI) / 180;
  const rq = rotate(q, t * m);
  const rk = rotate(k, t * n);
  const s = ropeScore(q, k, m, n, t);
  const s0 = ropeScore(q, k, m + 5, n + 5, t);
  return (
    <Widget
      title="Rotary positions: rotate, then compare"
      wide
      caption={
        <>
          A query (purple) for a word at position <em>m</em> and a key (orange) for a word at position <em>n</em>. Before comparing them, each is turned by{' '}
          <em>θ</em> × its position. Move both positions by the same amount and watch the score.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 8 }}>
        <div style={{ flex: '1 1 160px' }}>
          <Slider label="query position m" value={m} min={0} max={11} step={1} onChange={setM} format={(v) => String(v)} color="accent" />
        </div>
        <div style={{ flex: '1 1 160px' }}>
          <Slider label="key position n" value={n} min={0} max={11} step={1} onChange={setN} format={(v) => String(v)} color="c-b" />
        </div>
        <div style={{ flex: '1 1 160px' }}>
          <Slider label="θ (degrees per position)" value={theta} min={0} max={90} step={5} onChange={setTheta} format={(v) => `${v}°`} />
        </div>
        <button className="btn small" onClick={() => (m < 11 && n < 11 ? (setM(m + 1), setN(n + 1)) : (setM(m - Math.min(m, n)), setN(n - Math.min(m, n))))}>
          Move both one place later
        </button>
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '0 1 300px' }}>
          <Plot x={[-2, 2]} y={[-2, 2]} width={280} height={280} ariaLabel="The rotated query and key">
            <Arrow from={[0, 0]} to={q as [number, number]} color="var(--accent)" width={1.5} dashed opacity={0.4} />
            <Arrow from={[0, 0]} to={k as [number, number]} color="var(--c-b)" width={1.5} dashed opacity={0.4} />
            <Arrow from={[0, 0]} to={rq as [number, number]} color="var(--accent)" width={3} />
            <Arrow from={[0, 0]} to={rk as [number, number]} color="var(--c-b)" width={3} />
            <Label at={rq as [number, number]} dx={6} color="var(--accent)">
              q at {m}
            </Label>
            <Label at={rk as [number, number]} dx={6} color="var(--c-b)">
              k at {n}
            </Label>
          </Plot>
        </div>
        <div className="widget-col">
          <div className="readout">
            <span className="k">offset n − m                 </span> {n - m}
            {'\n'}
            <span className="k">angle turned: query, key      </span> {theta * m}°, {theta * n}°
            {'\n'}
            <span className="k">angle between them changes by </span> {theta * (n - m)}°
            {'\n'}
            <span className="k">score (rotated q · rotated k) </span> <strong>{f3(s)}</strong>
            {'\n'}
            <span className="k">same pair, 5 places later    </span> {f3(s0)}
          </div>
          <div className="live-note" style={{ marginTop: 10 }}>
            Both vectors turn, but the <strong>angle between them</strong> only depends on how far apart the words are: θ × (n − m) = {theta * (n - m)}°. So the
            score depends only on the <strong>offset</strong>, never on where in the text the pair sits — {f3(s)} here, and exactly the same five places later.
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. lab: build a previous-word head

const STORE = 'nl-positions-lab-v1';
type Saved = Record<string, { solved?: boolean; best?: number; impossible?: boolean }>;
const load = (): Saved => {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? '{}');
  } catch {
    return {};
  }
};


/** After a solve: the previous-word share for every whole-degree speed (one hand) or speed pair (two hands). */
function SpeedMap({ L, sp }: { L: Level; sp: number[] }) {
  if (L.hands === 1) return <SpeedStrip L={L} speed={sp[0]} />;
  return <SpeedPairCanvas L={L} sp={sp} />;
}

function SpeedStrip({ L, speed }: { L: Level; speed: number }) {
  const shares = useMemo(() => speedStrip(L), [L]);
  const H = 90;
  const bw = 3;
  const passing = shares.filter((v) => v >= L.target).length;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${SPEEDS.length * bw} ${H + 22}`} width="100%" style={{ minWidth: 360, display: 'block' }} role="img" aria-label="Previous-word share for every speed">
          <line x1={0} x2={SPEEDS.length * bw} y1={H - L.target * H} y2={H - L.target * H} stroke="var(--c-good)" strokeDasharray="4 3" strokeWidth={1} />
          {shares.map((v, i) => (
            <rect key={i} x={i * bw + 0.3} width={bw - 0.6} y={H - v * H} height={Math.max(0.6, v * H)} fill={v >= L.target ? 'var(--c-good)' : 'var(--text-faint)'} />
          ))}
          <polygon points={`${(speed - 0.5) * bw},${H + 2} ${(speed - 0.5) * bw - 4},${H + 9} ${(speed - 0.5) * bw + 4},${H + 9}`} fill="var(--text)" />
          {[1, 45, 90, 135, 180].map((t) => (
            <text key={t} x={t === 1 ? 0 : t === 180 ? SPEEDS.length * bw : (t - 0.5) * bw} y={H + 20} fontSize={8} textAnchor={t === 1 ? 'start' : t === 180 ? 'end' : 'middle'} className="plot-tick">
              {t}°
            </text>
          ))}
        </svg>
      </div>
      <p className="muted" style={{ fontSize: 13, margin: '6px 0 0' }}>
        Every whole-degree speed, 1° to 180°: bar height is the previous word's share, green bars reach the {pct(L.target)} goal (dashed line) — {passing} of 180. The
        triangle is your speed.{' '}
        {L.length <= 8
          ? 'Nothing below 31° passes: a slow hand can’t tell neighbours apart. The gaps above it sit at speeds such as 60°, 72°, 90°, 120° and 144°, where a word a few places further back lands on exactly the previous word’s angle (the speed times the extra steps is a whole number of turns).'
          : 'No bar reaches the line. Slow hands (left) can’t tell neighbours apart; fast hands (right) come round again and again inside 64 words, so a far-off word ties with the previous one.'}
      </p>
    </div>
  );
}

function SpeedPairCanvas({ L, sp }: { L: Level; sp: number[] }) {
  const map = useMemo(() => speedPairMap(L), [L]);
  const ref = useRef<HTMLCanvasElement>(null);
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ?? '');
  useEffect(() => {
    const obs = new MutationObserver(() => setTheme(document.documentElement.dataset.theme ?? ''));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);
  const n = SPEEDS.length;
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext('2d');
    if (!c || !ctx) return;
    const css = getComputedStyle(c);
    const good = css.getPropertyValue('--c-good').trim() || 'green';
    const faint = css.getPropertyValue('--text-faint').trim() || 'gray';
    ctx.clearRect(0, 0, n, n);
    for (let a = 0; a < n; a++)
      for (let b = 0; b < n; b++) {
        const v = map[a * n + b];
        ctx.globalAlpha = v >= L.target ? 0.9 : 0.08 + 0.5 * v;
        ctx.fillStyle = v >= L.target ? good : faint;
        ctx.fillRect(b, n - 1 - a, 1, 1);
      }
    ctx.globalAlpha = 1;
  }, [map, L.target, n, theme]);
  const passing = useMemo(() => {
    let k = 0;
    for (let a = 0; a < n; a++) for (let b = a; b < n; b++) if (map[a * n + b] >= L.target) k++;
    return k;
  }, [map, n, L.target]);
  const [h1, h2] = sp;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ position: 'relative', width: 'min(100%, 320px)', aspectRatio: '1' }}>
        <canvas ref={ref} width={n} height={n} style={{ width: '100%', height: '100%', imageRendering: 'pixelated', border: '1px solid var(--border)', borderRadius: 4 }} aria-label="Previous-word share for every pair of speeds" />
        <div
          style={{
            position: 'absolute',
            left: `${((h2 - 0.5) / n) * 100}%`,
            top: `${((n - h1 + 0.5) / n) * 100}%`,
            width: 12,
            height: 12,
            marginLeft: -6,
            marginTop: -6,
            borderRadius: '50%',
            border: '2px solid var(--text)',
          }}
        />
      </div>
      <p className="muted" style={{ fontSize: 13, margin: '6px 0 0' }}>
        Every pair of whole-degree speeds: hand 2 runs left to right (1° to 180°), hand 1 bottom to top. Green pixels reach the goal ({passing.toLocaleString()} of{' '}
        {((n * (n + 1)) / 2).toLocaleString()} distinct pairs); the map is symmetric because swapping the hands changes nothing. The ring is your pair. A pair wins when its two hands
        come back round at different places, so no far-off word matches the previous word on both at once: the same reason real rotary embeddings use many speeds.
      </p>
    </div>
  );
}

export function PreviousWordLab() {
  const [idx, setIdx] = useState(0);
  const [speeds, setSpeeds] = useState<Record<string, number[]>>({ short: [20], long: [20], two: [3, 20] });
  const [saved, setSaved] = useState<Saved>(load);
  const L = LEVELS[idx];
  const sp = speeds[L.id];
  const w = previousWordWeights(sp, L.length);
  const share = w[1];
  const ok = share >= L.target;
  const sol = useMemo(() => solveLevel(L), [L]);
  const rec = saved[L.id] ?? {};
  const store = (s: Saved) => {
    setSaved(s);
    try {
      localStorage.setItem(STORE, JSON.stringify(s));
    } catch {
      /* ignore */
    }
  };
  const setSpeed = (i: number, v: number) => {
    const next = sp.map((x, j) => (j === i ? v : x));
    setSpeeds({ ...speeds, [L.id]: next });
    const s2 = previousWordWeights(next, L.length)[1];
    if (s2 >= L.target && s2 > (rec.best ?? 0)) store({ ...saved, [L.id]: { ...rec, solved: true, best: s2 } });
  };
  const barW = Math.max(4, Math.floor(560 / L.length));
  const [showMap, setShowMap] = useState(false);
  const mapUnlocked = !!(rec.solved || rec.impossible);
  // which offset is the biggest rival?
  const rival = w.reduce((b, v, o) => (o !== 1 && v > w[b] ? o : b), 0);

  let note: ReactNode;
  if (ok) note = <>✓ The previous word gets {pct(share)}. {share >= sol.best.weight - 0.005 ? 'That’s the best any whole-degree setting achieves.' : `Optional: the best possible here is ${pct(sol.best.weight)} (found by trying every whole-degree setting).`}</>;
  else if (L.id === 'long' && rec.impossible)
    note = (
      <>
        <strong>Right: one hand can't do it.</strong> We tried all 180 whole-degree speeds; the best gets the previous word only {pct(sol.best.weight)}. A fast hand
        tells neighbours apart but comes back round to the same angle every 360/θ positions, so far-away words look exactly like the previous one. A slow hand never
        comes round, but then neighbours look almost identical. In 64 words you can't have both. Go on to level 3.
      </>
    );
  else
    note = (
      <>
        The previous word gets {pct(share)}; its biggest rival is the word {rival === 0 ? 'itself' : `${rival} back`} ({pct(w[rival])}).{' '}
        {rival > 2 ? 'That rival is far away: the hand has come round to almost the same angle. ' : rival <= 2 ? 'Neighbours look too similar: the hand turns too little per position. ' : ''}
      </>
    );

  return (
    <Widget
      title="Lab: build a previous-word head"
      wide
      caption={
        <>
          Every word in this text has the same content, so only position can decide. The last word looks back over the whole text; each bar is how much attention it
          gives to the word that many places back (0 = itself). Each <strong>hand</strong> is a rotary pair turning by the chosen number of degrees per position, aimed
          one place back. Goal: at least <strong>{pct(L.target)}</strong> on the previous word (offset 1).
        </>
      }
    >
      <div className="clinic-tabs" role="tablist" aria-label="Levels">
        {LEVELS.map((z, i) => (
          <button key={z.id} role="tab" aria-selected={i === idx} className={i === idx ? 'on' : ''} onClick={() => setIdx(i)}>
            {i + 1}. {z.title}
            {saved[z.id]?.solved || saved[z.id]?.impossible ? ' ✓' : ''}
          </button>
        ))}
      </div>
      <p style={{ margin: '4px 0 10px' }}>{L.brief}</p>
      <div className="controls inline" style={{ marginBottom: 10 }}>
        {sp.map((v, i) => (
          <div key={i} style={{ flex: '1 1 220px', maxWidth: 340 }}>
            <Slider label={`hand ${i + 1}: degrees per position`} value={v} min={1} max={180} step={1} onChange={(x) => setSpeed(i, x)} format={(x) => `${x}°`} color={i ? 'c-b' : 'accent'} />
          </div>
        ))}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${L.length * barW} 110`} width={L.length * barW} height={110} role="img" aria-label="Attention over offsets">
          {w.map((v, o) => (
            <g key={o}>
              <rect x={o * barW + 1} width={barW - 2} y={96 - v * 90} height={Math.max(0.8, v * 90)} fill={o === 1 ? 'var(--c-c)' : 'var(--text-faint)'} rx={1} />
              {(L.length <= 16 || o % 8 === 0) && (
                <text x={o * barW + barW / 2} y={108} fontSize={9.5} textAnchor="middle" className="plot-tick">
                  {o}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>
      <div className={`live-note`} style={{ marginTop: 10 }}>
        {note}
      </div>
      {L.id === 'long' && !rec.impossible && !ok && (
        <button className="btn" style={{ marginTop: 8 }} onClick={() => store({ ...saved, [L.id]: { ...rec, impossible: true } })}>
          I think one hand can't do this
        </button>
      )}
      <div className="muted" style={{ fontSize: 12.5, marginTop: 8 }}>
        {sol.solutions.length} of the {sol.tried} whole-degree settings for this level reach the goal.
      </div>
      {mapUnlocked && (
        <button className="btn" style={{ marginTop: 8 }} onClick={() => setShowMap((m) => !m)}>
          {showMap ? 'Hide' : 'Show'} every {L.hands === 1 ? 'speed' : 'pair of speeds'}
        </button>
      )}
      {mapUnlocked && showMap && <SpeedMap L={L} sp={sp} />}
    </Widget>
  );
}
