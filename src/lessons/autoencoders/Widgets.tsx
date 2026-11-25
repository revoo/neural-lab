// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Handle, Plot } from '../../components/plot';
import { AnswerReveal } from '../../components/AnswerReveal';
import { LiveCode } from '../../components/LabJob';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { createRng } from '../../lib/rng';
import {
  REAL_TOL,
  SIDE,
  face,
  independentPixelSample,
  makeDataset,
  mix,
  nearestFace,
  pixelMeans,
  type VAE,
} from './vae';
import {
  BETAS,
  BRIEFS,
  HIDDENS,
  LAB_SEEDS,
  LATENTS,
  LAB_STEPS,
  designKey,
  facePrinterHint,
  labData,
  labTest,
  labTrainer,
  paramCount,
  printAndJudge,
  verdict,
  type BriefId,
  type Design,
  type PrintResult,
} from './lab';
import { LAB_RESULTS, labOptimum } from './labResults';
import { facePrinterRealCode } from './realCode';

const f3 = (v: number) => v.toFixed(3);
/** Percentages of 200 prints come in steps of 0.5%, so show one decimal only when it's needed. */
const pct = (v: number) => {
  const x = Math.round(v * 1000) / 10;
  return `${Number.isInteger(x) ? x : x.toFixed(1)}%`;
};

// ------------------------------------------------------------------ drawing a face

/** A 12×12 face drawn as pixels: brightness becomes the opacity of the text colour. */
export function FaceImg({ img, size = 72, frame, title }: { img: ArrayLike<number>; size?: number; frame?: 'good' | 'bad' | 'accent'; title?: string }) {
  const cells = [];
  for (let r = 0; r < SIDE; r++)
    for (let c = 0; c < SIDE; c++) {
      const v = img[r * SIDE + c];
      if (v < 0.03) continue;
      cells.push(<rect key={r * SIDE + c} x={c} y={r} width={1.04} height={1.04} fill="var(--text)" opacity={Math.min(1, v)} />);
    }
  const stroke = frame === 'good' ? 'var(--c-good)' : frame === 'bad' ? 'var(--c-bad)' : frame === 'accent' ? 'var(--accent)' : 'var(--border)';
  return (
    <svg viewBox={`-0.5 -0.5 ${SIDE + 1} ${SIDE + 1}`} width={size} height={size} role="img" aria-label={title ?? 'a 12 by 12 pixel face'} style={{ display: 'block' }}>
      <rect x={-0.4} y={-0.4} width={SIDE + 0.8} height={SIDE + 0.8} rx={0.8} fill="var(--bg-sunken)" stroke={stroke} strokeWidth={frame ? 0.45 : 0.15} />
      {cells}
    </svg>
  );
}

function Captioned({ label, sub, children, color }: { label: string; sub?: string; children: React.ReactNode; color?: string }) {
  return (
    <div style={{ display: 'grid', justifyItems: 'center', gap: 4, minWidth: 0 }}>
      {children}
      <div style={{ fontSize: 12.5, fontWeight: 600, color: color ?? 'var(--text)', textAlign: 'center' }}>{label}</div>
      {sub && <div style={{ fontSize: 11.5, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textAlign: 'center' }}>{sub}</div>}
    </div>
  );
}

const realWord = (d: number) => (d <= REAL_TOL ? 'a face' : 'not a face');

// ------------------------------------------------------------------ 1. Where the faces live

const DATA400 = () => makeDataset(400, 7);
let meansCache: Float64Array | null = null;
const dataMeans = () => (meansCache ??= pixelMeans(DATA400().imgs));

export function FaceSpace() {
  const [a, setA] = useState({ smile: 1, eyes: 1 });
  const [b, setB] = useState({ smile: -1, eyes: 0 });
  const [t, setT] = useState(0.5);
  const [seed, setSeed] = useState(0);
  const fa = useMemo(() => face(a.smile, a.eyes), [a]);
  const fb = useMemo(() => face(b.smile, b.eyes), [b]);
  const pixBlend = useMemo(() => mix(fa, fb, t), [fa, fb, t]);
  const knobBlend = useMemo(() => face((1 - t) * a.smile + t * b.smile, (1 - t) * a.eyes + t * b.eyes), [a, b, t]);
  const dPix = useMemo(() => nearestFace(pixBlend).dist, [pixBlend]);
  const indep = useMemo(() => (seed ? independentPixelSample(dataMeans(), createRng(seed)) : null), [seed]);
  const dIndep = useMemo(() => (indep ? nearestFace(indep).dist : 0), [indep]);

  return (
    <Widget
      title="Where the faces live"
      wide
      caption={
        <>
          Every face here is 144 pixels drawn from just two knobs: <strong>smile</strong> (−1 frown to +1 grin) and <strong>eyes</strong> (0 shut to 1 wide). Set two faces, then slide the blend. The middle-left face averages the <em>pixels</em>; the middle-right one averages the <em>knobs</em>. Under each is its distance (pixel RMSE) to the nearest of 1,681 real faces on a fine grid of knob settings; we call anything within {REAL_TOL} “a face”. Then try <strong>Independent pixels</strong>: every pixel switched on with its own average brightness from 400 real faces.
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(230px, 100%), 1fr))', gap: 14, marginBottom: 12 }}>
        <div>
          <div className="field-label">Face A</div>
          <Slider label="smile" value={a.smile} min={-1} max={1} step={0.05} onChange={(v) => setA({ ...a, smile: v })} color="c-a" />
          <Slider label="eyes" value={a.eyes} min={0} max={1} step={0.05} onChange={(v) => setA({ ...a, eyes: v })} color="c-a" />
        </div>
        <div>
          <div className="field-label">Face B</div>
          <Slider label="smile" value={b.smile} min={-1} max={1} step={0.05} onChange={(v) => setB({ ...b, smile: v })} color="c-b" />
          <Slider label="eyes" value={b.eyes} min={0} max={1} step={0.05} onChange={(v) => setB({ ...b, eyes: v })} color="c-b" />
        </div>
        <div>
          <Slider label="blend: 0 = all A, 1 = all B" value={t} min={0} max={1} step={0.05} onChange={setT} color="c-p" />
          <div className="btn-row" style={{ marginTop: 6 }}>
            <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>Independent pixels</button>
          </div>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(96px, 1fr))', gap: 10, alignItems: 'start' }}>
        <Captioned label="Face A" color="var(--c-a)">
          <FaceImg img={fa} size={88} />
        </Captioned>
        <Captioned label="Average the pixels" sub={`${f3(dPix)} · ${realWord(dPix)}`} color={dPix <= REAL_TOL ? 'var(--c-good)' : 'var(--c-bad)'}>
          <FaceImg img={pixBlend} size={88} frame={dPix <= REAL_TOL ? 'good' : 'bad'} />
        </Captioned>
        <Captioned label="Average the knobs" sub="0.000 · a face" color="var(--c-good)">
          <FaceImg img={knobBlend} size={88} frame="good" />
        </Captioned>
        <Captioned label="Face B" color="var(--c-b)">
          <FaceImg img={fb} size={88} />
        </Captioned>
        {indep && (
          <Captioned label="Independent pixels" sub={`${f3(dIndep)} · ${realWord(dIndep)}`} color="var(--c-bad)">
            <FaceImg img={indep} size={88} frame="bad" />
          </Captioned>
        )}
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {dPix > REAL_TOL ? (
          <>
            Averaging pixels gives a double exposure: both mouths at half strength, {f3(dPix)} from the nearest real face. Averaging the <strong>knobs</strong> gives a genuine face every time, because it never leaves the set of faces. Real faces are a thin two-knob sheet inside a 144-dimensional space of pixel patterns, and a straight line between two of them in pixel space cuts through empty space.
          </>
        ) : (
          <>
            These two faces are similar enough that their pixel blend is still close to a real face. Pull them further apart (opposite smiles, different eyes) and the straight line between them in pixel space leaves the faces behind.
          </>
        )}
        {indep && ` Independent pixels are hopeless (${f3(dIndep)} away): each pixel knows how often it is lit, but not which other pixels it should be lit with.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ shared training store

type Run = { net: VAE; done: number; steps: number; result: PrintResult | null; history: { step: number; recon: number; kl: number }[] };
const runs = new Map<string, Run>();
const listeners = new Set<() => void>();
let version = 0;
const notify = () => {
  version++;
  listeners.forEach((l) => l());
};
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};
const useRuns = () => useSyncExternalStore(subscribe, () => version);
const runKey = (d: Design, seed: number) => `${designKey(d)}:${seed}`;
const queue: { d: Design; seed: number }[] = [];
let busy = false;

/** Queue a training run; runs are trained one at a time in small chunks so the page stays responsive. */
function startRun(d: Design, seed = 1) {
  const key = runKey(d, seed);
  if (runs.has(key)) return;
  const t = labTrainer(d, seed);
  runs.set(key, { net: t.net, done: 0, steps: LAB_STEPS, result: null, history: [] });
  queue.push({ d, seed });
  notify();
  if (!busy) pump();
}

function pump() {
  const job = queue.shift();
  if (!job) {
    busy = false;
    return;
  }
  busy = true;
  const key = runKey(job.d, job.seed);
  const run = runs.get(key)!;
  const t = labTrainer(job.d, job.seed);
  run.net = t.net;
  const tick = () => {
    const t0 = performance.now();
    while (performance.now() - t0 < 30 && !t.advance(25));
    run.done = t.done;
    if (t.done >= LAB_STEPS) {
      run.result = printAndJudge(t.net);
      notify();
      setTimeout(pump, 0);
      return;
    }
    notify();
    setTimeout(tick, 0);
  };
  setTimeout(tick, 0);
}

function Progress({ run }: { run: Run | undefined }) {
  if (!run || run.result) return null;
  return (
    <div style={{ display: 'grid', gap: 4 }}>
      <div className="progress-bar">
        <div style={{ width: `${(100 * run.done) / run.steps}%` }} />
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
        training: step {run.done} of {run.steps}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 2. Latent explorer

const smileColor = (s: number) => `color-mix(in srgb, var(--c-b) ${Math.round(((s + 1) / 2) * 100)}%, var(--c-a))`;

function useRealnessGrid(net: VAE | null, view: [number, number, number, number], n = 28) {
  return useMemo(() => {
    if (!net) return null;
    const [x0, x1, y0, y1] = view;
    const cells: { x: number; y: number; real: boolean }[] = [];
    let real = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const x = x0 + ((j + 0.5) * (x1 - x0)) / n;
        const y = y0 + ((i + 0.5) * (y1 - y0)) / n;
        const ok = nearestFace(net.decode([x, y]).out).dist <= REAL_TOL;
        if (ok) real++;
        cells.push({ x, y, real: ok });
      }
    return { cells, share: real / (n * n), dx: (x1 - x0) / n, dy: (y1 - y0) / n };
  }, [net, view, n]);
}

const AE_DESIGN: Design = { latent: 2, hidden: 32, beta: 0 };
const VAE_DESIGN: Design = { latent: 2, hidden: 32, beta: 1 };

export function LatentExplorer() {
  useRuns();
  const [kind, setKind] = useState<'ae' | 'vae'>('ae');
  const [viewKind, setViewKind] = useState<'codes' | 'bell'>('codes');
  const [probe, setProbe] = useState<[number, number] | null>(null);
  const design = kind === 'ae' ? AE_DESIGN : VAE_DESIGN;
  const run = runs.get(runKey(design, 1));
  const net = run?.result ? run.net : null;
  const data = useMemo(() => labData(), []);
  const codes = useMemo(() => (net ? data.imgs.map((x) => net.code(x) as [number, number]) : null), [net, data]);
  const view = useMemo<[number, number, number, number]>(() => {
    if (viewKind === 'bell' || !codes) return [-3, 3, -3, 3];
    const xs = codes.map((c) => c[0]);
    const ys = codes.map((c) => c[1]);
    const [ax, bx, ay, by] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const half = Math.max(bx - ax, by - ay) / 2 + 0.3;
    const cx = (ax + bx) / 2;
    const cy = (ay + by) / 2;
    return [cx - half, cx + half, cy - half, cy + half];
  }, [codes, viewKind]);
  const grid = useRealnessGrid(net, view);
  const z = probe ?? [(view[0] + view[1]) / 2, (view[2] + view[3]) / 2];
  const decoded = useMemo(() => (net ? net.decode(z).out : null), [net, z[0], z[1]]); // eslint-disable-line react-hooks/exhaustive-deps
  const nf = useMemo(() => (decoded ? nearestFace(decoded) : null), [decoded]);
  const width = 400;

  useEffect(() => setProbe(null), [kind, viewKind]);

  return (
    <Widget
      title="Squeeze 144 pixels through 2 numbers"
      wide
      caption={
        <>
          An encoder turns each face into a <strong>code</strong> of two numbers; a decoder turns codes back into faces. They were trained together, on 400 faces, only to make the face that comes out match the face that went in. Each dot is one training face’s code, coloured from <span style={{ color: 'var(--c-a)' }}>frown</span> to <span style={{ color: 'var(--c-b)' }}>grin</span>. The shaded backdrop shows, for every spot in this window, whether the decoder turns that code into a real face (<span style={{ color: 'var(--c-good)' }}>green</span>) or not (<span style={{ color: 'var(--c-bad)' }}>red</span>). Drag the probe to decode any code you like. Train the plain autoencoder first, then the VAE, and compare.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10, gap: 14 }}>
        <Segmented
          value={kind}
          onChange={setKind}
          options={[
            { value: 'ae', label: 'Plain autoencoder' },
            { value: 'vae', label: 'VAE (β = 1)' },
          ]}
        />
        <Segmented
          value={viewKind}
          onChange={setViewKind}
          options={[
            { value: 'codes', label: 'Zoom to the codes' },
            { value: 'bell', label: 'Window −3 … 3' },
          ]}
        />
        {!run && (
          <button type="button" className="btn primary" onClick={() => startRun(design)}>
            Train the {kind === 'ae' ? 'autoencoder' : 'VAE'}
          </button>
        )}
      </div>
      <Progress run={run} />
      {net && codes && grid && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
          <Plot
            width={width}
            height={width}
            x={[view[0], view[1]]}
            y={[view[2], view[3]]}
            ariaLabel="The code plane: training codes, and where codes decode to real faces"
            onPointerDown={(p) => setProbe([p.x, p.y])}
            style={{ maxWidth: '100%', height: 'auto', touchAction: 'none' }}
          >
            {({ x, y }) => (
              <g>
                {grid.cells.map((c, i) => (
                  <rect
                    key={i}
                    x={x(c.x - grid.dx / 2)}
                    y={y(c.y + grid.dy / 2)}
                    width={x(grid.dx) - x(0) + 0.5}
                    height={y(0) - y(grid.dy) + 0.5}
                    fill={c.real ? 'var(--c-good)' : 'var(--c-bad)'}
                    opacity={0.13}
                  />
                ))}
              </g>
            )}
            {({ x, y }) =>
              kind === 'vae' ? (
                <g fill="none" stroke="var(--accent)" strokeDasharray="5 4" strokeWidth={1.4}>
                  {[1, 2].map((r) => (
                    <ellipse key={r} cx={x(0)} cy={y(0)} rx={x(r) - x(0)} ry={y(0) - y(r)} opacity={0.8} />
                  ))}
                </g>
              ) : null
            }
            {({ x, y }) => (
              <g>
                {codes.map((c, i) =>
                  c[0] < view[0] || c[0] > view[1] || c[1] < view[2] || c[1] > view[3] ? null : (
                    <circle key={i} cx={x(c[0])} cy={y(c[1])} r={2.6} fill={smileColor(data.knobs[i].smile)} opacity={0.85} />
                  ),
                )}
              </g>
            )}
            <Handle x={z[0]} y={z[1]} onDrag={(px, py) => setProbe([px, py])} color="var(--accent)" label="probe code" />
          </Plot>
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ display: 'flex', gap: 14, alignItems: 'end', flexWrap: 'wrap' }}>
              <Captioned label="decoded probe" sub={nf ? `${f3(nf.dist)} · ${realWord(nf.dist)}` : ''} color={nf && nf.dist <= REAL_TOL ? 'var(--c-good)' : 'var(--c-bad)'}>
                {decoded && <FaceImg img={decoded} size={110} frame={nf && nf.dist <= REAL_TOL ? 'good' : 'bad'} />}
              </Captioned>
              {nf && (
                <Captioned label="nearest real face" sub={`smile ${nf.knobs.smile.toFixed(2)}, eyes ${nf.knobs.eyes.toFixed(2)}`}>
                  <FaceImg img={face(nf.knobs.smile, nf.knobs.eyes)} size={70} />
                </Captioned>
              )}
            </div>
            <div className="readout">
              <span className="k">probe code </span>({z[0].toFixed(2)}, {z[1].toFixed(2)}){'\n'}
              <span className="k">codes span </span>
              {Math.min(...codes.map((c) => c[0])).toFixed(1)} … {Math.max(...codes.map((c) => c[0])).toFixed(1)} by {Math.min(...codes.map((c) => c[1])).toFixed(1)} … {Math.max(...codes.map((c) => c[1])).toFixed(1)}
              {'\n'}
              <span className="k">window that decodes to faces </span>
              {pct(grid.share)}
            </div>
            <div className="live-note">
              {kind === 'ae' ? (
                viewKind === 'codes' ? (
                  <>
                    The autoencoder has discovered the two knobs on its own: run your eye along the dots and the smile changes smoothly. But it put the codes wherever it liked, in a lopsided patch, and only {pct(grid.share)} of the window around them decodes to faces. Drag the probe into a red corner. Now switch to the window −3 … 3: if you fed this decoder “ordinary” random numbers, what would you get?
                  </>
                ) : (
                  <>
                    Only {pct(grid.share)} of the square from −3 to 3 decodes to faces. Nothing in the training asked the codes to sit anywhere in particular, so to generate a new face you would first have to go and find out where this particular network happened to put them.
                  </>
                )
              ) : (
                <>
                  The VAE’s codes fill a round patch centred on 0 and reaching out to about 2 in every direction (dashed circles at 1 and 2), because training pulled them there. {pct(grid.share)} of this window decodes to faces, gaps between the dots included. Smile still changes smoothly in one direction and eyes in another. Random numbers from the standard bell are now a recipe for faces.
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. The β dial

const DIAL_BETAS = [0.1, 0.3, 1, 3, 10] as const;

export function BetaDial() {
  useRuns();
  const [beta, setBeta] = useState<number>(1);
  const design: Design = { latent: 2, hidden: 32, beta };
  const run = runs.get(runKey(design, 1));
  const res = run?.result ?? null;
  const net = res ? run!.net : null;
  const test = labTest();
  const mosaic = useMemo(() => {
    if (!net) return null;
    const out: Float64Array[] = [];
    for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) out.push(net.decode([-2.4 + (4.8 * j) / 6, 2.4 - (4.8 * i) / 6]).out);
    return out;
  }, [net]);
  const kl = useMemo(() => {
    if (!net) return 0;
    let s = 0;
    for (const x of test.imgs) {
      const { mu, logvar } = net.encode(x);
      for (let k = 0; k < net.L; k++) s += (0.5 * (mu[k] ** 2 + Math.exp(logvar[k]) - 1 - logvar[k])) / test.imgs.length;
    }
    return s;
  }, [net, test]);

  return (
    <Widget
      title="How hard to pull towards the bell"
      wide
      caption={
        <>
          β sets how much the VAE is charged for codes that stray from the standard bell, relative to how much it is charged for blurry reconstructions. Pick a β and train (same 400 faces, same seed, 3,000 steps). The map decodes a 7 × 7 grid of codes from −2.4 to 2.4; the printer row feeds the decoder random numbers from the standard bell, with a <span style={{ color: 'var(--c-bad)' }}>red frame</span> on anything that isn’t a face. Try 0.1, then 1, then 10.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <Segmented value={beta} onChange={setBeta} options={DIAL_BETAS.map((b) => ({ value: b as number, label: `β = ${b}` }))} />
        {!run && (
          <button type="button" className="btn primary" onClick={() => startRun(design)}>
            Train at β = {beta}
          </button>
        )}
      </div>
      <Progress run={run} />
      {res && mosaic && net && (
        <div style={{ display: 'grid', gap: 14 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
            <div>
              <div className="field-label">The map: decoded codes</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 3, maxWidth: 330 }}>
                {mosaic.map((m, i) => (
                  <FaceImg key={i} img={m} size={44} />
                ))}
              </div>
            </div>
            <div style={{ display: 'grid', gap: 10 }}>
              <div>
                <div className="field-label">Held-out faces: original over reconstruction</div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {test.imgs.slice(0, 5).map((x, i) => (
                    <div key={i} style={{ display: 'grid', gap: 3 }}>
                      <FaceImg img={x} size={46} />
                      <FaceImg img={net.reconstruct(x)} size={46} frame="accent" />
                    </div>
                  ))}
                </div>
              </div>
              <div className="readout">
                <span className="k">reconstruction error (pixel RMSE) </span>
                {f3(res.recon)}
                {'\n'}
                <span className="k">pull term: KL per face </span>
                {kl.toFixed(2)}
                {'\n'}
                <span className="k">printed faces that are real </span>
                {pct(res.realShare)}
                {'\n'}
                <span className="k">kinds of face reached (of 16) </span>
                {res.cells}
              </div>
            </div>
          </div>
          <div>
            <div className="field-label">The printer: 16 random codes from the standard bell</div>
            <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
              {res.prints.slice(0, 16).map((p, i) => (
                <FaceImg key={i} img={p.img} size={46} frame={p.real ? undefined : 'bad'} />
              ))}
            </div>
          </div>
          <div className="live-note">
            {beta <= 0.1 ? (
              <>
                With a gentle pull the VAE behaves almost like a plain autoencoder: sharp reconstructions ({f3(res.recon)}), but its codes don’t fill the bell evenly, so {pct(1 - res.realShare)} of printed codes land somewhere the decoder never learned to draw. Look for the junk in a corner of the map.
              </>
            ) : beta >= 10 ? (
              <>
                The pull has won completely. Every face is encoded as the standard bell itself (KL {kl.toFixed(2)}): the code carries <strong>no information</strong>, so the decoder can only learn to draw the one face that is least wrong on average. Every printed face is “real”, and they are all the same face. This failure is called <strong>posterior collapse</strong>.
              </>
            ) : beta >= 3 ? (
              <>
                Now the pull costs detail. Look along the map: one direction still changes the face, the other barely changes anything. The VAE has kept one knob and given up the other, so reconstructions blur ({f3(res.recon)}) and the printer reaches only {res.cells} of 16 kinds of face. The KL has fallen to {kl.toFixed(2)}: the code carries less information about each face.
              </>
            ) : (
              <>
                A balance: the map is smooth and full, reconstructions are close ({f3(res.recon)}), and {pct(res.realShare)} of printed faces are real, reaching {res.cells} of 16 kinds. The price of the pull is a little blur compared with a plain autoencoder.
              </>
            )}
          </div>
        </div>
      )}
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: the face printer

const LAB_KEY = 'nl-face-printer-lab-v1';
type LabEntry = { design: Design; solved: boolean; best: number | null };
type LabState = Record<BriefId, LabEntry>;
const START: Design = { latent: 1, hidden: 8, beta: 0 };

function loadLab(): LabState {
  const d: LabState = { printer: { design: START, solved: false, best: null }, compressor: { design: START, solved: false, best: null } };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return d;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of BRIEFS) {
      const v = parsed[b.id];
      if (!v?.design) continue;
      const { latent, hidden, beta } = v.design;
      if (!(LATENTS as readonly number[]).includes(latent) || !(HIDDENS as readonly number[]).includes(hidden) || !(BETAS as readonly number[]).includes(beta)) continue;
      d[b.id] = { design: { latent, hidden, beta }, solved: !!v.solved, best: typeof v.best === 'number' ? v.best : null };
    }
  } catch {
    // ignore storage errors
  }
  return d;
}

const betaLabel = (b: number) => (b === 0 ? 'none (plain AE)' : `β = ${b}`);
const mark = (ok: boolean) => <span style={{ color: ok ? 'var(--c-good)' : 'var(--c-bad)' }}>{ok ? '✓' : '✗'}</span>;

export function FacePrinterLab() {
  useRuns();
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [hint, setHint] = useState(false);
  const [guesses, setGuesses] = useState<Record<string, 'pass' | 'fail' | null>>({});
  const log = useMachineLog(20);
  const loggedRuns = useRef<Set<string>>(new Set());

  const brief = BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const d = entry.design;
  const dKey = designKey(d);
  const guessKey = `${brief.id}:${dKey}`;
  const guess = guesses[guessKey] ?? null;

  const seedRuns = LAB_SEEDS.map((seed) => runs.get(runKey(d, seed)));
  const started = seedRuns.some(Boolean);
  const results = seedRuns.map((r) => r?.result ?? null);
  const finished = results.every(Boolean);
  const verdicts = results.map((r) => (r ? verdict(brief, r) : null));
  const pass = finished && verdicts.every((v) => v!.pass);
  const recorded = LAB_RESULTS[dKey];
  const drift = finished && results.some((r, i) => Math.abs(r!.recon - recorded.seeds[i].recon) > 0.002);
  const current = seedRuns.find((r) => r && !r.result);
  const savedRef = useRef(lab);
  savedRef.current = lab;

  const persist = (next: LabState) => {
    setLab(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };

  useEffect(() => {
    if (!finished) return;
    const tag = `${brief.id}:${dKey}`;
    if (!loggedRuns.current.has(tag)) {
      loggedRuns.current.add(tag);
      const r0 = results[0]!;
      const nPass = verdicts.filter((v) => v!.pass).length;
      const metric = brief.minReal > 0 ? `recon ${f3(r0.recon)}, real ${pct(r0.realShare)}, ${r0.cells}/16 kinds` : `recon ${f3(r0.recon)}`;
      log.push(
        `${brief.id === 'printer' ? 'B1' : 'B2'}: L=${d.latent} H=${d.hidden} β=${d.beta} (${paramCount(d).toLocaleString('en-GB')}p) → ${metric} · ${nPass}/3 seeds (${pass ? 'PASS' : 'FAIL'})`,
        pass ? 'good' : 'bad',
      );
    }
    if (!pass) return;
    const cur = savedRef.current[brief.id];
    const params = paramCount(d);
    if (cur.solved && cur.best !== null && cur.best <= params) return;
    persist({ ...savedRef.current, [brief.id]: { ...cur, solved: true, best: cur.best === null ? params : Math.min(cur.best, params) } });
  }, [finished, pass, brief.id, dKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const setDesign = (patch: Partial<Design>) => persist({ ...lab, [brief.id]: { ...entry, design: { ...d, ...patch } } });
  const build = () => {
    log.push(`Training 3 seeds: L=${d.latent}, H=${d.hidden}, β=${d.beta} (${paramCount(d).toLocaleString('en-GB')} params)…`, 'info');
    LAB_SEEDS.forEach((seed) => startRun(d, seed));
  };
  const opt = labOptimum(brief.id);
  const usesPrinter = brief.minReal > 0;
  const first = results[0];
  const hintText = useMemo(
    () => facePrinterHint(brief.id, d, finished ? (results as PrintResult[]) : null),
    [brief.id, d, finished, results],
  );
  const liveCode = useMemo(() => facePrinterRealCode(d), [d]);

  return (
    <Widget
      title="Lab: The Face Printer"
      wide
      caption={
        <>
          Design a face generator: how many numbers in the code, how wide the hidden layers, and whether to train a plain autoencoder or a VAE (and how hard to pull towards the bell). <strong>Build</strong> trains your design for real in this browser, three times with seeds 1, 2 and 3 (3,000 steps each, on 400 faces), and a design only meets the spec if all three runs do: one lucky run proves nothing. Held-out faces are 60 the network never saw; the printer feeds 200 standard-bell codes to the decoder. Your designs are saved in this browser.
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setHint(false);
        }}
        options={BRIEFS.map((b, i) => ({ value: i, label: `${lab[b.id].solved ? '✓ ' : ''}${b.title.split(' · ')[0]}` }))}
      />
      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
        <div style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 6 }}>
          Spec: held-out faces rebuilt within {brief.maxRecon} pixel RMSE
          {usesPrinter && <> · at least {pct(brief.minReal)} of printed faces real · prints reach at least {brief.minCells} of the 16 kinds of face</>}. Optional: use as few parameters as you can.
        </div>
      </div>
      <div style={{ display: 'grid', gap: 10, marginBottom: 12 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
          <Segmented label="Numbers in the code" value={d.latent} onChange={(latent) => setDesign({ latent })} options={LATENTS.map((l) => ({ value: l as number, label: String(l) }))} />
          <Segmented label="Hidden width" value={d.hidden} onChange={(hidden) => setDesign({ hidden })} options={HIDDENS.map((h) => ({ value: h as number, label: String(h) }))} />
        </div>
        <Segmented label="Pull towards the bell" value={d.beta} onChange={(beta) => setDesign({ beta })} options={BETAS.map((b) => ({ value: b as number, label: betaLabel(b) }))} />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
          <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>Predict first (optional): will all 3 seeds pass?</span>
          <div style={{ display: 'inline-flex', gap: 6 }}>
            <button
              type="button"
              className={`btn small${guess === 'pass' ? ' primary' : ''}`}
              disabled={started}
              onClick={() => setGuesses((g) => ({ ...g, [guessKey]: g[guessKey] === 'pass' ? null : 'pass' }))}
            >
              Pass
            </button>
            <button
              type="button"
              className={`btn small${guess === 'fail' ? ' primary' : ''}`}
              disabled={started}
              onClick={() => setGuesses((g) => ({ ...g, [guessKey]: g[guessKey] === 'fail' ? null : 'fail' }))}
            >
              Fail
            </button>
          </div>
          {finished && guess !== null && (
            <span style={{ fontSize: 12.5, fontFamily: 'var(--mono)', color: (guess === 'pass') === pass ? 'var(--c-good)' : 'var(--c-b)' }}>
              {(guess === 'pass') === pass ? 'Prediction matched!' : `Predicted ${guess}, got ${pass ? 'pass' : 'fail'}.`}
            </span>
          )}
        </div>
        <div className="btn-row">
          <button type="button" className="btn primary" disabled={started} onClick={build}>
            {finished ? 'Built' : started ? 'Training…' : 'Build (3 seeds)'}
          </button>
          <button type="button" className="btn" onClick={() => setHint((h) => !h)}>{hint ? 'Hide hint' : 'Hint'}</button>
          <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{paramCount(d).toLocaleString('en-GB')} parameters</span>
          {entry.solved && <span className="status good">Solved{entry.best !== null ? ` · smallest so far ${entry.best.toLocaleString('en-GB')}` : ''}</span>}
        </div>
        {hint && <div className="live-note">{hintText}</div>}
      </div>
      {started && !finished && current && (
        <div style={{ display: 'grid', gap: 4, marginBottom: 12 }}>
          <div className="progress-bar">
            <div style={{ width: `${(100 * (results.filter(Boolean).length * LAB_STEPS + current.done)) / (LAB_SEEDS.length * LAB_STEPS)}%` }} />
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            training seed {results.filter(Boolean).length + 1} of {LAB_SEEDS.length}: step {current.done} of {LAB_STEPS}
          </div>
        </div>
      )}
      {finished && first && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginBottom: 12 }}>
          <div>
            {usesPrinter ? (
              <>
                <div className="field-label">Seed 1: the first 24 prints</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 4, maxWidth: 320 }}>
                  {first.prints.slice(0, 24).map((p, i) => (
                    <FaceImg key={i} img={p.img} size={48} frame={p.real ? undefined : 'bad'} />
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="field-label">Seed 1: held-out faces, original over rebuilt</div>
                <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                  {labTest().imgs.slice(0, 6).map((x, i) => (
                    <div key={i} style={{ display: 'grid', gap: 3 }}>
                      <FaceImg img={x} size={46} />
                      <FaceImg img={seedRuns[0]!.net.reconstruct(x)} size={46} frame="accent" />
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
          <div style={{ display: 'grid', gap: 10 }}>
            <span className={`status ${pass ? 'good' : 'bad'}`} style={{ justifySelf: 'start' }}>
              {pass ? 'Meets the spec with all three seeds' : `Fails with ${verdicts.filter((v) => !v!.pass).length} of 3 seeds`}
            </span>
            <div style={{ overflowX: 'auto' }}>
              <table className="num-table" style={{ width: '100%' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left' }}>Seed</th>
                    <th>Rebuild error</th>
                    {usesPrinter && <th>Real prints</th>}
                    {usesPrinter && <th>Kinds</th>}
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i}>
                      <td style={{ textAlign: 'left' }}>{LAB_SEEDS[i]}</td>
                      <td>
                        {mark(verdicts[i]!.recon)} {f3(r!.recon)}
                      </td>
                      {usesPrinter && (
                        <td>
                          {mark(verdicts[i]!.real)} {pct(r!.realShare)}
                        </td>
                      )}
                      {usesPrinter && (
                        <td>
                          {mark(verdicts[i]!.cells)} {r!.cells}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {drift && (
              <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
                Our recorded runs of this design gave rebuild errors {recorded.seeds.map((s) => f3(s.recon)).join(', ')}. Browsers and Node can round the last binary digit of an exponential differently, and 3,000 training steps can amplify that.
              </div>
            )}
            <div className="live-note">{diagnose(brief.id, d, results as PrintResult[], verdicts as ReturnType<typeof verdict>[], pass)}</div>
          </div>
        </div>
      )}

      <LiveCode note="PyTorch VAE / Autoencoder" code={liveCode} />

      <div style={{ marginTop: 12 }}>
        <AnswerReveal solved={entry.solved} stuckKey={`face-printer:${brief.id}`} label="the 54-design exhaustive map">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
            <div className="live-note">
              All {LATENTS.length * HIDDENS.length * BETAS.length} designs were trained with all three seeds. <strong>{opt.passing}</strong> meet this spec every time. The smallest is{' '}
              <strong>
                {opt.design.latent} code numbers, width {opt.design.hidden}, {betaLabel(opt.design.beta)}
              </strong>{' '}
              ({opt.params.toLocaleString('en-GB')} parameters){opt.ties === 1 ? ', the only design that small' : `, one of ${opt.ties} that size`}. Judged on seed 1 alone, {opt.seed1Passing} designs would have “passed”, and the smallest would have been {opt.seed1Smallest.latent} code numbers, width {opt.seed1Smallest.hidden}, {betaLabel(opt.seed1Smallest.beta)}: a lucky run.
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 11 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>L · H \ β</th>
                    {BETAS.map((b) => (
                      <th key={b} style={{ textAlign: 'center', padding: '4px 5px', borderBottom: '1px solid var(--border)' }}>
                        {b === 0 ? 'AE' : `β=${b}`}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {LATENTS.flatMap((L) =>
                    HIDDENS.map((H) => (
                      <tr key={`${L}-${H}`}>
                        <td style={{ padding: '3px 6px', fontWeight: 600 }}>L={L}, H={H}</td>
                        {BETAS.map((b) => {
                          const des: Design = { latent: L, hidden: H, beta: b };
                          const seeds = LAB_RESULTS[designKey(des)].seeds;
                          const nPass = seeds.filter((s) => verdict(brief, s).pass).length;
                          const seed1Only = verdict(brief, seeds[0]).pass && nPass < 3;
                          const isWinner = designKey(des) === designKey(opt.design);
                          return (
                            <td
                              key={b}
                              style={{
                                textAlign: 'center',
                                padding: '3px 5px',
                                borderRadius: 4,
                                color: nPass === 3 ? 'var(--c-good)' : seed1Only ? 'var(--c-b)' : 'var(--text-muted)',
                                fontWeight: nPass === 3 || isWinner ? 700 : 400,
                                background: nPass === 3 ? 'color-mix(in srgb, var(--c-good) 14%, transparent)' : seed1Only ? 'color-mix(in srgb, var(--c-b) 12%, transparent)' : 'transparent',
                              }}
                            >
                              {isWinner ? '★ 3/3' : nPass === 3 ? '3/3' : seed1Only ? '1s!' : `${nPass}/3`}
                            </td>
                          );
                        })}
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                <code>★ 3/3</code> = unique smallest winner · <code>3/3</code> = passes all 3 seeds · <code>1s!</code> = lucky seed-1 trap (fails on seeds 2 or 3).
              </div>
            </div>
          </div>
        </AnswerReveal>
      </div>

      <div style={{ marginTop: 12 }}>
        <MachineLog
          title="FACE PRINTER LOG"
          lines={log.lines}
          onClear={log.clear}
          empty="Pick latent width, hidden size, and β, then press Build (3 seeds)."
        />
      </div>
    </Widget>
  );
}

function diagnose(brief: BriefId, d: Design, rs: PrintResult[], vs: ReturnType<typeof verdict>[], pass: boolean) {
  const nBad = (k: 'recon' | 'real' | 'cells') => vs.filter((v) => !v[k]).length;
  if (pass)
    return `It works, three times out of three. ${d.latent > 2 ? 'You gave it more code numbers than the faces have knobs: that is allowed, and the spare numbers simply end up carrying little. ' : ''}Optional goal: meet the spec with as few parameters as possible.`;
  if (d.latent === 1 && nBad('recon')) return 'One number can’t hold two independent knobs. The decoder has to squeeze smile and eyes onto a single line, so most faces come back wrong.';
  if (brief === 'printer' && d.beta === 0 && nBad('real'))
    return `The plain autoencoder never agreed to put its codes where the printer looks: only ${rs.map((r) => pct(r.realShare)).join(', ')} of standard-bell codes decode to faces.`;
  if (d.beta >= 3 && nBad('recon')) return 'The pull towards the bell is so strong that the codes stop carrying detail. Faces come back blurred or as the average face, and printed faces bunch together.';
  if (d.beta === 0 && nBad('recon'))
    return 'Some runs of this plain autoencoder never got going: its codes started out large, the decoder’s tanh units saturated, and it settled for the average face. A little pull towards the bell would have kept the codes small.';
  if (vs.some((v) => v.pass)) return `Passes with ${vs.filter((v) => v.pass).length} of 3 seeds. That’s a design that sometimes works, which the spec doesn’t accept: look at which seed failed, and why.`;
  if (nBad('real')) return `Too gentle a pull: some standard-bell codes land where the decoder never learned to draw.`;
  if (nBad('cells')) return 'The prints are real faces but too samey: they don’t reach enough kinds of face.';
  return 'Faces don’t come back sharp enough. More width, a weaker pull, or more code numbers?';
}
