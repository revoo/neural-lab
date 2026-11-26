// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { drawSprite, IMG, patchify, Vit } from './vision';
import { caption, CLIP_TEST, encodeImage, encodeText, SHAPES, SIDES, trainClip, WORDS, zeroShot, type ClipModel } from './clip';
import { attnCost, RACE_SIZES, RACE_VIT, runOne, SEEDS, type Arch } from './visionLab';
import { VISION_RESULTS } from './visionResults';
import { createRng } from '../../lib/rng';

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

/** An 8×8 image as SVG squares; optional patch grid and highlighted patch. */
export function Pixels({ x, size = 160, grid, label }: { x: ArrayLike<number>; size?: number; grid?: number; label?: string }) {
  const c = size / IMG;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} style={{ display: 'block', background: 'var(--bg-sunken)', borderRadius: 4 }} role="img" aria-label={label ?? 'An 8 by 8 pixel image'}>
      {Array.from(x).map((v, i) => (v > 0 ? <rect key={i} x={(i % IMG) * c} y={Math.floor(i / IMG) * c} width={c} height={c} fill="var(--text)" opacity={0.15 + 0.85 * v} /> : null))}
      {grid &&
        Array.from({ length: IMG / grid - 1 }, (_, k) => (k + 1) * grid * c).map((p) => (
          <g key={p}>
            <line x1={p} x2={p} y1={0} y2={size} stroke="var(--accent)" strokeWidth={1.5} />
            <line y1={p} y2={p} x1={0} x2={size} stroke="var(--accent)" strokeWidth={1.5} />
          </g>
        ))}
    </svg>
  );
}

// ------------------------------------------------------------------ patches as tokens

export function PatchifyWidget() {
  const [P, setP] = useState<1 | 2 | 4>(2);
  const [shuffled, setShuffled] = useState(false);
  const [shuffleSeed, setShuffleSeed] = useState(1);
  const x = useMemo(() => drawSprite(0, 2, 3), []);
  const n = IMG / P;
  const N = n * n;
  const order = useMemo(() => {
    const o = Array.from({ length: N }, (_, i) => i);
    if (!shuffled) return o;
    const r = createRng(shuffleSeed);
    for (let i = N - 1; i > 0; i--) {
      const j = r.int(i + 1);
      [o[i], o[j]] = [o[j], o[i]];
    }
    return o;
  }, [N, shuffled, shuffleSeed]);
  const patches = patchify(x, P);
  // reassemble the (possibly shuffled) image from its patches
  const shown = new Float64Array(IMG * IMG);
  order.forEach((src, t) => {
    const pi = Math.floor(t / n);
    const pj = t % n;
    patches[src].forEach((v, k) => (shown[(pi * P + Math.floor(k / P)) * IMG + pj * P + (k % P)] = v));
  });
  const d = 6;
  const blind = useMemo(() => new Vit(P, d, false, 3), [P]);
  const sighted = useMemo(() => new Vit(P, d, true, 3), [P]);
  const fmt = (v: number[]) => v.map((a) => (a >= 0 ? ' ' : '') + a.toFixed(3)).join(' ');
  const b0 = blind.pooledFrom(x);
  const b1 = blind.pooledFrom(x, order);
  const s0 = sighted.pooledFrom(x);
  const s1 = sighted.pooledFrom(x, order);
  const diff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  return (
    <Widget
      title="Patches as tokens, and what a shuffle reveals"
      wide
      caption={
        <>
          An 8×8 image cut into P×P patches. Each patch becomes one token: its pixels flattened into a list and multiplied by a learned matrix. Below, two small
          vision transformers with random weights read the image and summarise it as 6 numbers; one adds a position embedding to each token, one doesn’t. Shuffle the
          patches and compare.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <Segmented label="Patch size" options={[{ value: 1, label: '1 × 1' }, { value: 2, label: '2 × 2' }, { value: 4, label: '4 × 4' }]} value={P} onChange={(v) => setP(v as 1 | 2 | 4)} />
        <button className="btn" onClick={() => (shuffled ? setShuffleSeed((s) => s + 1) : setShuffled(true))}>
          🔀 Shuffle patches
        </button>
        {shuffled && (
          <button className="btn" onClick={() => setShuffled(false)}>
            Restore
          </button>
        )}
      </div>
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <Pixels x={shown} grid={P} label="The image, cut into patches" />
        <div style={{ flex: '1 1 260px', minWidth: 0 }}>
          <div style={{ fontSize: 13 }}>
            <strong className="mono">{N}</strong> tokens of <strong className="mono">{P * P}</strong> pixels each. Attention compares every token with every other:{' '}
            <strong className="mono">{(N * N).toLocaleString()}</strong> pairs, about <strong className="mono">{attnCost(P, d).toLocaleString()}</strong> multiply-adds at width {d}.
          </div>
          <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', margin: '8px 0' }}>
            {order.slice(0, 16).map((src, t) => (
              <div key={t} title={`token ${t + 1}: patch ${src + 1}`} style={{ display: 'flex', border: '1px solid var(--border)' }}>
                {patches[src].map((v, k) => (
                  <span key={k} style={{ width: 6, height: 10, background: v > 0 ? 'var(--text)' : 'var(--bg-sunken)' }} />
                ))}
              </div>
            ))}
            {N > 16 && <span style={{ fontSize: 12, color: 'var(--text-faint)' }}>… {N - 16} more</span>}
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="num-table" style={{ fontSize: 12 }}>
              <tbody>
                <tr>
                  <td>no positions</td>
                  <td className="mono">{fmt(shuffled ? b1 : b0)}</td>
                </tr>
                <tr>
                  <td>with positions</td>
                  <td className="mono">{fmt(shuffled ? s1 : s0)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <p className="live-note">
        {shuffled
          ? `Shuffled. The model without positions gives the same summary as for the original image (largest change ${diff(b0, b1).toExponential(1)}, rounding error): to it, an image is a bag of patches. With position embeddings the summary moves by up to ${diff(s0, s1).toFixed(3)}.`
          : `${N} tokens. ${P === 1 ? 'One-pixel patches keep every detail but make the sequence long: attention cost grows with the square of the token count.' : P === 4 ? 'Four big patches: a short sequence, but the 3×3 plus is split across patches and each token blurs a lot of pixels together.' : 'Press shuffle to rearrange the patches.'}`}
      </p>
    </Widget>
  );
}

// ------------------------------------------------------------------ the race

export function InductiveRace() {
  const recorded = RACE_SIZES.map((n) => ({
    n,
    cnn: mean(VISION_RESULTS.find((r) => r.group === 'race' && r.arch.kind === 'cnn' && r.n === n)!.accs),
    vit: mean(VISION_RESULTS.find((r) => r.group === 'race' && r.arch.kind === 'vit' && r.n === n)!.accs),
  }));
  const [live, setLive] = useState<Record<number, { cnn: number; vit: number }>>({});
  const [busy, setBusy] = useState<number | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const rerun = async (n: number) => {
    setBusy(n);
    const acc: Record<'cnn' | 'vit', number[]> = { cnn: [], vit: [] };
    for (const seed of SEEDS)
      for (const [key, arch] of [
        ['cnn', { kind: 'cnn' } as Arch],
        ['vit', RACE_VIT],
      ] as const) {
        await new Promise((r) => setTimeout(r, 0));
        if (!alive.current) return;
        acc[key].push(runOne(arch, n, seed));
      }
    setLive((l) => ({ ...l, [n]: { cnn: mean(acc.cnn), vit: mean(acc.vit) } }));
    setBusy(null);
  };
  return (
    <Widget
      title="CNN versus ViT: who needs how much data?"
      wide
      caption={
        <>
          Plus or X, anywhere in the image. A small CNN (4 filters, 45 weights) and a small ViT (2×2 patches, width 8, learned position embeddings, 369 weights)
          train on 16 to 1,024 images. Every run sees the same 8,192 training examples in total, so bigger datasets aren’t also given more training. Bars average
          seeds 1–3 on 400 test images: <span className="c-a">CNN</span> and <span className="c-b">ViT</span>. Press a size to retrain it live.
        </>
      }
    >
      <div style={{ display: 'grid', gap: 10 }}>
        {recorded.map((r) => (
          <div key={r.n} style={{ display: 'grid', gridTemplateColumns: '92px 1fr', gap: 8, alignItems: 'center' }}>
            <button className="btn" style={{ fontSize: 12, padding: '4px 6px' }} disabled={busy !== null} onClick={() => rerun(r.n)}>
              {busy === r.n ? 'training…' : `${r.n} images`}
            </button>
            <div style={{ fontSize: 12 }}>
              {(['cnn', 'vit'] as const).map((k) => {
                const v = live[r.n]?.[k] ?? r[k];
                return (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                    <div style={{ flex: 1, background: 'var(--bg-sunken)', borderRadius: 3, height: 12, position: 'relative' }}>
                      <div style={{ width: `${100 * v}%`, background: k === 'cnn' ? 'var(--c-a)' : 'var(--c-b)', height: 12, borderRadius: 3 }} />
                      <div style={{ position: 'absolute', left: '50%', top: -1, height: 14, width: 1, background: 'var(--text-faint)' }} />
                    </div>
                    <span className="mono" style={{ width: 48, textAlign: 'right' }}>
                      {pct(v)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 4 }}>thin line: guessing (50%)</div>
      <p className="live-note">
        With 16 images the CNN averages {pct(recorded[0].cnn)} and the ViT {pct(recorded[0].vit)}, hardly better than guessing. At 1,024 the gap is down to{' '}
        {(100 * (recorded[3].cnn - recorded[3].vit)).toFixed(1)} points. The CNN starts out knowing that a pattern means the same thing wherever it appears; the ViT has to learn it from examples.
        {Object.keys(live).length > 0 ? ' Live re-runs replace the recorded bars; small differences from the record come from your browser’s floating-point maths.' : ''}
      </p>
    </Widget>
  );
}

// ------------------------------------------------------------------ mini-CLIP

const SETTING = { tau: 0.3, k: 8, batch: 16 };

export function ClipWorkbench() {
  const [seed, setSeed] = useState(1);
  const [trained, setTrained] = useState(true);
  const model = useMemo<ClipModel>(() => trainClip(SETTING, seed, trained ? undefined : 0), [seed, trained]);
  const [shape, setShape] = useState(2);
  const [side, setSide] = useState(1);
  const examples = useMemo(() => Array.from({ length: 8 }, (_, c) => CLIP_TEST.find((p) => p.cls === c)!), []);
  const caps = Array.from({ length: 8 }, (_, c) => encodeText(model, [c >> 1, 4 + (c & 1)]).u);
  const sims = examples.map((p) => {
    const u = encodeImage(model, p.img).u;
    return caps.map((t) => u.reduce((a, x, i) => a + x * t[i], 0));
  });
  const zs = useMemo(() => zeroShot(model), [model]);
  const prompt = encodeText(model, [shape, 4 + side]).u;
  const ranked = useMemo(
    () =>
      CLIP_TEST.slice(0, 120)
        .map((p) => ({ p, s: encodeImage(model, p.img).u.reduce((a, x, i) => a + x * prompt[i], 0) }))
        .sort((a, b) => b.s - a.s)
        .slice(0, 6),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model, shape, side],
  );
  const hits = ranked.filter((r) => r.p.shape === shape && r.p.side === side).length;
  return (
    <Widget
      title="Mini-CLIP: pictures and sentences in one space"
      wide
      caption={
        <>
          An image encoder (6 learned 3×3 filters, pooled over each half, projected to 8 numbers) and a text encoder (one learned 8-number vector per word, summed)
          trained together on 400 batches of 16 (image, caption) pairs, temperature 0.3. The grid compares 8 new images with the 8 captions: darker is more similar.
          Untrain it to see the starting point; build a prompt and see which images it finds.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <Segmented label="Encoders" options={[{ value: 'on', label: 'trained' }, { value: 'off', label: 'untrained' }]} value={trained ? 'on' : 'off'} onChange={(v) => setTrained(v === 'on')} />
        <Segmented label="Seed" options={[1, 2, 3]} value={seed} onChange={(v) => setSeed(v as number)} />
      </div>
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr>
                <th />
                {Array.from({ length: 8 }, (_, c) => (
                  <th key={c} style={{ padding: 2, fontWeight: 500, textTransform: 'none', color: 'var(--text-muted)', writingMode: 'vertical-rl', transform: 'rotate(180deg)', height: 96, textAlign: 'left' }}>
                    {caption(c >> 1, c & 1)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {examples.map((p, i) => (
                <tr key={i}>
                  <td style={{ padding: 2 }}>
                    <Pixels x={p.img} size={28} label={caption(p.shape, p.side)} />
                  </td>
                  {sims[i].map((s, j) => {
                    const best = sims[i].indexOf(Math.max(...sims[i])) === j;
                    return (
                      <td key={j} title={s.toFixed(3)} style={{ width: 30, height: 30, padding: 0, border: '1px solid var(--bg)', background: `color-mix(in srgb, var(--accent) ${Math.round(100 * Math.max(0, s))}%, var(--bg-sunken))`, outline: best ? '2px solid var(--text)' : undefined, outlineOffset: -2 }} />
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ flex: '1 1 240px', minWidth: 0 }}>
          <div style={{ fontSize: 13, marginBottom: 6 }}>Build a prompt:</div>
          <div className="btn-row" style={{ marginBottom: 6 }}>
            <Segmented options={SHAPES.map((s, i) => ({ value: i, label: s }))} value={shape} onChange={(v) => setShape(v as number)} />
            <Segmented options={SIDES.map((s, i) => ({ value: i, label: s }))} value={side} onChange={(v) => setSide(v as number)} />
          </div>
          <div className="mono" style={{ fontSize: 13, margin: '4px 0 8px' }}>“{caption(shape, side)}”</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {ranked.map((r, i) => (
              <div key={i} style={{ textAlign: 'center', fontSize: 11 }}>
                <Pixels x={r.p.img} size={44} label={caption(r.p.shape, r.p.side)} />
                <span className="mono">{r.s.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>The 6 most similar of 120 new images (cosine similarity).</div>
        </div>
      </div>
      <p className="live-note">
        {trained
          ? `Zero-shot accuracy on 400 new images, classifying each by its most similar caption: ${pct(zs)} (guessing: 12.5%). ${hits} of the 6 images your prompt finds really are a ${SHAPES[shape]} on the ${SIDES[side]}. Nobody trained a classifier: the classes are just sentences built from the words ${WORDS.join(', ')}.`
          : `Untrained, the encoders point in arbitrary directions: zero-shot accuracy ${pct(zs)} (guessing: 12.5%). Training pulls each image towards its own caption and pushes it away from the other captions in its batch.`}
      </p>
    </Widget>
  );
}
