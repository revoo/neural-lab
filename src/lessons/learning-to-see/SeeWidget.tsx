// authored-by: Claude Opus 5.5
/**
 * Learning to see, live: train a small CNN on Fashion-MNIST in the browser and scrub one timeline
 * that moves every panel: first-layer filters, feature maps of a chosen (or drawn) image, the class
 * map of the network's inner layer, its prediction, and what it looks at. All real computation.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { usePrefersReducedMotion } from '../../components/motion';
import binUrl from './fashion.bin?url';
import { Cnn, cnnLayout } from './cnn';
import { CLASSES, decodeFashion, type ImageSet } from './fashion';
import { measureSee, occlusion, pca2, penultimate, SEE_RUN, seeMoments, trainSeeAsync, type SeeMoment } from './see';
import { SEE_RESULTS } from './seeResults';
import { DrawPad, PixelGrid } from './views';

let dataPromise: Promise<ImageSet> | null = null;
export function loadFashion() {
  dataPromise ??= fetch(binUrl)
    .then((r) => r.arrayBuffer())
    .then((b) => decodeFashion(new Uint8Array(b)));
  return dataPromise;
}
export function useFashion() {
  const [d, setD] = useState<ImageSet | null>(null);
  useEffect(() => {
    let ok = true;
    loadFashion().then((x) => ok && setD(x));
    return () => {
      ok = false;
    };
  }, []);
  return d;
}

/** The first test image of each class: the picker's thumbnails. */
export function firstOfEachClass(d: ImageSet): number[] {
  return CLASSES.map((_, k) => Array.from(d.testY).indexOf(k));
}

type Cp = { step: number; params: Float32Array; acc: number; nc: number; edge: number[] };
const BAR = SEE_RESULTS.edgeBar;
const RAW = SEE_RESULTS.rawNC;
const fmt = (n: number) => n.toLocaleString('en-GB');
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

function Scatter({ pts, ys, label, w = 230 }: { pts: [number, number][]; ys: ArrayLike<number>; label: string; w?: number }) {
  const xs = pts.map((p) => p[0]);
  const yv = pts.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...yv), Math.max(...yv)];
  const sx = (v: number) => 6 + ((w - 12) * (v - x0)) / (x1 - x0 || 1);
  const sy = (v: number) => 6 + ((w - 12) * (y1 - v)) / (y1 - y0 || 1);
  return (
    <svg viewBox={`0 0 ${w} ${w}`} style={{ width: '100%', maxWidth: w, display: 'block', background: 'var(--bg-elev)', border: '1px solid var(--border)', borderRadius: 4 }} role="img" aria-label={label}>
      {pts.map((p, i) => (
        <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={2.2} fill={`var(--cls-${ys[i]})`} opacity={0.8} />
      ))}
    </svg>
  );
}

export function SeeWidget() {
  const data = useFashion();
  const reduced = usePrefersReducedMotion();
  const [cps, setCps] = useState<Cp[]>([]);
  const [sel, setSel] = useState(0);
  const [follow, setFollow] = useState(true);
  const [running, setRunning] = useState(false);
  const [timing, setTiming] = useState('');
  const [pick, setPick] = useState(0);
  const [drawing, setDrawing] = useState<Float32Array | null>(null);
  const [look, setLook] = useState<'occlusion' | 'saliency'>('occlusion');
  const [flash, setFlash] = useState<SeeMoment | null>(null);
  const stop = useRef(false);
  const seen = useRef(new Set<string>());

  // the untrained network first
  useEffect(() => {
    if (!data || cps.length) return;
    const net = new Cnn(SEE_RUN.cfg);
    net.init(SEE_RUN.seed);
    const m = measureSee(net, data);
    setCps([{ step: 0, params: Float32Array.from(net.p), acc: m.acc, nc: m.nc, edge: m.edge }]);
  }, [data, cps.length]);
  useEffect(() => () => void (stop.current = true), []);

  const run = async () => {
    if (!data) return;
    stop.current = false;
    seen.current = new Set();
    setRunning(true);
    setFollow(true);
    const acc: Cp[] = [];
    const t0 = performance.now();
    await trainSeeAsync(SEE_RUN, data, async (step, params, net) => {
      const m = measureSee(net, data);
      acc.push({ step, params, acc: m.acc, nc: m.nc, edge: m.edge });
      setCps([...acc]);
      setTiming(`step ${fmt(step)} / ${fmt(SEE_RUN.steps)} · ${((performance.now() - t0) / 1000).toFixed(0)} s`);
      await new Promise((r) => requestAnimationFrame(() => r(null)));
      return !stop.current;
    });
    setRunning(false);
  };

  useEffect(() => {
    if (follow) setSel(Math.max(0, cps.length - 1));
  }, [cps.length, follow]);

  const moments = useMemo(() => seeMoments(cps, BAR, RAW), [cps]);
  useEffect(() => {
    for (const m of moments)
      if (running && !seen.current.has(m.id)) {
        seen.current.add(m.id);
        setFlash(m);
      }
  }, [moments, running]);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), reduced ? 6000 : 4500);
    return () => clearTimeout(t);
  }, [flash, reduced]);

  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      setSel(cps.length - 1);
      setPlaying(false);
      return;
    }
    const t = setInterval(() => setSel((s) => (s >= cps.length - 1 ? (setPlaying(false), s) : s + 1)), 450);
    return () => clearInterval(t);
  }, [playing, reduced, cps.length]);

  const cur = cps[Math.min(sel, cps.length - 1)];
  const first = cps[0];
  const net = useMemo(() => {
    if (!cur) return null;
    const n = new Cnn(SEE_RUN.cfg);
    n.p.set(cur.params);
    return n;
  }, [cur]);
  const net0 = useMemo(() => {
    if (!first) return null;
    const n = new Cnn(SEE_RUN.cfg);
    n.p.set(first.params);
    return n;
  }, [first]);
  const thumbs = useMemo(() => (data ? firstOfEachClass(data) : []), [data]);
  const img = drawing ?? (data ? data.test[thumbs[pick]] : null);
  const truth = drawing ? null : pick;

  const view = useMemo(() => {
    if (!net || !img) return null;
    const tr = net.newTrace();
    const probs = Array.from(net.forward(img, tr));
    const top = probs.map((p, k) => ({ k, p })).sort((a, b) => b.p - a.p);
    const target = truth ?? top[0].k;
    const occ = occlusion(net, img, target);
    const tr2 = net.newTrace();
    net.forward(img, tr2);
    const dl = new Float64Array(10);
    dl[target] = 1;
    const sal = new Float64Array(784);
    const keep = Float64Array.from(net.g);
    net.backward(tr2, dl, sal);
    net.g.set(keep);
    return { tr, top, occ, sal, target };
  }, [net, img, truth]);
  const pcaNow = useMemo(() => (net && data ? pca2(penultimate(net, data.test)).proj : null), [net, data]);
  const pca0 = useMemo(() => (net0 && data ? pca2(penultimate(net0, data.test)).proj : null), [net0, data]);

  const haveRun = cps.length > 1;
  const edges = cur ? cur.edge.filter((e) => e > BAR).length : 0;
  let note = 'Loading 3,500 small photos of clothes…';
  if (cur && view)
    note = !haveRun
      ? `Step 0: random weights. It calls this picture “${CLASSES[view.top[0].k]}” with ${pct(view.top[0].p)} confidence, which means nothing; on the 500 test photos it is right ${pct(cur.acc)} of the time (guessing gets 10%). None of its 8 filters looks like an edge detector. Press Train.`
      : `Step ${fmt(cur.step)}: right on ${pct(cur.acc)} of the 500 test photos. ${edges} of its 8 first-layer filters ${edges === 1 ? 'is an edge or bar detector' : 'are edge or bar detectors'} (matching an oriented template better than 99% of random filters). In its inner layer, sorting by the nearest class average gets ${pct(cur.nc)} right; the same trick on raw pixels gets ${pct(RAW)}.`;

  // both filter rows share the current step's scale, so a faint random start looks faint
  const fscale = cur ? Math.max(1e-6, ...Array.from(cur.params.subarray(0, 200), Math.abs)) : 1;
  const fs = (v: ArrayLike<number>, n: number, key: string, label: string) => <PixelGrid key={key} v={v} n={n} mode="signed" scale={fscale} size={54} label={label} />;

  return (
    <Widget
      title="Learning to see, live: one timeline, every layer"
      wide
      actions={<span className="backend-timing mono">{timing || '◼ CPU · this page'}</span>}
      caption={
        <>
          A real convolutional network (8 first-layer 5 × 5 filters, 16 second-layer filters, 32 inner units, 10 classes; {fmt(cnnLayout(SEE_RUN.cfg).total)} weights) learning 3,000 Fashion-MNIST
          photos. Every panel is recomputed from the network as it was at the selected step. Filters: <span className="c-a">positive</span> and{' '}
          <span className="c-b">negative</span> weights. Pick a photo or draw one, then drag the timeline or press Replay. The class map places each of the 500 test
          photos by the network’s two strongest inner directions (PCA), coloured by its true class.
        </>
      }
    >
      <div className="btn-row" style={{ flexWrap: 'wrap', gap: 8 }}>
        <button className={`btn${running ? '' : ' primary'}`} disabled={!data} onClick={() => (running ? (stop.current = true) : run())}>
          {running ? '❚❚ Stop' : haveRun ? '↻ Train again' : '▶ Train (about a minute)'}
        </button>
        {haveRun && (
          <button className="btn" disabled={running} onClick={() => (setFollow(false), setSel(0), setPlaying(true))}>
            ▶ Replay its life
          </button>
        )}
      </div>
      {cps.length > 0 && (
        <label className="slider" style={{ display: 'block', margin: '8px 0' }}>
          <span className="slider-top">
            <span className="slider-label">timeline</span>
            <span className="slider-value">step {cur ? fmt(cur.step) : 0}</span>
          </span>
          <input type="range" min={0} max={Math.max(0, cps.length - 1)} value={Math.min(sel, cps.length - 1)} onChange={(e) => (setFollow(false), setSel(+e.target.value))} aria-label="Training timeline" />
        </label>
      )}
      {flash && (
        <div className="birth-flash" role="status">
          Step {fmt(flash.step)}: {flash.label}.
        </div>
      )}
      {moments.length > 0 && (
        <div className="btn-row" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
          {moments.map((m, k) => (
            <button key={m.id} className="btn" style={{ fontSize: 12.5 }} onClick={() => (setFollow(false), setSel(cps.findIndex((c) => c.step === m.step)))}>
              {k + 1}. step {fmt(m.step)}: {m.label}
            </button>
          ))}
        </div>
      )}
      {cur && first && (
        <div className="gpt-panel">
          <div className="field-label" style={{ textTransform: 'none' }}>First-layer filters: step 0 (top) and step {fmt(cur.step)} (bottom), drawn on the same scale</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
            {first.params && Array.from({ length: 8 }, (_, f) => fs(first.params.slice(f * 25, f * 25 + 25), 5, `a${f}`, `filter ${f + 1} at step 0`))}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
            {Array.from({ length: 8 }, (_, f) => (
              <div key={f} style={{ textAlign: 'center' }}>
                {fs(cur.params.slice(f * 25, f * 25 + 25), 5, `b${f}`, `filter ${f + 1} at step ${cur.step}`)}
                <div className="plot-tick" style={{ color: cur.edge[f] > BAR ? 'var(--c-c)' : undefined }}>
                  {cur.edge[f].toFixed(2)}
                  {cur.edge[f] > BAR ? ' ✓' : ''}
                </div>
              </div>
            ))}
          </div>
          <div className="plot-tick">Numbers: best match with an oriented edge or bar template (a random filter beats {BAR.toFixed(2)} only 1% of the time).</div>
        </div>
      )}
      {data && view && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 12, marginTop: 12 }}>
          <div className="gpt-panel">
            <div className="field-label" style={{ textTransform: 'none' }}>Show it a photo (or draw one)</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, margin: '6px 0' }}>
              {thumbs.map((t, k) => (
                <button key={k} onClick={() => (setDrawing(null), setPick(k))} title={CLASSES[k]} style={{ padding: 0, border: 'none', background: 'none', cursor: 'pointer' }}>
                  <PixelGrid v={data.test[t]} n={28} mode="grey" size={34} label={CLASSES[k]} highlight={!drawing && pick === k} />
                </button>
              ))}
            </div>
            <Segmented
              options={[
                { value: 'photo', label: 'test photo' },
                { value: 'draw', label: 'draw' },
              ]}
              value={drawing ? 'draw' : 'photo'}
              onChange={(v) => setDrawing(v === 'draw' ? new Float32Array(784) : null)}
            />
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8, alignItems: 'flex-start' }}>
              {drawing ? <DrawPad img={drawing} onChange={setDrawing} size={140} /> : <PixelGrid v={img!} n={28} mode="grey" size={140} label={`Test photo: ${CLASSES[pick]}`} />}
              <div style={{ flex: '1 1 120px' }}>
                {view.top.slice(0, 4).map(({ k, p }) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5 }}>
                    <span style={{ width: 78 }}>{CLASSES[k]}</span>
                    <span style={{ height: 9, width: `${60 * p}%`, background: `var(--cls-${k})`, borderRadius: 3 }} />
                    <span className="mono">{pct(p)}</span>
                  </div>
                ))}
                {truth !== null && <div className="plot-tick">It is a {CLASSES[truth]}.</div>}
                {drawing && <div className="plot-tick">Drag to draw, shift-drag to erase. It has only ever seen photos of clothes.</div>}
              </div>
            </div>
          </div>
          <div className="gpt-panel">
            <div className="field-label" style={{ textTransform: 'none' }}>What it looks at (for “{CLASSES[view.target]}”)</div>
            <Segmented
              options={[
                { value: 'occlusion', label: 'cover and see' },
                { value: 'saliency', label: 'gradient' },
              ]}
              value={look}
              onChange={(v) => setLook(v)}
            />
            <div style={{ display: 'flex', gap: 10, marginTop: 8, flexWrap: 'wrap' }}>
              <PixelGrid v={img!} n={28} mode="grey" size={120} label="the image" />
              {look === 'occlusion' ? (
                <PixelGrid v={view.occ.map} n={view.occ.n} mode="heat" size={120} label="Occlusion map" />
              ) : (
                <PixelGrid v={view.sal} n={28} mode="signed" size={120} label="Saliency map" />
              )}
            </div>
            <div className="plot-tick" style={{ marginTop: 4 }}>
              {look === 'occlusion'
                ? `A grey 6 × 6 patch was slid over the picture; the strongest shading marks where covering it lowered the ${CLASSES[view.target]} probability most (largest drop ${pct(Math.max(...view.occ.map))}).`
                : 'How much the score would rise (first colour) or fall (second colour) if that one pixel got a little more ink.'}
            </div>
          </div>
        </div>
      )}
      {view && (
        <div className="gpt-panel" style={{ marginTop: 12 }}>
          <div className="field-label" style={{ textTransform: 'none' }}>Feature maps of this picture at step {fmt(cur!.step)}: layer 1 (8 maps, 24 × 24) and layer 2 (16 maps, 8 × 8)</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
            {Array.from({ length: 8 }, (_, f) => (
              <PixelGrid key={f} v={view.tr.a1.slice(f * 576, (f + 1) * 576)} n={24} mode="heat" size={56} label={`layer 1 map ${f + 1}`} />
            ))}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
            {Array.from({ length: 16 }, (_, f) => (
              <PixelGrid key={f} v={view.tr.a2.slice(f * 64, (f + 1) * 64)} n={8} mode="heat" size={36} label={`layer 2 map ${f + 1}`} />
            ))}
          </div>
        </div>
      )}
      {pcaNow && pca0 && data && (
        <div className="gpt-panel" style={{ marginTop: 12 }}>
          <div className="field-label" style={{ textTransform: 'none' }}>The class map: 500 test photos in the network’s inner space (step 0 and step {fmt(cur!.step)})</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(200px, 100%), 1fr))', gap: 10, marginTop: 6 }}>
            <Scatter pts={pca0} ys={data.testY} label="Class map at step 0" />
            <Scatter pts={pcaNow} ys={data.testY} label={`Class map at step ${cur!.step}`} />
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 10px', marginTop: 6, fontSize: 12 }}>
            {CLASSES.map((c, k) => (
              <span key={c}>
                <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 5, background: `var(--cls-${k})`, marginRight: 4 }} />
                {c}
              </span>
            ))}
          </div>
        </div>
      )}
      <p className="live-note">{note}</p>
    </Widget>
  );
}
