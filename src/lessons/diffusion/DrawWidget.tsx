// authored-by: Claude Opus 5.5
/**
 * Learning to draw: the recorded diffusion run's 16 fixed-seed pictures at every checkpoint, one
 * picture's journey out of the noise, and an optional live re-run (CPU or WebGPU).
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { usePrefersReducedMotion } from '../../components/motion';
import { BackendChip, useComputeTier } from '../../components/BackendChip';
import { createMlpEngine } from '../../lib/compute/selectBackend';
import type { MlpEngine } from '../../lib/compute/types';
import binUrl from './fashion8.bin?url';
import { decodeDraw, DRAW_RUN, drawMlp, drawMoments, nearestTrain, sampleDraw, trainDraw, type DrawMoment, type DrawSet } from './draw';
import { DRAW_RESULTS } from './drawResults';

const BAR = DRAW_RESULTS.bar;
const fmt = (n: number) => n.toLocaleString('en-GB');

function useThemeKey() {
  const [k, setK] = useState(() => document.documentElement.dataset.theme ?? '');
  useEffect(() => {
    const obs = new MutationObserver(() => setK(document.documentElement.dataset.theme ?? ''));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);
  return k;
}

/** An 8 × 8 picture with pixels in [−1, 1]: −1 = background, +1 = full ink in the text colour. */
function Pic({ v, size = 46, ok, label }: { v: ArrayLike<number>; size?: number; ok?: boolean; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useThemeKey();
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext('2d');
    if (!c || !ctx) return;
    const css = getComputedStyle(c);
    ctx.globalAlpha = 1;
    ctx.fillStyle = css.getPropertyValue('--bg-elev').trim() || '#fff';
    ctx.fillRect(0, 0, 8, 8);
    ctx.fillStyle = css.getPropertyValue('--text').trim() || '#000';
    for (let i = 0; i < 64; i++) {
      ctx.globalAlpha = Math.max(0, Math.min(1, (v[i] + 1) / 2));
      ctx.fillRect(i % 8, Math.floor(i / 8), 1, 1);
    }
    ctx.globalAlpha = 1;
  }, [v, theme]);
  return (
    <canvas
      ref={ref}
      width={8}
      height={8}
      role="img"
      aria-label={label}
      style={{ width: size, height: size, imageRendering: 'pixelated', border: `2px solid ${ok ? 'var(--c-c)' : 'var(--border)'}`, borderRadius: 3, display: 'block' }}
    />
  );
}

type Cp = { step: number; nn: readonly number[]; samples: readonly (readonly number[])[] };

let dataPromise: Promise<DrawSet> | null = null;
const loadDraw = () =>
  (dataPromise ??= fetch(binUrl)
    .then((r) => r.arrayBuffer())
    .then((b) => decodeDraw(new Uint8Array(b))));

export function DrawWidget() {
  const tier = useComputeTier();
  const reduced = usePrefersReducedMotion();
  const recorded = DRAW_RESULTS.checkpoints as unknown as Cp[];
  const [live, setLive] = useState<Cp[] | null>(null);
  const cps = live && live.length > 1 ? live : recorded;
  const [sel, setSel] = useState(recorded.length - 1);
  const [pick, setPick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [flash, setFlash] = useState<DrawMoment | null>(null);
  const [running, setRunning] = useState(false);
  const [timing, setTiming] = useState('');
  const [data, setData] = useState<DrawSet | null>(null);
  const stop = useRef(false);
  const eng = useRef<MlpEngine | null>(null);
  useEffect(() => {
    let ok = true;
    loadDraw().then((d) => ok && setData(d));
    return () => {
      ok = false;
      stop.current = true;
      eng.current?.destroy();
    };
  }, []);
  const moments = useMemo(() => drawMoments(cps, BAR), [cps]);
  const i = Math.min(sel, cps.length - 1);
  const cur = cps[i];
  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      setSel(cps.length - 1);
      setPlaying(false);
      return;
    }
    const t = setInterval(() => setSel((s) => (s >= cps.length - 1 ? (setPlaying(false), s) : s + 1)), 600);
    return () => clearInterval(t);
  }, [playing, reduced, cps.length]);
  useEffect(() => {
    if (!playing) return;
    const m = moments.find((mm) => mm.step === cur.step);
    if (m) setFlash(m);
  }, [i, playing, moments, cur.step]);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), reduced ? 6000 : 2500);
    return () => clearTimeout(t);
  }, [flash, reduced]);

  const nearest = useMemo(() => {
    if (!data) return null;
    const x = cur.samples[pick];
    let best = 0;
    let bd = Infinity;
    data.train.forEach((im, k) => {
      let s = 0;
      for (let j = 0; j < 64; j++) s += (x[j] - im[j]) ** 2;
      if (s < bd) [bd, best] = [s, k];
    });
    return { img: data.train[best], d: Math.sqrt(bd) };
  }, [data, cur, pick]);

  const run = async () => {
    if (!data) return;
    stop.current = false;
    setRunning(true);
    eng.current?.destroy();
    const e = await createMlpEngine(drawMlp(DRAW_RUN));
    eng.current = e;
    const acc: Cp[] = [];
    const t0 = performance.now();
    try {
      await trainDraw(e, data, DRAW_RUN, async (step) => {
        const xs = await sampleDraw(e, 16, 7);
        acc.push({ step, nn: xs.map((x) => +nearestTrain(x, data.train).toFixed(2)), samples: xs.map((x) => Array.from(x)) });
        setLive([...acc]);
        setSel(acc.length - 1);
        setTiming(`step ${fmt(step)} / ${fmt(DRAW_RUN.steps)} · ${((performance.now() - t0) / 1000).toFixed(0)} s`);
        await new Promise((r) => requestAnimationFrame(() => r(null)));
        return !stop.current;
      }, e.kind === 'cpu' ? 10 : 50);
    } catch {
      /* engine destroyed */
    }
    setRunning(false);
  };

  const inRange = cur.nn.filter((v) => v <= BAR).length;
  return (
    <Widget
      title="Learning to draw: from static to clothes"
      wide
      actions={<BackendChip tier={tier} timing={timing} />}
      caption={
        <>
          A diffusion model learning 1,500 tiny (8 × 8) pictures of trousers, sneakers and bags. Each row of thumbnails is the same 16 starting patches of noise,
          denoised by the model as it was at the selected step. A thumbnail is outlined when it is as close to some training picture as a genuinely new real picture
          usually is (distance {BAR.toFixed(2)} or less). {live ? 'You are looking at your own run.' : 'This is the recorded run; Train it yourself runs it again here.'}
        </>
      }
    >
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <button className="btn" onClick={() => (setSel(0), setPlaying(true))}>
          ▶ Replay its life
        </button>
        <button className={`btn${running ? '' : ' primary'}`} disabled={!data || tier.kind === 'checking'} onClick={() => (running ? (stop.current = true) : run())}>
          {running ? '❚❚ Stop' : 'Train it yourself'}
        </button>
        <label className="slider" style={{ flex: '1 1 220px' }}>
          <span className="slider-top">
            <span className="slider-label">timeline</span>
            <span className="slider-value">step {fmt(cur.step)}</span>
          </span>
          <input type="range" min={0} max={cps.length - 1} value={i} onChange={(e) => setSel(+e.target.value)} aria-label="Training timeline" />
        </label>
      </div>
      {flash && (
        <div className="birth-flash" role="status">
          Step {fmt(flash.step)}: {flash.label}.
        </div>
      )}
      <div className="btn-row" style={{ flexWrap: 'wrap', gap: 6, margin: '6px 0' }}>
        {moments.map((m, k) => (
          <button key={m.id} className="btn" style={{ fontSize: 12.5 }} onClick={() => setSel(cps.findIndex((c) => c.step === m.step))}>
            {k + 1}. step {fmt(m.step)}: {m.label}
          </button>
        ))}
      </div>
      {[0, i].filter((v, k, a) => a.indexOf(v) === k).map((idx) => (
        <div key={idx} style={{ marginTop: 8 }}>
          <div className="plot-tick">step {fmt(cps[idx].step)}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {cps[idx].samples.map((s, k) => (
              <button key={k} onClick={() => setPick(k)} style={{ padding: 0, border: 'none', background: 'none', cursor: 'pointer', outline: idx === i && k === pick ? '2px solid var(--accent)' : 'none', borderRadius: 4 }}>
                <Pic v={s} ok={cps[idx].nn[k] <= BAR} label={`Sample ${k + 1} at step ${cps[idx].step}`} />
              </button>
            ))}
          </div>
        </div>
      ))}
      {data && (
        <div style={{ marginTop: 8 }}>
          <div className="plot-tick">for comparison: 16 real training pictures at the same size (every 94th)</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {Array.from({ length: 16 }, (_, k) => (
              <Pic key={k} v={data.train[k * 94]} label={`Real training picture ${k + 1}`} />
            ))}
          </div>
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12, marginTop: 12 }}>
        <div className="gpt-panel">
          <div className="field-label" style={{ textTransform: 'none' }}>Picture {pick + 1} and its nearest training picture</div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 6 }}>
            <Pic v={cur.samples[pick]} size={96} label="The selected sample" ok={cur.nn[pick] <= BAR} />
            {nearest && <Pic v={nearest.img} size={96} label="Its nearest training picture" />}
          </div>
          <div className="plot-tick" style={{ marginTop: 4 }}>
            distance {cur.nn[pick].toFixed(2)}; a new real picture is usually within {BAR.toFixed(2)} of some training picture. A distance above 0 means it is not a copy.
          </div>
        </div>
        <div className="gpt-panel">
          <div className="field-label" style={{ textTransform: 'none' }}>One picture appearing, at the last step: noise level 99 down to 0</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
            {DRAW_RESULTS.journey.map((x, k) => (
              <Pic key={k} v={x} size={40} label={`Denoising frame ${k + 1}`} />
            ))}
          </div>
          <div className="plot-tick" style={{ marginTop: 4 }}>The model removes a little predicted noise at each of the 100 levels; the frames are levels 99, 90, 75, 60, 45, 30, 20, 10, 5 and 0.</div>
        </div>
      </div>
      <p className="live-note">
        Step {fmt(cur.step)}: {inRange} of the 16 pictures are within the real-picture distance of some training picture. {cur.step === 0 ? 'At step 0 the denoiser knows nothing, so the "pictures" are just smoothed noise.' : ''}
      </p>
    </Widget>
  );
}
