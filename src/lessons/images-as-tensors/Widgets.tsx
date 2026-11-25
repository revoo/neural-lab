// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweened } from '../../components/motion';
import { Segmented, Slider } from '../../components/controls';
import {
  ALIAS_METHODS,
  COLOR_REPS,
  LAB_BRIEFS,
  MEMORY_PRESETS,
  PRIMARY_SWATCHES,
  SPRITES,
  compareLighting,
  downsampleGray,
  evalVisionBrief1,
  evalVisionBrief2,
  evalVisionBrief3,
  gridMean,
  makeGrayPattern,
  naiveGray,
  rec709Luminance,
  rgbToHsv,
  tensorBytes,
  visionLabExhaustive,
  visionLabHint,
  type AliasMethod,
  type ColorRep,
  type PatternId,
  type RGB,
  type TensorDtype,
  type TensorLayout,
  type TensorNorm,
  type VisionBriefId,
} from './vision';
import { visionCodeLines } from './realCode';

const fmtBytes = (b: number) => {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / (1024 * 1024)).toFixed(2)} MiB`;
};

// ------------------------------------------------------------------ 1. Pixel-Grid Tensor Inspector

export function PixelTensorWidget() {
  const [spriteId, setSpriteId] = useState<string>(SPRITES[0].id);
  const [sel, setSel] = useState<[number, number]>([2, 2]);
  const [keepR, setKeepR] = useState(true);
  const [keepG, setKeepG] = useState(true);
  const [keepB, setKeepB] = useState(true);
  const [presetIdx, setPresetIdx] = useState<number>(2);

  const sprite = useMemo(() => SPRITES.find((s) => s.id === spriteId) ?? SPRITES[0], [spriteId]);
  const rawPix = sprite.pixels[sel[0]][sel[1]];
  const shownPix: RGB = [keepR ? rawPix[0] : 0, keepG ? rawPix[1] : 0, keepB ? rawPix[2] : 0];
  const preset = MEMORY_PRESETS[presetIdx];

  const renderGrid = (mode: 'rgb' | 'r' | 'g' | 'b', size: number) => (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${sprite.w}, ${size}px)`,
        gap: 1,
        background: 'var(--border)',
        padding: 1,
        borderRadius: 6,
        width: 'fit-content',
      }}
    >
      {sprite.pixels.map((row, r) =>
        row.map(([R, G, B], c) => {
          const isSel = r === sel[0] && c === sel[1];
          const rr = keepR ? R : 0;
          const gg = keepG ? G : 0;
          const bb = keepB ? B : 0;
          const bg =
            mode === 'rgb'
              ? `rgb(${rr},${gg},${bb})`
              : mode === 'r'
                ? `rgb(${rr},0,0)`
                : mode === 'g'
                  ? `rgb(0,${gg},0)`
                  : `rgb(0,0,${bb})`;
          return (
            <button
              key={`${r}-${c}`}
              type="button"
              aria-label={`Pixel row ${r} col ${c}`}
              onClick={() => setSel([r, c])}
              style={{
                width: size,
                height: size,
                padding: 0,
                border: isSel ? '2px solid var(--accent)' : 'none',
                background: bg,
                cursor: 'pointer',
                boxSizing: 'border-box',
              }}
            />
          );
        }),
      )}
    </div>
  );

  return (
    <Widget
      title="Inside an image tensor: (H, W, C) pixels, RGB channels & memory"
      wide
      caption={
        <>
          Click any pixel to inspect its exact <code>[R, G, B]</code> coordinates inside the <code>(12, 12, 3)</code> tensor. Mute channels to see how yellow is pure <strong>Red + Green</strong> and white needs all three. Below, compare exact tensor memory across resolutions and dtypes.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Sprite"
          value={spriteId}
          onChange={setSpriteId}
          options={SPRITES.map((s) => ({ value: s.id, label: s.name }))}
        />
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button type="button" className="btn" onClick={() => setKeepR((v) => !v)} style={{ borderColor: keepR ? 'var(--c-bad)' : 'var(--border)' }}>
            {keepR ? '✓ Red' : '✕ Red muted'}
          </button>
          <button type="button" className="btn" onClick={() => setKeepG((v) => !v)} style={{ borderColor: keepG ? 'var(--c-good)' : 'var(--border)' }}>
            {keepG ? '✓ Green' : '✕ Green muted'}
          </button>
          <button type="button" className="btn" onClick={() => setKeepB((v) => !v)} style={{ borderColor: keepB ? 'var(--c-a)' : 'var(--border)' }}>
            {keepB ? '✓ Blue' : '✕ Blue muted'}
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Composite RGB (12×12×3)</div>
          {renderGrid('rgb', 16)}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Channel 0 (Red)</div>
            {renderGrid('r', 10)}
          </div>
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Channel 1 (Green)</div>
            {renderGrid('g', 10)}
          </div>
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Channel 2 (Blue)</div>
            {renderGrid('b', 10)}
          </div>
        </div>

        <div style={{ flex: '1 1 220px', padding: '10px 12px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', fontSize: 13 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <div style={{ width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)', background: `rgb(${shownPix[0]},${shownPix[1]},${shownPix[2]})` }} />
            <div>
              <div style={{ fontWeight: 600 }}>Pixel <code>img[{sel[0]}, {sel[1]}]</code></div>
              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>PyTorch NCHW: <code>x[0, :, {sel[0]}, {sel[1]}]</code></div>
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 12.5, lineHeight: 1.6 }}>
            <div>uint8: [{shownPix[0]}, {shownPix[1]}, {shownPix[2]}]</div>
            <div>float: [{(shownPix[0] / 255).toFixed(2)}, {(shownPix[1] / 255).toFixed(2)}, {(shownPix[2] / 255).toFixed(2)}]</div>
            <div>Rec.709 Y: {rec709Luminance(shownPix).toFixed(1)} · Hue: {rgbToHsv(shownPix).h.toFixed(0)}°</div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontSize: 12.5, color: 'var(--text-muted)', fontWeight: 600 }}>Memory calculator:</span>
          {MEMORY_PRESETS.map((p, i) => (
            <button
              key={p.label}
              type="button"
              className="btn"
              style={{
                fontSize: 12,
                padding: '4px 8px',
                borderColor: presetIdx === i ? 'var(--accent)' : 'var(--border)',
                background: presetIdx === i ? 'var(--bg-sunken)' : 'transparent',
              }}
              onClick={() => setPresetIdx(i)}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
          <div className="readout">
            <div className="readout-label">Shape (N, C, H, W)</div>
            <div className="readout-value" style={{ fontSize: 15 }}>({preset.batch}, {preset.c}, {preset.h}, {preset.w})</div>
          </div>
          <div className="readout">
            <div className="readout-label">uint8 (1 B)</div>
            <div className="readout-value">{fmtBytes(tensorBytes(preset.batch, preset.h, preset.w, preset.c, 1))}</div>
          </div>
          <div className="readout">
            <div className="readout-label">FP16 (2 B)</div>
            <div className="readout-value" style={{ color: 'var(--c-a)' }}>{fmtBytes(tensorBytes(preset.batch, preset.h, preset.w, preset.c, 2))}</div>
          </div>
          <div className="readout">
            <div className="readout-label">FP32 (4 B)</div>
            <div className="readout-value" style={{ color: 'var(--c-b)' }}>{fmtBytes(tensorBytes(preset.batch, preset.h, preset.w, preset.c, 4))}</div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Colour Spaces: Luminance & Shadow Invariance

export function ColorSpaceWidget() {
  const [swatchId, setSwatchId] = useState<string>('red');
  const [dimFactor, setDimFactor] = useState<number>(0.5);

  const base: RGB = swatchId === 'red' ? [235, 38, 38] : swatchId === 'amber' ? [240, 185, 25] : [34, 210, 82];
  const cmp = useMemo(() => compareLighting(base, dimFactor), [base, dimFactor]);

  return (
    <Widget
      title="Why (R+G+B)/3 lies to your eyes — and why shadows break raw RGB"
      wide
      caption={
        <>
          Top: compare naive <code>(R+G+B)/3</code> against human <strong>Rec. 709 luminance</strong> on pure primaries — green is <strong>9.9×</strong> brighter to human cones than blue. Bottom: slide <strong>Lighting brightness</strong> down to <code>30%</code> and watch raw RGB distance explode while <strong>HSV Hue</strong> stays pinned at <strong>0° shift</strong>.
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginBottom: 14 }}>
        {PRIMARY_SWATCHES.slice(0, 3).map((s) => {
          const avg = naiveGray(s.rgb);
          const y = rec709Luminance(s.rgb);
          return (
            <div key={s.id} style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', fontSize: 12.5 }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>{s.name}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4, marginBottom: 6 }}>
                <div style={{ height: 28, borderRadius: 4, background: `rgb(${s.rgb[0]},${s.rgb[1]},${s.rgb[2]})` }} title="Original" />
                <div style={{ height: 28, borderRadius: 4, background: `rgb(${Math.round(avg)},${Math.round(avg)},${Math.round(avg)})` }} title="Naive avg" />
                <div style={{ height: 28, borderRadius: 4, background: `rgb(${Math.round(y)},${Math.round(y)},${Math.round(y)})` }} title="Rec. 709 Y" />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span>Avg: <strong>{avg.toFixed(1)}</strong></span>
                <span>Rec.709: <strong>{y.toFixed(1)}</strong></span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ paddingTop: 12, borderTop: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 10 }}>
          <Segmented
            label="Signal colour"
            value={swatchId}
            onChange={setSwatchId}
            options={[
              { value: 'red', label: 'Traffic Red' },
              { value: 'amber', label: 'Traffic Amber' },
              { value: 'green', label: 'Traffic Green' },
            ]}
          />
          <div style={{ flex: '1 1 200px' }}>
            <Slider
              label="Lighting brightness"
              value={dimFactor}
              min={0.2}
              max={1.0}
              step={0.05}
              onChange={setDimFactor}
              format={(v) => `${Math.round(v * 100)}%`}
            />
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: 52, height: 52, borderRadius: 8, border: '1px solid var(--border)', background: `rgb(${cmp.base[0]},${cmp.base[1]},${cmp.base[2]})` }} />
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 3 }}>Daylight</div>
            </div>
            <span style={{ color: 'var(--text-muted)' }}>→</span>
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: 52, height: 52, borderRadius: 8, border: '1px solid var(--border)', background: `rgb(${cmp.dimmed[0]},${cmp.dimmed[1]},${cmp.dimmed[2]})` }} />
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 3 }}>Dimmed</div>
            </div>
          </div>

          <div className="readouts" style={{ flex: '1 1 260px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">RGB Euclidean shift</div>
              <div className="readout-value" style={{ color: cmp.rgbDist > 40 ? 'var(--c-bad)' : 'var(--text)' }}>
                {cmp.rgbDist.toFixed(1)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">HSV Hue shift</div>
              <div className="readout-value" style={{ color: 'var(--c-good)' }}>
                {cmp.hueShift.toFixed(1)}°
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">HSV Value drop</div>
              <div className="readout-value">{(100 * cmp.valShift).toFixed(0)} pp</div>
            </div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Aliasing vs Box-Average Downsampling

export function AliasingWidget() {
  const [patternId, setPatternId] = useState<PatternId>('checker');
  const [factor, setFactor] = useState<2 | 4>(2);

  const pat = useMemo(() => makeGrayPattern(patternId), [patternId]);
  const strided = useMemo(() => downsampleGray(pat.grid, factor, 'stride'), [pat, factor]);
  const boxed = useMemo(() => downsampleGray(pat.grid, factor, 'box'), [pat, factor]);

  const drawGray = (g: number[][], cellPx: number) => (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${g[0].length}, ${cellPx}px)`,
        gap: 1,
        background: 'var(--border)',
        padding: 1,
        borderRadius: 6,
        width: 'fit-content',
      }}
    >
      {g.map((row, r) =>
        row.map((v, c) => {
          const b = Math.round(v);
          return <div key={`${r}-${c}`} style={{ width: cellPx, height: cellPx, background: `rgb(${b},${b},${b})` }} />;
        }),
      )}
    </div>
  );

  const origMean = gridMean(pat.grid);
  const strMean = gridMean(strided);
  const boxMean = gridMean(boxed);

  return (
    <Widget
      title="Aliasing live: why strided slicing img[::s, ::s] destroys fine detail"
      wide
      caption={
        <>
          Downsample a <code>16×16</code> pattern by <code>{factor}×</code>. On a <strong>1-px Checkerboard</strong>, strided slicing <code>img[::2, ::2]</code> lands exclusively on white squares and turns a 50% grey pattern into <strong>100% solid white (mean 255.0)</strong>. Box averaging integrates every block first, preserving exact mean brightness (<strong>127.5</strong>).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Pattern"
          value={patternId}
          onChange={setPatternId}
          options={[
            { value: 'checker', label: '1-px Checkerboard' },
            { value: 'stripes', label: '1-px Vertical Bars' },
            { value: 'grid', label: 'Thin Blueprint Grid' },
          ]}
        />
        <Segmented
          label="Downsample factor"
          value={factor}
          onChange={setFactor}
          options={[
            { value: 2, label: '2× (16→8)' },
            { value: 4, label: '4× (16→4)' },
          ]}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 14, alignItems: 'start' }}>
        <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>Original (16×16)</div>
          {drawGray(pat.grid, 8)}
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 6 }}>Mean: <strong>{origMean.toFixed(1)}</strong></div>
        </div>
        <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6, color: 'var(--c-bad)' }}>Strided [::{factor}, ::{factor}]</div>
          {drawGray(strided, factor === 2 ? 17 : 35)}
          <div style={{ fontSize: 12.5, color: Math.abs(strMean - origMean) > 10 ? 'var(--c-bad)' : 'var(--text-muted)', marginTop: 6 }}>
            Mean: <strong>{strMean.toFixed(1)}</strong> (err {Math.abs(strMean - origMean).toFixed(1)})
          </div>
        </div>
        <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6, color: 'var(--c-good)' }}>{factor}×{factor} Box Average</div>
          {drawGray(boxed, factor === 2 ? 17 : 35)}
          <div style={{ fontSize: 12.5, color: 'var(--c-good)', marginTop: 6 }}>
            Mean: <strong>{boxMean.toFixed(1)}</strong> (err {Math.abs(boxMean - origMean).toFixed(1)})
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Build the Vision Preprocessor

type VLabState = {
  rep: ColorRep;
  alias: AliasMethod;
  layout: TensorLayout;
  dtype: TensorDtype;
  norm: TensorNorm;
  solved: Record<VisionBriefId, boolean>;
};

const VLAB_KEY = 'nl-img-lab-v1';
const VLAB_DEFAULT: VLabState = {
  rep: 'raw-rgb',
  alias: 'stride-0',
  layout: 'NHWC',
  dtype: 'fp32',
  norm: 'none',
  solved: { lighting: false, aliasing: false, vram: false },
};

function loadVLab(): VLabState {
  try {
    const raw = localStorage.getItem(VLAB_KEY);
    if (!raw) return VLAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...VLAB_DEFAULT, ...p, solved: { ...VLAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return VLAB_DEFAULT;
  }
}

function seedVLabProgress() {
  const old = loadVLab();
  const solved: Record<string, boolean> = {};
  for (const b of LAB_BRIEFS) {
    if (old.solved[b.id]) solved[b.id] = true;
  }
  return { solved, optimal: {} };
}

const SIGNAL_NAMES = ['Red', 'Amber', 'Green'] as const;

export function VisionPreprocessorLab() {
  const [briefIdx, setBriefIdx] = useState(0);
  const [st, setSt] = useState<VLabState>(loadVLab);
  const progress = useLabProgress('nl-img-lab-v1-shell', seedVLabProgress);
  const log = useMachineLog();

  useEffect(() => {
    try {
      localStorage.setItem(VLAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const brief = LAB_BRIEFS[briefIdx];
  const briefId = brief.id;

  const r1 = useMemo(() => evalVisionBrief1(st.rep), [st.rep]);
  const r2 = useMemo(() => evalVisionBrief2(st.alias), [st.alias]);
  const r3 = useMemo(() => evalVisionBrief3(st.layout, st.dtype, st.norm), [st.layout, st.dtype, st.norm]);
  const exhaust = useMemo(() => visionLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'lighting' ? r1.pass : briefId === 'aliasing' ? r2.pass : r3.pass;
  const curOpt =
    curPass && (briefId === 'lighting' ? r1.acc >= 1 - 1e-9 : true);

  const shownMib = useTweened(r3.mib, 240);

  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const updateState = (patch: Partial<VLabState>) => {
    const next = { ...st, ...patch };
    setSt(next);
    if (briefId === 'lighting') {
      const r = evalVisionBrief1(next.rep);
      log.push(
        `colour=${next.rep} → day ${(100 * r.dayAcc).toFixed(1)}% · shadow ${(100 * r.shadowAcc).toFixed(1)}% · overall ${(100 * r.acc).toFixed(1)}% ${r.pass ? '✓' : '✗'}`,
        r.pass ? 'good' : 'info',
      );
    } else if (briefId === 'aliasing') {
      const r = evalVisionBrief2(next.alias);
      log.push(
        `downsample=${next.alias} → mean err ${r.meanErr.toFixed(1)} · block RMSE ${r.rmse.toFixed(1)} ${r.pass ? '✓' : '✗'}`,
        r.pass ? 'good' : 'info',
      );
    } else {
      const r = evalVisionBrief3(next.layout, next.dtype, next.norm);
      log.push(
        `batch ${next.layout} ${next.dtype.toUpperCase()} norm=${next.norm} → ${r.mib.toFixed(3)} MiB · μ=${r.outMean.toFixed(2)} σ=${r.outStd.toFixed(2)} ${r.pass ? '✓' : '✗'}`,
        r.pass ? 'good' : 'info',
      );
    }
  };

  const resetBrief = () => {
    if (briefId === 'lighting') updateState({ rep: 'raw-rgb' });
    else if (briefId === 'aliasing') updateState({ alias: 'stride-0' });
    else updateState({ layout: 'NHWC', dtype: 'fp32', norm: 'none' });
  };

  const liveCode = useMemo(() => visionCodeLines(briefId, st), [briefId, st]);
  const shellBriefs = useMemo(
    () => LAB_BRIEFS.map((b, i) => ({ id: b.id, tab: `Brief ${i + 1}`, title: b.title, text: b.story })),
    [],
  );

  const statusNode =
    briefId === 'lighting' ? (
      r1.pass ? (
        <>
          <strong>Optimal — 100.0% across both daylight and heavy shadow!</strong> All 90 road-camera frames match their daylight reference cleanly.
        </>
      ) : (
        <>
          <strong>Not yet.</strong> Overall {(100 * r1.acc).toFixed(1)}% (need ≥ 95.0%) · daylight {(100 * r1.dayAcc).toFixed(1)}%, shadow <strong>{(100 * r1.shadowAcc).toFixed(1)}%</strong>. Look at which shadowed frames below get mislabelled.
        </>
      )
    ) : briefId === 'aliasing' ? (
      r2.pass ? (
        <>
          <strong>Unique anti-aliased winner!</strong> Mean brightness error <strong>{r2.meanErr.toFixed(1)}</strong> · block RMSE <strong>{r2.rmse.toFixed(1)}</strong> (exact match to true area average).
        </>
      ) : (
        <>
          <strong>Not yet — aliased!</strong> Mean error {r2.meanErr.toFixed(1)} (need ≤ 5.0) · block RMSE {r2.rmse.toFixed(1)} (need ≤ 15.0). Compare your 4×4 thumbnail to the true reference on the right.
        </>
      )
    ) : r3.pass ? (
      <>
        <strong>Unique passing PyTorch batch!</strong> <code>(64, 3, 224, 224)</code> FP16 z-scored fits in <strong>{r3.mib.toFixed(3)} MiB</strong> (under 24 MiB) with μ = 0.00, σ = 1.00.
      </>
    ) : (
      <>
        <strong>Not yet.</strong> {r3.mib.toFixed(3)} MiB ({r3.memOk ? 'fits ≤ 24 MiB' : 'over 24 MiB!'}) · layout {r3.layoutOk ? 'NCHW ✓' : 'NHWC ✗'} · normalisation {r3.normOk ? 'centred ✓' : st.dtype === 'uint8' && st.norm === 'zscore' ? 'uint8 clips negatives to 0!' : 'not centred'}.
      </>
    );

  // Brief 3 channel-distribution bins (12 buckets)
  const distBins = useMemo(() => {
    const vals = r3.rVals;
    const minV = st.norm === 'none' ? 0 : st.norm === 'div255' ? 0 : -2.5;
    const maxV = st.norm === 'none' ? 255 : st.norm === 'div255' ? 1 : 2.5;
    const nBins = 12;
    const counts = new Array<number>(nBins).fill(0);
    let negClipped = 0;
    for (const v of vals) {
      if (st.dtype === 'uint8' && st.norm === 'zscore' && v === 0) negClipped++;
      const t = (v - minV) / Math.max(1e-9, maxV - minV);
      const b = Math.max(0, Math.min(nBins - 1, Math.floor(t * nBins)));
      counts[b]++;
    }
    const maxC = Math.max(1, ...counts);
    return { counts, maxC, minV, maxV, negClipped };
  }, [r3.rVals, st.norm, st.dtype]);

  return (
    <LabShell
      title="Lab: Build the Vision Preprocessor"
      caption={
        <>
          Three real computer-vision input pipelines checked exhaustively. Inspect the live traffic-light frames (Brief 1), the 16×16 blueprint grid vs 4×4 thumbnail (Brief 2), and the batch VRAM bar + channel value histogram (Brief 3).
        </>
      }
      briefs={shellBriefs}
      active={briefIdx}
      onSelect={(i) => setBriefIdx(i)}
      progress={progress}
      passing={curPass}
      optimal={curOpt}
      status={statusNode}
      statusTone={curPass ? 'good' : 'neutral'}
      hint={visionLabHint(briefId, st)}
      reveal={{
        label: `the exhaustive search (${exhaust.total} configs)`,
        content: (
          <div style={{ fontSize: 13.5, lineHeight: 1.55 }}>
            All <strong>{exhaust.total}</strong> configurations were checked exhaustively: <strong>{exhaust.passing}</strong> pass. {exhaust.summary}.
          </div>
        ),
      }}
      log={{
        lines: log.lines,
        onClear: log.clear,
        title: 'Vision preprocessor log',
        empty: 'Change representation, downsampling or tensor settings to record a run.',
      }}
      actions={
        <button type="button" className="btn small" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {/* Brief 1: Live Traffic Light Frames (Daylight vs Tunnel Shadow) + Confusion Matrix */}
      {briefId === 'lighting' && (
        <div style={{ display: 'grid', gap: 12 }}>
          <Segmented
            label="Colour representation"
            value={st.rep}
            onChange={(v) => updateState({ rep: v })}
            options={COLOR_REPS}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Daylight (45 frames)</div>
              <div className="readout-value">{(100 * r1.dayAcc).toFixed(1)}%</div>
            </div>
            <div className="readout">
              <div className="readout-label">Shadow (45 frames)</div>
              <div className="readout-value" style={{ color: r1.shadowAcc >= 0.95 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {(100 * r1.shadowAcc).toFixed(1)}%
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Overall (≥ 95%)</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {(100 * r1.acc).toFixed(1)}%
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12, alignItems: 'start' }}>
            {/* Live sample frames */}
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 8 }}>
              <div className="field-label" style={{ margin: 0 }}>Live road-camera crops (top: sunny · bottom: tunnel shadow)</div>
              {[false, true].map((shadowRow) => (
                <div key={String(shadowRow)} style={{ display: 'grid', gap: 4 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }} className="mono">
                    {shadowRow ? 'Tunnel shadow (~25% brightness)' : 'Sunny daylight (~100% brightness)'}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 5 }}>
                    {r1.preview
                      .filter((f) => f.isShadow === shadowRow)
                      .map((f, idx) => {
                        const ok = f.pred === f.truth;
                        return (
                          <div
                            key={idx}
                            title={`True ${SIGNAL_NAMES[f.truth]} · RGB(${f.rgb.join(', ')}) · Predicted ${SIGNAL_NAMES[f.pred]}`}
                            style={{
                              borderRadius: 6,
                              padding: 4,
                              border: `2px solid ${ok ? 'var(--c-good)' : 'var(--c-bad)'}`,
                              background: 'var(--bg)',
                              display: 'grid',
                              gap: 3,
                              justifyItems: 'center',
                            }}
                          >
                            <div
                              style={{
                                width: 22,
                                height: 22,
                                borderRadius: '50%',
                                background: `rgb(${f.rgb[0]},${f.rgb[1]},${f.rgb[2]})`,
                                border: '1px solid var(--border)',
                              }}
                            />
                            <div className="mono" style={{ fontSize: 10, color: ok ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
                              {SIGNAL_NAMES[f.pred][0]} {ok ? '✓' : '✗'}
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              ))}
            </div>

            {/* 3x3 confusion matrix */}
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 6 }}>
              <div className="field-label" style={{ margin: 0 }}>90-frame confusion matrix (rows: true · cols: predicted)</div>
              <div style={{ overflowX: 'auto' }}>
                <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>True \ Pred</th>
                      <th>Red</th>
                      <th>Amber</th>
                      <th>Green</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SIGNAL_NAMES.map((name, rIdx) => (
                      <tr key={name}>
                        <td style={{ fontWeight: 600 }}>{name}</td>
                        {[0, 1, 2].map((cIdx) => {
                          const v = r1.confusion[rIdx][cIdx];
                          const diag = rIdx === cIdx;
                          return (
                            <td
                              key={cIdx}
                              className="mono"
                              style={{
                                fontWeight: diag ? 700 : v > 0 ? 600 : 400,
                                color: diag ? 'var(--c-good)' : v > 0 ? 'var(--c-bad)' : 'var(--text-muted)',
                              }}
                            >
                              {v}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Brief 2: Live 16x16 Blueprint Source -> Your 4x4 Thumbnail vs True Reference */}
      {briefId === 'aliasing' && (
        <div style={{ display: 'grid', gap: 12 }}>
          <Segmented
            label="Downsampling method"
            value={st.alias}
            onChange={(v) => updateState({ alias: v })}
            options={ALIAS_METHODS}
          />
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Mean error (≤ 5.0)</div>
              <div className="readout-value" style={{ color: r2.meanErr <= 5 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.meanErr.toFixed(1)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Block RMSE (≤ 15.0)</div>
              <div className="readout-value" style={{ color: r2.rmse <= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.rmse.toFixed(1)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Output mean (true 125.0)</div>
              <div className="readout-value">{gridMean(r2.out).toFixed(1)}</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(180px, 100%), 1fr))', gap: 12, alignItems: 'center', padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
            {/* Source 16x16 grid with sampled pixels marked */}
            <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)' }}>16×16 blueprint (sampled px ringed)</div>
              <svg viewBox="0 0 160 160" width={156} height={156} style={{ maxWidth: '100%', borderRadius: 6, border: '1px solid var(--border)' }}>
                {r2.src.map((row, r) =>
                  row.map((v, c) => {
                    const g = Math.round(v);
                    const hit =
                      (st.alias === 'stride-0' && r % 4 === 0 && c % 4 === 0) ||
                      (st.alias === 'stride-1' && r % 4 === 1 && c % 4 === 1);
                    return (
                      <g key={`${r}-${c}`}>
                        <rect x={c * 10} y={r * 10} width={10} height={10} fill={`rgb(${g},${g},${g})`} />
                        {hit && <circle cx={c * 10 + 5} cy={r * 10 + 5} r={3.2} fill="none" stroke="var(--accent)" strokeWidth={1.6} />}
                      </g>
                    );
                  }),
                )}
                {[40, 80, 120].map((p) => (
                  <g key={p}>
                    <line x1={p} x2={p} y1={0} y2={160} stroke="var(--accent)" strokeOpacity={0.45} strokeDasharray="2 2" />
                    <line x1={0} x2={160} y1={p} y2={p} stroke="var(--accent)" strokeOpacity={0.45} strokeDasharray="2 2" />
                  </g>
                ))}
              </svg>
            </div>

            {/* Learner's 4x4 output */}
            <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                Your 4×4 output ({st.alias})
              </div>
              <svg viewBox="0 0 120 120" width={120} height={120} style={{ borderRadius: 6, border: `2px solid ${r2.pass ? 'var(--c-good)' : 'var(--c-bad)'}` }}>
                {r2.out.map((row, r) =>
                  row.map((v, c) => {
                    const g = Math.round(v);
                    return (
                      <g key={`${r}-${c}`}>
                        <rect x={c * 30} y={r * 30} width={30} height={30} fill={`rgb(${g},${g},${g})`} />
                        <text x={c * 30 + 15} y={r * 30 + 18} textAnchor="middle" fontSize={9.5} fill={g > 130 ? 'rgb(18,18,18)' : 'rgb(242,242,242)'} fontFamily="monospace">
                          {g}
                        </text>
                      </g>
                    );
                  }),
                )}
              </svg>
            </div>

            {/* True area-averaged 4x4 reference */}
            <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-muted)' }}>True 4×4 area average</div>
              <svg viewBox="0 0 120 120" width={120} height={120} style={{ borderRadius: 6, border: '1px solid var(--border)' }}>
                {r2.truth.map((row, r) =>
                  row.map((v, c) => {
                    const g = Math.round(v);
                    return (
                      <g key={`${r}-${c}`}>
                        <rect x={c * 30} y={r * 30} width={30} height={30} fill={`rgb(${g},${g},${g})`} />
                        <text x={c * 30 + 15} y={r * 30 + 18} textAnchor="middle" fontSize={9.5} fill={g > 130 ? 'rgb(18,18,18)' : 'rgb(242,242,242)'} fontFamily="monospace">
                          {g}
                        </text>
                      </g>
                    );
                  }),
                )}
              </svg>
            </div>
          </div>
        </div>
      )}

      {/* Brief 3: VRAM Budget Bar + Pixel Distribution Clipping X-Ray */}
      {briefId === 'vram' && (
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Axis layout"
              value={st.layout}
              onChange={(v) => updateState({ layout: v })}
              options={[
                { value: 'NHWC', label: 'NHWC (channels last)' },
                { value: 'NCHW', label: 'NCHW (PyTorch)' },
              ]}
            />
            <Segmented
              label="Data type"
              value={st.dtype}
              onChange={(v) => updateState({ dtype: v })}
              options={[
                { value: 'uint8', label: 'uint8 (1 B)' },
                { value: 'fp16', label: 'FP16 (2 B)' },
                { value: 'fp32', label: 'FP32 (4 B)' },
              ]}
            />
            <Segmented
              label="Normalisation"
              value={st.norm}
              onChange={(v) => updateState({ norm: v })}
              options={[
                { value: 'none', label: 'Raw 0..255' },
                { value: 'div255', label: 'Divide by 255' },
                { value: 'zscore', label: 'Channel z-score' },
              ]}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12 }}>
            {/* Animated VRAM bar + shape readout */}
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span className="field-label" style={{ margin: 0 }}>Batch VRAM vs 24 MiB ceiling</span>
                <span className="mono" style={{ fontSize: 12.5, fontWeight: 700, color: r3.memOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {shownMib.toFixed(2)} / 24.0 MiB
                </span>
              </div>
              <div style={{ position: 'relative', height: 18, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${Math.min(100, (shownMib / 40) * 100)}%`,
                    height: '100%',
                    background: r3.memOk ? 'var(--c-good)' : 'var(--c-bad)',
                    opacity: 0.85,
                  }}
                />
                <div
                  title="24 MiB budget ceiling"
                  style={{
                    position: 'absolute',
                    left: `${(24 / 40) * 100}%`,
                    top: 0,
                    bottom: 0,
                    width: 2,
                    background: 'var(--text)',
                  }}
                />
              </div>
              <div className="mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                shape: <strong>{st.layout === 'NCHW' ? '(64, 3, 224, 224)' : '(64, 224, 224, 3)'}</strong>{' '}
                {r3.layoutOk ? '✓' : '✗ PyTorch expects NCHW'}
              </div>
            </div>

            {/* Channel value distribution showing uint8 negative clipping! */}
            <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span className="field-label" style={{ margin: 0 }}>R-channel value distribution</span>
                <span className="mono" style={{ fontSize: 11.5, color: r3.normOk ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 600 }}>
                  {r3.normOk ? 'μ=0.00, σ=1.00 ✓' : st.dtype === 'uint8' && st.norm === 'zscore' ? `${distBins.negClipped} px clipped to 0!` : `μ=${r3.outMean.toFixed(1)}, σ=${r3.outStd.toFixed(1)}`}
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 3, alignItems: 'end', height: 52 }}>
                {distBins.counts.map((c, idx) => (
                  <div
                    key={idx}
                    style={{
                      height: `${Math.max(4, (c / distBins.maxC) * 100)}%`,
                      borderRadius: '3px 3px 0 0',
                      background: r3.normOk ? 'var(--c-good)' : st.dtype === 'uint8' && st.norm === 'zscore' && idx === 6 ? 'var(--c-bad)' : 'var(--c-a)',
                      opacity: 0.85,
                    }}
                  />
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, color: 'var(--text-muted)' }} className="mono">
                <span>{distBins.minV}</span>
                <span>{((distBins.minV + distBins.maxV) / 2).toFixed(1)}</span>
                <span>{distBins.maxV}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      <LiveCode code={liveCode} note="NumPy / PyTorch preprocessing pipeline" />
    </LabShell>
  );
}
