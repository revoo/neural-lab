// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
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
  type AliasMethod,
  type ColorRep,
  type PatternId,
  type RGB,
  type TensorDtype,
  type TensorLayout,
  type TensorNorm,
  type VisionBriefId,
} from './vision';

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

export function VisionPreprocessorLab() {
  const [briefId, setBriefId] = useState<VisionBriefId>('lighting');
  const [st, setSt] = useState<VLabState>(loadVLab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(VLAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalVisionBrief1(st.rep), [st.rep]);
  const r2 = useMemo(() => evalVisionBrief2(st.alias), [st.alias]);
  const r3 = useMemo(() => evalVisionBrief3(st.layout, st.dtype, st.norm), [st.layout, st.dtype, st.norm]);
  const exhaust = useMemo(() => visionLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'lighting' ? r1.pass : briefId === 'aliasing' ? r2.pass : r3.pass;

  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const brief = LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Build the Vision Preprocessor"
      wide
      caption={
        <>
          Three real computer-vision input pipelines checked exhaustively across all menu combinations. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {LAB_BRIEFS.map((b, i) => (
          <button
            key={b.id}
            type="button"
            className="btn"
            style={{
              borderColor: briefId === b.id ? 'var(--accent)' : 'var(--border)',
              background: briefId === b.id ? 'var(--bg-sunken)' : 'transparent',
              fontWeight: briefId === b.id ? 600 : 400,
            }}
            onClick={() => {
              setBriefId(b.id);
              setShowOpt(false);
            }}
          >
            {st.solved[b.id] ? '✓ ' : ''}Brief {i + 1}
          </button>
        ))}
      </div>

      <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{brief.title}</div>
      <p style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 0, marginBottom: 12, lineHeight: 1.5 }}>{brief.story}</p>

      {briefId === 'lighting' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="Colour representation"
            value={st.rep}
            onChange={(v) => setSt((p) => ({ ...p, rep: v }))}
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
                {(100 * r1.acc).toFixed(1)}% {r1.pass ? 'PASS' : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'aliasing' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Segmented
            label="Downsampling method"
            value={st.alias}
            onChange={(v) => setSt((p) => ({ ...p, alias: v }))}
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
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.pass ? 'PASS (Unique!)' : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'vram' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Axis layout"
              value={st.layout}
              onChange={(v) => setSt((p) => ({ ...p, layout: v }))}
              options={[
                { value: 'NHWC', label: 'NHWC (channels last)' },
                { value: 'NCHW', label: 'NCHW (PyTorch)' },
              ]}
            />
            <Segmented
              label="Data type"
              value={st.dtype}
              onChange={(v) => setSt((p) => ({ ...p, dtype: v }))}
              options={[
                { value: 'uint8', label: 'uint8 (1 B)' },
                { value: 'fp16', label: 'FP16 (2 B)' },
                { value: 'fp32', label: 'FP32 (4 B)' },
              ]}
            />
            <Segmented
              label="Normalisation"
              value={st.norm}
              onChange={(v) => setSt((p) => ({ ...p, norm: v }))}
              options={[
                { value: 'none', label: 'Raw 0..255' },
                { value: 'div255', label: 'Divide by 255' },
                { value: 'zscore', label: 'Channel z-score' },
              ]}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Batch VRAM (≤ 24 MiB)</div>
              <div className="readout-value" style={{ color: r3.memOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.mib.toFixed(3)} MiB
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Layout</div>
              <div className="readout-value" style={{ color: r3.layoutOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.layoutOk ? 'NCHW ✓' : 'Wrong axis'}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Normalised?</div>
              <div className="readout-value" style={{ color: r3.normOk ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.normOk ? 'μ=0, σ=1 ✓' : st.dtype === 'uint8' && st.norm === 'zscore' ? 'uint8 clips <0!' : 'Not centred'}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r3.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.pass ? 'PASS (Unique!)' : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
        <button type="button" className="btn" onClick={() => setShowOpt((v) => !v)}>
          {showOpt ? 'Hide exhaustive check' : `Reveal exhaustive search (${exhaust.total} configs)`}
        </button>
        {showOpt && (
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            <strong>{exhaust.passing}/{exhaust.total}</strong> pass · {exhaust.summary}
          </div>
        )}
      </div>
    </Widget>
  );
}
