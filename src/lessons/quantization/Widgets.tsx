// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { LiveCode } from '../../components/LabJob';
import { llamaCppLines } from './realCode';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  BASE_WEIGHTS,
  FLOAT_FORMATS,
  LAB_BIT_BUDGET,
  LAB_BLOCKS,
  LAB_RMSE_GOAL,
  LAB_SCHEMES,
  decodeFloatBits,
  encodeFloatBits,
  evaluateMix,
  quantizeFloat,
  quantizeIntSymmetric,
  solveQuantLab,
  type FloatSpec,
} from './quant';

const fmtVal = (v: number): string => {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? '+Infinity' : '−Infinity';
  if (v === 0) return Object.is(v, -0) ? '−0' : '0';
  const av = Math.abs(v);
  if (av >= 1e6 || av < 1e-3) return v.toExponential(3);
  return Number(v.toPrecision(6)).toString();
};

// ------------------------------------------------------------------ 1. Bit-level float explorer

const INTERACTIVE_SPECS = FLOAT_FORMATS.slice(1); // FP16, BF16, FP8
const PRESET_NUMS = [
  { label: '3.14159', val: 3.14159 },
  { label: '100,000', val: 100000 },
  { label: '10⁻⁹', val: 1e-9 },
  { label: '−0.375', val: -0.375 },
  { label: '500', val: 500 },
];

export function FloatBitExplorer() {
  const [specIdx, setSpecIdx] = useState(1); // BF16 default
  const [presetVal, setPresetVal] = useState(3.14159);
  const spec = INTERACTIVE_SPECS[specIdx];

  const [bits, setBits] = useState(() => encodeFloatBits(3.14159, INTERACTIVE_SPECS[1]));

  const loadNumber = (val: number, s: FloatSpec = spec) => {
    setPresetVal(val);
    setBits(encodeFloatBits(val, s));
  };

  const switchSpec = (idx: number) => {
    setSpecIdx(idx);
    setBits(encodeFloatBits(presetVal, INTERACTIVE_SPECS[idx]));
  };

  const toggleSign = () => {
    const sign = bits.sign ? 0 : 1;
    setBits({ ...bits, sign, value: decodeFloatBits(sign, bits.exp, bits.mant, spec.expBits, spec.mantBits, spec.hasInf) });
  };

  const toggleExpBit = (bitPos: number) => {
    const exp = bits.exp ^ (1 << bitPos);
    setBits({ ...bits, exp, value: decodeFloatBits(bits.sign, exp, bits.mant, spec.expBits, spec.mantBits, spec.hasInf) });
  };

  const toggleMantBit = (bitPos: number) => {
    const mant = bits.mant ^ (1 << bitPos);
    setBits({ ...bits, mant, value: decodeFloatBits(bits.sign, bits.exp, mant, spec.expBits, spec.mantBits, spec.hasInf) });
  };

  const expBitList = Array.from({ length: spec.expBits }, (_, i) => spec.expBits - 1 - i);
  const mantBitList = Array.from({ length: spec.mantBits }, (_, i) => spec.mantBits - 1 - i);

  const bitBtnStyle = (on: boolean, color: string) => ({
    width: 26,
    height: 30,
    borderRadius: 5,
    border: `1px solid ${on ? color : 'var(--border)'}`,
    background: on ? `color-mix(in srgb, ${color} 24%, transparent)` : 'var(--bg-sunken)',
    color: on ? 'var(--text)' : 'var(--text-faint)',
    fontWeight: on ? 700 : 500,
    cursor: 'pointer',
    fontSize: 13,
    padding: 0,
  });

  return (
    <Widget
      title="Flip the bits: FP16 vs BF16 vs FP8"
      wide
      caption={
        <>
          Click any preset number — especially <strong>100,000</strong> and <strong>10⁻⁹</strong> — to compare how FP32, FP16, BF16 and FP8 represent it, or click individual sign (orange), exponent (accent) and mantissa (blue) bits to flip them directly.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="format to inspect"
          options={INTERACTIVE_SPECS.map((s, i) => ({ value: i, label: s.name }))}
          value={specIdx}
          onChange={switchSpec}
        />
        <div className="btn-row">
          {PRESET_NUMS.map((p) => (
            <button key={p.label} className={presetVal === p.val ? 'btn primary' : 'btn'} onClick={() => loadNumber(p.val)}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="field-label">
        {spec.name} bits ({spec.bits} bits total — click any bit to flip)
      </div>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px 12px',
          padding: '10px 12px',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          marginBottom: 10,
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 10, color: 'var(--c-b)', textTransform: 'uppercase' }}>sign</span>
          <button type="button" className="mono" style={bitBtnStyle(bits.sign === 1, 'var(--c-b)')} onClick={toggleSign}>
            {bits.sign}
          </button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 10, color: 'var(--accent-text)', textTransform: 'uppercase' }}>
            exponent ({spec.expBits} b, bias {spec.bias})
          </span>
          <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            {expBitList.map((bp) => {
              const on = ((bits.exp >> bp) & 1) === 1;
              return (
                <button key={bp} type="button" className="mono" style={bitBtnStyle(on, 'var(--accent)')} onClick={() => toggleExpBit(bp)}>
                  {on ? 1 : 0}
                </button>
              );
            })}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 10, color: 'var(--c-a)', textTransform: 'uppercase' }}>mantissa ({spec.mantBits} b)</span>
          <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            {mantBitList.map((bp) => {
              const on = ((bits.mant >> bp) & 1) === 1;
              return (
                <button key={bp} type="button" className="mono" style={bitBtnStyle(on, 'var(--c-a)')} onClick={() => toggleMantBit(bp)}>
                  {on ? 1 : 0}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="readout" style={{ fontSize: 13, marginBottom: 12 }}>
        decoded {spec.name} value = <strong style={{ color: !Number.isFinite(bits.value) ? 'var(--c-bad)' : 'var(--c-c)' }}>{fmtVal(bits.value)}</strong> ·{' '}
        max finite = <strong>{fmtVal(spec.maxFinite)}</strong> · exponent field = <strong>{bits.exp}</strong> (unbiased{' '}
        <strong>{bits.exp === 0 ? 1 - spec.bias : bits.exp - spec.bias}</strong>)
      </div>

      <div className="field-label">side-by-side encoding of {fmtVal(presetVal)} across all four formats</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
        {FLOAT_FORMATS.map((f) => {
          const q = quantizeFloat(presetVal, f);
          const broken = !Number.isFinite(q) || (presetVal !== 0 && q === 0);
          return (
            <div
              key={f.id}
              style={{
                background: 'var(--bg-sunken)',
                border: `1px solid ${broken ? 'var(--c-bad)' : 'var(--border)'}`,
                borderRadius: 'var(--radius-sm)',
                padding: '7px 10px',
              }}
            >
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                <strong>{f.name}</strong> ({f.bits} b)
              </div>
              <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: broken ? 'var(--c-bad)' : 'var(--text)', marginTop: 2 }}>
                {fmtVal(q)}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
                {!Number.isFinite(q) ? 'overflow!' : presetVal !== 0 && q === 0 ? 'underflow to 0!' : f.id === 'fp8' && Math.abs(presetVal) > f.maxFinite ? 'clamped at 448' : `1+${f.expBits}+${f.mantBits} bits`}
              </div>
            </div>
          );
        })}
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        <strong>{spec.name}:</strong> {spec.blurb} Notice why modern training runs prefer <strong>BF16 over FP16</strong>: stealing 3 mantissa bits for the exponent gives BF16 the full range of FP32 (no 65,504 overflow, no 10⁻⁹ underflow), while neural networks barely miss the extra decimal digits.
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Outlier catastrophe & group-wise rescue

export function OutlierQuantizer() {
  const [bits, setBits] = useState<4 | 8>(4);
  const [outlier, setOutlier] = useState(6.5);
  const [groupSize, setGroupSize] = useState(64);

  const weights = useMemo(() => {
    const w = [...BASE_WEIGHTS];
    w[0] = outlier;
    return w;
  }, [outlier]);

  const res = useMemo(() => quantizeIntSymmetric(weights, bits, groupSize), [weights, bits, groupSize]);
  const model70bGb = (70e9 * res.effBits) / 8 / 1e9;

  return (
    <Widget
      title="The outlier catastrophe — and how group-wise quantisation rescues INT4"
      wide
      caption={
        <>
          64 weights: 63 normal Gaussian weights in [−0.9, +0.9] plus one outlier weight <code>w[0]</code>. Watch per-tensor INT4 crush 52 of the 63 normal weights to exact zero — then switch group size to <strong>16</strong> or <strong>8</strong> to isolate the outlier to its own tiny block.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 10 }}>
        <Segmented
          label="bit width"
          options={[
            { value: 8 as const, label: 'INT8 (−127..+127)' },
            { value: 4 as const, label: 'INT4 (−7..+7)' },
          ]}
          value={bits}
          onChange={setBits}
        />
        <Segmented
          label="group size"
          options={[
            { value: 64, label: '64 (per-tensor)' },
            { value: 16, label: '16' },
            { value: 8, label: '8' },
          ]}
          value={groupSize}
          onChange={setGroupSize}
        />
      </div>

      <Slider
        label="outlier weight w[0]"
        value={outlier}
        min={0.5}
        max={8.0}
        step={0.5}
        onChange={setOutlier}
        format={(v) => v.toFixed(1)}
        color="c-b"
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8, margin: '10px 0' }}>
        {[
          { label: 'effective bits/w', val: `${res.effBits.toFixed(2)} b`, sub: `70B ≈ ${model70bGb.toFixed(1)} GB`, color: 'var(--accent-text)' },
          { label: 'overall RMSE', val: res.rmse.toFixed(4), sub: 'lower is better', color: res.rmse > 0.15 ? 'var(--c-bad)' : 'var(--c-c)' },
          { label: 'normal → zero', val: `${res.normalZeros} / 63`, sub: `${Math.round((res.normalZeros / 63) * 100)}% wiped out`, color: res.normalZeros > 20 ? 'var(--c-bad)' : 'var(--c-a)' },
          { label: 'scale factors', val: `${res.scales.length}`, sub: `1 FP16 per ${groupSize} w`, color: 'var(--c-b)' },
        ].map((card) => (
          <div
            key={card.label}
            style={{
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: '7px 10px',
            }}
          >
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{card.label}</div>
            <div className="mono" style={{ fontSize: 17, fontWeight: 700, color: card.color }}>
              {card.val}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>{card.sub}</div>
          </div>
        ))}
      </div>

      <div className="field-label">the 63 normal weights (blue dots = true weight, orange dashes = quantised ŵ, shaded = outlier’s group)</div>
      <Plot
        width={620}
        height={185}
        x={[1, 63]}
        y={[-1.1, 1.1]}
        pad={36}
        grid
        axes
        xLabel="weight index (1..63)"
        yLabel="value"
        ariaLabel="Original versus quantised values for the 63 normal weights"
        style={{ width: '100%', height: 'auto', display: 'block' }}
      >
        {({ x, y }) => (
          <g>
            {/* Shaded outlier group */}
            <rect
              x={x(1)}
              y={y(1.1)}
              width={x(Math.min(63, groupSize - 1)) - x(1)}
              height={y(-1.1) - y(1.1)}
              fill="var(--c-b)"
              opacity={0.09}
            />
            {/* Group boundaries */}
            {Array.from({ length: Math.floor(64 / groupSize) - 1 }, (_, gi) => {
              const boundary = (gi + 1) * groupSize;
              return (
                <line
                  key={boundary}
                  x1={x(boundary - 0.5)}
                  x2={x(boundary - 0.5)}
                  y1={y(-1.1)}
                  y2={y(1.1)}
                  stroke="var(--border-strong)"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
              );
            })}
            {/* Error lines + true & quantised markers */}
            {weights.slice(1).map((w, idx) => {
              const i = idx + 1;
              const q = res.reconstructed[i];
              const isZero = res.codes[i] === 0;
              return (
                <g key={i}>
                  <line x1={x(i)} x2={x(i)} y1={y(w)} y2={y(q)} stroke={isZero ? 'var(--c-bad)' : 'var(--text-faint)'} strokeWidth={1.1} opacity={0.7} />
                  <circle cx={x(i)} cy={y(w)} r={2.4} fill="var(--c-a)" />
                  <line x1={x(i) - 3} x2={x(i) + 3} y1={y(q)} y2={y(q)} stroke={isZero ? 'var(--c-bad)' : 'var(--c-b)'} strokeWidth={2.2} />
                </g>
              );
            })}
          </g>
        )}
      </Plot>

      <div className="live-note" style={{ marginTop: 10 }}>
        {bits === 4 && groupSize === 64 && outlier >= 5 ? (
          <>
            <strong>Outlier catastrophe!</strong> One weight at {outlier.toFixed(1)} stretched the single INT4 scale to {(outlier / 7).toFixed(2)}, collapsing <strong>{res.normalZeros} of 63</strong> normal weights straight onto exact zero (RMSE {res.rmse.toFixed(4)}). Switch group size to <strong>16</strong> or <strong>8</strong> to see group-wise quantisation rescue the other groups.
          </>
        ) : bits === 4 && groupSize < 64 && outlier >= 5 ? (
          <>
            <strong>Damage contained to the shaded group!</strong> Only the {groupSize - 1} neighbours sharing group 0 with <code>w[0]</code> suffer a coarse scale; the remaining {64 - groupSize} weights get tight per-group scales, slashing RMSE to <strong>{res.rmse.toFixed(4)}</strong> and zero-collapse to <strong>{res.normalZeros}/63</strong> — for only {(16 / groupSize).toFixed(2)} extra bits/weight.
          </>
        ) : bits === 8 ? (
          <>
            <strong>INT8 handles the outlier cleanly</strong> (RMSE {res.rmse.toFixed(4)}, {res.normalZeros}/63 zeros): with 255 integer levels (−127..+127) instead of 15, even a large scale leaves plenty of resolution for normal weights.
          </>
        ) : (
          <>
            <strong>No outlier (w[0] = {outlier.toFixed(1)}):</strong> even per-tensor INT4 stays well behaved (RMSE {res.rmse.toFixed(4)}, {res.normalZeros}/63 zeros). Slide <code>w[0]</code> up to 6.5 to see why real LLMs need group-wise quantisation.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Lab: Mixed-precision quantisation

const QUANT_LAB_KEY = 'nl-quant-lab-v1';

function loadBestRmse(): number | null {
  try {
    const raw = localStorage.getItem(QUANT_LAB_KEY);
    if (raw) return Number(raw);
  } catch {
    // ignore
  }
  return null;
}

export function QuantLab() {
  const [choices, setChoices] = useState<[number, number, number, number]>([2, 2, 2, 2]); // start at uniform INT4 tensor
  const [bestSaved, setBestSaved] = useState<number | null>(loadBestRmse);

  const solvedRef = useMemo(() => solveQuantLab(), []);
  const res = useMemo(() => evaluateMix(choices), [choices]);

  const bitsOk = res.avgBits <= LAB_BIT_BUDGET + 1e-9;
  const rmseOk = res.avgRmse <= LAB_RMSE_GOAL;
  const passes = bitsOk && rmseOk;
  const isOpt = passes && res.avgRmse <= solvedRef.best.avgRmse + 1e-5;

  useEffect(() => {
    if (passes) {
      setBestSaved((prev) => {
        const next = prev === null ? res.avgRmse : Math.min(prev, res.avgRmse);
        try {
          localStorage.setItem(QUANT_LAB_KEY, next.toFixed(5));
        } catch {
          // ignore
        }
        return next;
      });
    }
  }, [passes, res.avgRmse]);

  const setBlock = (bIdx: number, sIdx: number) => {
    setChoices((prev) => {
      const next = [...prev] as [number, number, number, number];
      next[bIdx] = sIdx;
      return next;
    });
  };

  return (
    <Widget
      title="Lab: Mixed-precision quantisation (the GGUF Q4_K trick)"
      wide
      caption={
        <>
          Four weight blocks with different outlier profiles. Pick a quantisation format for each block to stay within <strong>≤ {LAB_BIT_BUDGET} average bits/weight</strong> while driving average RMSE down to <strong>≤ {LAB_RMSE_GOAL}</strong>. Uniform INT4 fails — spend your extra bit budget where the outliers live.
        </>
      }
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 12,
          padding: '8px 12px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          marginBottom: 12,
          fontSize: 13,
        }}
      >
        <span style={{ color: bitsOk ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 650 }}>
          {bitsOk ? '✓' : '✕'} Avg bits: {res.avgBits.toFixed(3)} / {LAB_BIT_BUDGET} b
        </span>
        <span style={{ color: rmseOk ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 650 }}>
          {rmseOk ? '✓' : '✕'} Avg RMSE: {res.avgRmse.toFixed(4)} (goal ≤ {LAB_RMSE_GOAL})
        </span>
        {bestSaved !== null && (
          <span style={{ color: 'var(--text-muted)' }}>
            Your best: {bestSaved.toFixed(4)} · Optimum: {solvedRef.best.avgRmse.toFixed(4)}
          </span>
        )}
      </div>

      <div style={{ display: 'grid', gap: 10, marginBottom: 12 }}>
        {LAB_BLOCKS.map((blk, bi) => {
          const stat = res.perBlock[bi];
          return (
            <div
              key={blk.id}
              style={{
                padding: '9px 12px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--bg-sunken)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: 6, marginBottom: 6 }}>
                <div>
                  <strong style={{ fontSize: 14 }}>{blk.name}</strong>{' '}
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>— {blk.character}</span>
                </div>
                <div className="mono" style={{ fontSize: 12.5 }}>
                  <span style={{ color: 'var(--text-muted)' }}>{stat.effBits.toFixed(2)} b</span> · RMSE{' '}
                  <strong style={{ color: stat.rmse > 0.08 ? 'var(--c-bad)' : stat.rmse < 0.03 ? 'var(--c-good)' : 'var(--text)' }}>
                    {stat.rmse.toFixed(4)}
                  </strong>
                </div>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {LAB_SCHEMES.map((sc, si) => (
                  <button
                    key={sc.id}
                    type="button"
                    className={choices[bi] === si ? 'btn primary' : 'btn'}
                    style={{ fontSize: 12, padding: '4px 8px' }}
                    onClick={() => setBlock(bi, si)}
                  >
                    {sc.label}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button className="btn" onClick={() => setChoices([2, 2, 2, 2])}>
          Uniform INT4 tensor (4.25 b)
        </button>
        <button className="btn" onClick={() => setChoices([3, 3, 3, 3])}>
          Uniform INT4 g=16 (5.00 b)
        </button>
      </div>

      <LiveCode code={llamaCppLines(res.avgBits)} note="llama.cpp; the lab's INT4/INT8 blocks are simpler than K-quants, so this is the nearest real format" />

      <div className="live-note">
        {isOpt ? (
          <>
            <strong>Exhaustive optimum found ({res.avgRmse.toFixed(4)} RMSE at {res.avgBits.toFixed(3)} bits/w)!</strong> Out of all 625 mixed-precision assignments under {LAB_BIT_BUDGET} bits/w, only <strong>{solvedRef.passing}</strong> clear the 0.045 RMSE bar, and this is the exact minimum — keeping outlier-heavy <code>Attn QKV</code> in INT8 while leaving smooth FFN layers in INT4. That is exactly how real <code>GGUF Q4_K_M</code> quantisation works.
          </>
        ) : passes ? (
          <>
            <strong>Goal cleared ({res.avgRmse.toFixed(4)} RMSE at {res.avgBits.toFixed(3)} bits/w)!</strong> You found one of the {solvedRef.passing} passing recipes out of 625. Optional challenge: can you push RMSE all the way down to the exhaustive minimum of <strong>{solvedRef.best.avgRmse.toFixed(4)}</strong>?
          </>
        ) : !bitsOk ? (
          <>
            <strong>Over bit budget ({res.avgBits.toFixed(3)} &gt; {LAB_BIT_BUDGET} bits/w).</strong> Smooth FFN blocks barely lose accuracy in INT4 — save your extra bits for the outlier-heavy attention blocks.
          </>
        ) : (
          <>
            <strong>Within budget, but RMSE ({res.avgRmse.toFixed(4)}) is too high.</strong> Look at which single block is dominating the error (right now <code>Attn QKV</code> = {res.perBlock[0].rmse.toFixed(4)}) and upgrade it!
          </>
        )}
      </div>
    </Widget>
  );
}
