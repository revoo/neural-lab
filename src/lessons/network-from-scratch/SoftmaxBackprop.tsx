// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import {
  CLASS_NAMES,
  SOFTMAX_PRESETS,
  softmaxBackprop,
  stepSoftmaxLogits,
  type SoftmaxPreset,
} from './softmax';

const fmtS = (v: number, d = 4) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;

export function SoftmaxBackprop() {
  const [presetId, setPresetId] = useState<SoftmaxPreset['id']>('predict-case');
  const [logits, setLogits] = useState<[number, number, number]>([...SOFTMAX_PRESETS[0].logits]);
  const [target, setTarget] = useState<0 | 1 | 2>(SOFTMAX_PRESETS[0].targetClass);
  const [steps, setSteps] = useState(0);

  const res = useMemo(() => softmaxBackprop(logits, target), [logits, target]);

  const applyPreset = (id: SoftmaxPreset['id']) => {
    const p = SOFTMAX_PRESETS.find((x) => x.id === id) ?? SOFTMAX_PRESETS[0];
    setPresetId(p.id);
    setLogits([...p.logits]);
    setTarget(p.targetClass);
    setSteps(0);
  };

  const setLogit = (idx: 0 | 1 | 2, val: number) => {
    setLogits((cur) => {
      const nxt: [number, number, number] = [...cur];
      nxt[idx] = val;
      return nxt;
    });
  };

  const takeStep = () => {
    const nxt = stepSoftmaxLogits(logits, target, 1.0);
    setLogits([
      Math.max(-3, Math.min(4, Number(nxt[0].toFixed(3)))),
      Math.max(-3, Math.min(4, Number(nxt[1].toFixed(3)))),
      Math.max(-3, Math.min(4, Number(nxt[2].toFixed(3)))),
    ]);
    setSteps((s) => s + 1);
  };

  const reset = () => applyPreset(presetId);

  const clsColors = ['var(--c-a)', 'var(--c-b)', 'var(--c-c)'] as const;

  return (
    <Widget
      title="Softmax + cross-entropy: why the gradient is still p − y"
      wide
      caption={
        <>
          Drag the three raw scores (logits <span className="mono">z₀, z₁, z₂</span>) or switch the true class. The
          right-hand table compares the analytic shortcut <span className="mono">p − y</span> against a live numerical
          central finite difference — notice they match to <span className="mono">&lt; 10⁻⁹</span> and always sum to
          zero. Press <strong>Take 1 logit step</strong> to see the true logit rise and every wrong logit fall.
        </>
      }
    >
      {/* Presets + true class + step */}
      <div className="controls inline" style={{ gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <Segmented
          options={SOFTMAX_PRESETS.map((p) => ({ value: p.id, label: p.label }))}
          value={presetId}
          onChange={applyPreset}
        />
        <Segmented
          options={([0, 1, 2] as const).map((c) => ({ value: String(c), label: `True: ${CLASS_NAMES[c]}` }))}
          value={String(target)}
          onChange={(v) => setTarget(Number(v) as 0 | 1 | 2)}
        />
        <div className="btn-row" style={{ marginLeft: 'auto' }}>
          <button type="button" className="btn primary small" onClick={takeStep}>
            Take 1 logit step (η=1)
          </button>
          <button type="button" className="btn small" onClick={reset}>
            Reset
          </button>
        </div>
      </div>

      {/* Main split */}
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap', marginTop: 12 }}>
        {/* Left: Logit sliders + Softmax probability bars */}
        <div style={{ flex: '1 1 280px', minWidth: 250, display: 'flex', flexDirection: 'column', gap: 10 }}>
          {([0, 1, 2] as const).map((i) => (
            <Slider
              key={i}
              label={`z${i} (${CLASS_NAMES[i]}${i === target ? ' · true' : ''})`}
              value={logits[i]}
              min={-3}
              max={4}
              step={0.05}
              onChange={(v) => setLogit(i, v)}
              format={(v) => fmtS(v, 2)}
            />
          ))}

          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              marginTop: 2,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 8 }}>
              <strong>Softmax probabilities p</strong>
              <span className="mono" style={{ color: 'var(--text-muted)' }}>
                L = −ln p<sub>true</sub> = <strong style={{ color: 'var(--text)' }}>{res.loss.toFixed(4)}</strong>
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {([0, 1, 2] as const).map((i) => (
                <div key={i} style={{ display: 'grid', gridTemplateColumns: '76px 1fr 56px', gap: 8, alignItems: 'center', fontSize: 12 }}>
                  <span className="mono" style={{ fontWeight: i === target ? 700 : 400, whiteSpace: 'nowrap' }}>
                    {CLASS_NAMES[i]}
                    {i === target ? ' ★' : ''}
                  </span>
                  <div style={{ height: 14, background: 'var(--bg-elev)', borderRadius: 4, overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${Math.max(2, res.probs[i] * 100)}%`,
                        height: '100%',
                        background: clsColors[i],
                      }}
                    />
                  </div>
                  <span className="mono" style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                    {(res.probs[i] * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Analytic (p - y) vs live central finite difference table */}
        <div
          style={{
            flex: '1 1 320px',
            minWidth: 260,
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-sunken)',
            border: '1px solid var(--border)',
            overflowX: 'auto',
          }}
        >
          <div style={{ fontSize: 12.5, fontWeight: 700, marginBottom: 8 }}>
            Logit gradients: analytic <span className="mono">p − y</span> vs numerical check
          </div>
          <table className="num-table" style={{ width: '100%', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '5px 4px' }}>Class</th>
                <th style={{ textAlign: 'right', textTransform: 'none', padding: '5px 4px' }}>p</th>
                <th style={{ textAlign: 'right', textTransform: 'none', padding: '5px 4px' }}>y</th>
                <th style={{ textAlign: 'right', color: 'var(--c-grad)', textTransform: 'none', padding: '5px 4px' }}>p − y</th>
                <th style={{ textAlign: 'right', padding: '5px 4px' }}>FD</th>
                <th style={{ textAlign: 'center', padding: '5px 4px' }}>step</th>
              </tr>
            </thead>
            <tbody>
              {([0, 1, 2] as const).map((i) => {
                const g = res.analyticGrad[i];
                const n = res.numericGrad[i];
                return (
                  <tr
                    key={i}
                    style={i === target ? { background: 'color-mix(in srgb, var(--c-c) 12%, transparent)' } : undefined}
                  >
                    <td className="mono" style={{ textAlign: 'left', fontWeight: i === target ? 700 : 400, whiteSpace: 'nowrap', padding: '5px 4px' }}>
                      {CLASS_NAMES[i]}
                    </td>
                    <td className="mono" style={{ textAlign: 'right', whiteSpace: 'nowrap', padding: '5px 4px' }}>
                      {res.probs[i].toFixed(3)}
                    </td>
                    <td className="mono" style={{ textAlign: 'right', whiteSpace: 'nowrap', padding: '5px 4px' }}>
                      {res.oneHot[i]}
                    </td>
                    <td className="mono" style={{ textAlign: 'right', color: 'var(--c-grad)', fontWeight: 700, whiteSpace: 'nowrap', padding: '5px 4px' }}>
                      {fmtS(g, 3)}
                    </td>
                    <td className="mono" style={{ textAlign: 'right', whiteSpace: 'nowrap', padding: '5px 4px' }}>
                      {fmtS(n, 3)}
                    </td>
                    <td
                      className="mono"
                      style={{
                        textAlign: 'center',
                        fontWeight: 700,
                        color: g < 0 ? 'var(--c-a)' : 'var(--c-b)',
                        whiteSpace: 'nowrap',
                        padding: '5px 4px',
                      }}
                    >
                      {g < 0 ? '↑' : '↓'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div
            className="mono"
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 8,
              marginTop: 10,
              paddingTop: 8,
              borderTop: '1px solid var(--border)',
              fontSize: 11.5,
              color: 'var(--text-muted)',
            }}
          >
            <span>
              Σ(p − y) = <strong style={{ color: 'var(--text)' }}>{Math.abs(res.gradSum).toFixed(6)}</strong>
            </span>
            <span>
              max |diff| = <strong style={{ color: 'var(--c-good)' }}>{res.maxDiff.toExponential(1)}</strong> ✓
            </span>
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 12 }}>
        <strong>{steps > 0 ? `After ${steps} logit step${steps === 1 ? '' : 's'}: ` : ''}</strong>
        The true class ({CLASS_NAMES[target]}, <span className="mono">p = {res.probs[target].toFixed(3)}</span>) gets a{' '}
        <strong>negative</strong> gradient <span className="mono">{fmtS(res.analyticGrad[target], 3)}</span>, so
        gradient descent pushes its logit <strong>up</strong>. Every wrong class gets a <strong>positive</strong>{' '}
        gradient equal to its own probability — pushing each wrong logit <strong>down</strong> in exact proportion to
        how much confidence it stole, with the pulls always balancing to <span className="mono">0</span>.
      </div>
    </Widget>
  );
}
