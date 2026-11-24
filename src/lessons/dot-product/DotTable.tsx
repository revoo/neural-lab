import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Tex, col, fmt, pn } from '../../components/Tex';

const LABELS = ['apples', 'bread', 'milk', 'coffee'];

export function DotTable() {
  const [q, setQ] = useState([2, 1, 3, 0]);
  const [p, setP] = useState([0.5, 2.2, 1.2, 4.5]);
  const [step, setStep] = useState(4); // how many loop iterations have run
  const products = q.map((v, i) => v * p[i]);
  const running = products.slice(0, step).reduce((s, v) => s + v, 0);
  const total = products.reduce((s, v) => s + v, 0);

  const set = (arr: number[], setter: (v: number[]) => void, i: number, v: string) => {
    const n = Number(v);
    if (Number.isNaN(n)) return;
    const next = arr.slice();
    next[i] = n;
    setter(next);
  };

  const terms = q.map((v, i) => {
    const t = `${col('c-a', pn(v, 1))} \\times ${col('c-b', pn(p[i], 2))}`;
    return i < step ? t : `\\htmlStyle{opacity:0.3}{${t}}`;
  });

  return (
    <Widget
      title="A dot product is a shopping receipt"
      wide
      caption={
        <>
          Edit any quantity or price. Each row multiplies a <span className="c-a">quantity</span> by a <span className="c-b">price</span>; the
          total adds up the rows. Use <strong>Step</strong> to run the loop one iteration at a time, exactly as the code below would. Try giving
          an item a <em>negative</em> price — a coupon — and watch it pull the total down.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <table className="num-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>i</th>
                <th style={{ textAlign: 'left' }}>item</th>
                <th className="c-a">quantity a[i]</th>
                <th className="c-b">price b[i]</th>
                <th>a[i] × b[i]</th>
              </tr>
            </thead>
            <tbody>
              {q.map((v, i) => (
                <tr key={i} style={{ opacity: i < step ? 1 : 0.4, background: i === step - 1 ? 'var(--accent-soft)' : undefined }}>
                  <td style={{ textAlign: 'left' }}>{i}</td>
                  <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)' }}>{LABELS[i]}</td>
                  <td>
                    <input className="num-input" type="number" step={1} value={v} onChange={(e) => set(q, setQ, i, e.target.value)} aria-label={`quantity of ${LABELS[i]}`} />
                  </td>
                  <td>
                    <input className="num-input" type="number" step={0.1} value={p[i]} onChange={(e) => set(p, setP, i, e.target.value)} aria-label={`price of ${LABELS[i]}`} />
                  </td>
                  <td>{i < step ? products[i].toFixed(2) : '·'}</td>
                </tr>
              ))}
              <tr>
                <td colSpan={4} style={{ textAlign: 'right', fontFamily: 'var(--font-ui)', fontWeight: 600 }}>
                  {step < 4 ? `running total after ${step} step${step === 1 ? '' : 's'}` : 'total = a · b'}
                </td>
                <td className="c-c" style={{ fontWeight: 700 }}>{running.toFixed(2)}</td>
              </tr>
            </tbody>
          </table>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <button className="btn small" onClick={() => setStep(0)}>⟲ Reset loop</button>
            <button className="btn small primary" disabled={step >= 4} onClick={() => setStep(step + 1)}>Step →</button>
            <button className="btn small" disabled={step >= 4} onClick={() => setStep(4)}>Run all</button>
          </div>
        </div>
        <div className="widget-col">
          <pre className="code" style={{ margin: 0 }}>
            {[
              'def dot(a, b):',
              '    total = 0',
              '    for i in range(len(a)):',
              '        total += a[i] * b[i]',
              '    return total',
            ].map((line, k) => (
              <div key={k} style={{ background: k === 3 && step > 0 && step < 4 ? 'var(--accent-soft)' : undefined }}>
                {line}
              </div>
            ))}
          </pre>
          <div style={{ marginTop: 12, fontSize: 15 }}>
            <Tex block>{`\\mathbf{a}\\cdot\\mathbf{b} = ${terms.join(' + ')}`}</Tex>
            <Tex block>{`= ${col('c-c', fmt(step === 4 ? total : running, 2))}`}</Tex>
          </div>
        </div>
      </div>
    </Widget>
  );
}
