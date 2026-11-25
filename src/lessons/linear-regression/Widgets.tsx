// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import {
  LAB_BRIEFS,
  LAB_FEATURES,
  LAB_LAMBDAS,
  LAB_THRESHOLDS,
  briefPasses,
  collinearExperiment,
  confusionAt,
  evaluateLab,
  getFraudDataset,
  getLabData,
  labExhaustive,
  linearVsLogistic1D,
  sigmoid,
  type FeatureKey,
  type LabBrief,
} from './regression';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

// ------------------------------------------------------------------ 1. Collinearity & Normal Equations vs GD

const RHOS = [0, 0.7, 0.9, 0.99, 0.999] as const;

export function NormalEqWidget() {
  const [rho, setRho] = useState<number>(0.99);
  const [lambda, setLambda] = useState<number>(0.5);

  const res = useMemo(() => collinearExperiment(rho, lambda), [rho, lambda]);

  const maxW = Math.max(2.5, Math.abs(res.wOls[0]), Math.abs(res.wOls[1]), Math.abs(res.wRidge[0]), Math.abs(res.wRidge[1]));

  const BarRow = ({ label, w, color }: { label: string; w: [number, number]; color: string }) => (
    <div style={{ display: 'grid', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5 }}>
        <span style={{ fontWeight: 600 }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>
          w₁ = {w[0].toFixed(2)}, w₂ = {w[1].toFixed(2)} (truth: 1.50, 1.50)
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[0, 1].map((i) => {
          const v = w[i];
          const frac = Math.min(1, Math.abs(v) / maxW);
          return (
            <div key={i} style={{ display: 'grid', gap: 2 }}>
              <div style={{ height: 14, borderRadius: 4, background: 'var(--bg-sunken)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ width: `${frac * 100}%`, height: '100%', background: color, opacity: 0.85 }} />
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>w{i + 1} = {v.toFixed(2)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <Widget
      title="Normal equations vs gradient descent (and the collinearity trap)"
      wide
      caption={
        <>
          Two features predicting <code>y = 1.5x₁ + 1.5x₂ + noise</code> (40 training points, 200 shifted test points). Increase feature correlation from <code>0</code> to <code>0.999</code>: watch the condition number of <code>XᵀX</code> jump from 1.6 to 3,091, gradient descent slow from 4 steps to 500+, and unregularised OLS split into unstable weights that Ridge snaps straight back to truth.
        </>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 10 }}>
          <Segmented
            label="Feature correlation"
            value={rho}
            onChange={setRho}
            options={RHOS.map((r) => ({ value: r, label: String(r) }))}
          />
          <Slider label="Ridge penalty λ" value={lambda} min={0} max={2} step={0.1} onChange={setLambda} format={(v) => v.toFixed(1)} color="c-c" />
          <div className="readout">
            <span className="k">condition number κ(XᵀX) </span>
            <span style={{ color: res.condNumber > 100 ? 'var(--c-bad)' : 'var(--c-good)', fontWeight: 700 }}>{res.condNumber.toFixed(1)}</span>
            {'\n'}
            <span className="k">normal equations solves in </span>
            <strong>1 matrix solve</strong>
            {'\n'}
            <span className="k">batch GD steps (optimal lr = 1/L) </span>
            <strong>{res.gdStepsToTol >= 500 ? '500+ (not converged)' : `${res.gdStepsToTol} steps`}</strong>
            {'\n'}
            <span className="k">OLS test MSE (λ = 0) </span>
            <span style={{ color: res.testMseOls > 0.28 ? 'var(--c-bad)' : 'var(--text)' }}>{res.testMseOls.toFixed(3)}</span>
            {'\n'}
            <span className="k">Ridge test MSE (λ = {lambda.toFixed(1)}) </span>
            <span style={{ color: 'var(--c-good)', fontWeight: 650 }}>{res.testMseRidge.toFixed(3)}</span>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 14 }}>
          <BarRow label="Exact OLS: (XᵀX)⁻¹Xᵀy" w={res.wOls} color="var(--c-b)" />
          <BarRow label={`Exact Ridge: (XᵀX + ${lambda.toFixed(1)}I)⁻¹Xᵀy`} w={res.wRidge} color="var(--c-c)" />
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            Notice how OLS still gets low <em>training</em> MSE ({res.trainMseOls.toFixed(3)}) even at correlation 0.999 — because two opposing errors cancel when x₁ ≈ x₂. The moment the correlation loosens slightly on test data, those split weights hurt test MSE ({res.testMseOls.toFixed(3)} vs Ridge’s {res.testMseRidge.toFixed(3)}).
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {rho <= 0.7
          ? `Low correlation (${rho}): the loss bowl is nearly round (κ = ${res.condNumber.toFixed(1)}), so both OLS and batch gradient descent (${res.gdStepsToTol} steps) recover weights right near (1.50, 1.50).`
          : rho === 0.99
            ? `At correlation 0.99, the bowl stretches into a 303:1 canyon: batch GD takes ${res.gdStepsToTol} steps even with its theoretically optimal step size, while the normal equations jump straight to the exact floor in one solve.`
            : `At correlation 0.999 (κ = ${res.condNumber.toFixed(0)}), XᵀX is nearly singular: OLS splits into (${res.wOls[0].toFixed(2)}, ${res.wOls[1].toFixed(2)}) and test MSE jumps to ${res.testMseOls.toFixed(3)}. Adding Ridge λ = ${lambda.toFixed(1)} lifts the tiny eigenvalue and restores (${res.wRidge[0].toFixed(2)}, ${res.wRidge[1].toFixed(2)}) with test MSE ${res.testMseRidge.toFixed(3)}.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Why Linear Regression Breaks on Classification

export function LinearVsLogisticWidget() {
  const [outlierX, setOutlierX] = useState(7);
  const res = useMemo(() => linearVsLogistic1D(outlierX), [outlierX]);

  const xMax = Math.max(10, outlierX + 1.5);

  return (
    <Widget
      title="Why linear regression breaks on classification"
      wide
      caption={
        <>
          Six negative points (orange, y = 0) and six positive points (blue, y = 1). Drag the <strong>rightmost positive point</strong> from <code>x = 7</code> out to <code>x = 25</code> — an even easier, more obvious positive! Watch OLS linear regression tilt downward to avoid squared error above 1, dragging its 0.5 boundary right through real positives, while logistic regression never budges.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        {[7, 15, 25].map((x) => (
          <button key={x} type="button" className={`btn small${outlierX === x ? ' primary' : ''}`} onClick={() => setOutlierX(x)}>
            Easy positive at x = {x}
          </button>
        ))}
      </div>

      <Slider label="position of rightmost positive example" value={outlierX} min={7} max={25} step={0.5} onChange={setOutlierX} format={(v) => `x = ${v.toFixed(1)}`} color="c-a" />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, marginTop: 10, alignItems: 'start' }}>
        <div>
          <Plot width={480} height={260} x={[0, xMax]} y={[-0.25, 1.35]} pad={34} xLabel="feature x" yLabel="predicted score / P(y=1)" ariaLabel="OLS linear regression vs Logistic regression with a far-right positive example">
            {({ x, y }) => {
              const linPath = `M${x(0).toFixed(1)},${y(res.wLin[0]).toFixed(1)} L${x(xMax).toFixed(1)},${y(res.wLin[0] + res.wLin[1] * xMax).toFixed(1)}`;
              const logPts: string[] = [];
              for (let i = 0; i <= 80; i++) {
                const xv = (i / 80) * xMax;
                const pv = sigmoid(res.wLog[0] + res.wLog[1] * xv);
                logPts.push(`${i === 0 ? 'M' : 'L'}${x(xv).toFixed(1)},${y(pv).toFixed(1)}`);
              }
              return (
                <>
                  {/* 0.5 threshold line */}
                  <line x1={x(0)} x2={x(xMax)} y1={y(0.5)} y2={y(0.5)} stroke="var(--text-muted)" strokeWidth={1} strokeDasharray="4 4" opacity={0.6} />
                  {/* Vertical boundary markers */}
                  <line x1={x(res.boundLin)} x2={x(res.boundLin)} y1={y(-0.2)} y2={y(1.25)} stroke="var(--c-b)" strokeWidth={1.5} strokeDasharray="3 3" />
                  <line x1={x(res.boundLog)} x2={x(res.boundLog)} y1={y(-0.2)} y2={y(1.25)} stroke="var(--c-a)" strokeWidth={1.5} strokeDasharray="3 3" />
                  {/* Curves */}
                  <path d={linPath} fill="none" stroke="var(--c-b)" strokeWidth={2.4} />
                  <path d={logPts.join(' ')} fill="none" stroke="var(--c-a)" strokeWidth={2.6} />
                  {/* Data points */}
                  {res.pts.map((pt, idx) => {
                    const isDragged = idx === res.pts.length - 1;
                    return (
                      <circle
                        key={idx}
                        cx={x(pt.x)}
                        cy={y(pt.y)}
                        r={isDragged ? 6.5 : 4.8}
                        fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                        stroke="var(--bg)"
                        strokeWidth={isDragged ? 2 : 1.2}
                      />
                    );
                  })}
                </>
              );
            }}
          </Plot>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            <span>
              <span style={{ color: 'var(--c-b)', fontWeight: 700 }}>━</span> OLS line (cut x = {res.boundLin.toFixed(2)})
            </span>
            <span>
              <span style={{ color: 'var(--c-a)', fontWeight: 700 }}>━</span> Logistic sigmoid (cut x = {res.boundLog.toFixed(2)})
            </span>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">OLS decision boundary </span>
            <span style={{ color: res.linErrors > 0 ? 'var(--c-bad)' : 'var(--c-good)', fontWeight: 700 }}>x = {res.boundLin.toFixed(2)}</span>
            {'\n'}
            <span className="k">OLS mistakes </span>
            <span style={{ color: res.linErrors > 0 ? 'var(--c-bad)' : 'var(--c-good)', fontWeight: 700 }}>
              {res.linErrors}/12 ({pct1(res.linAcc)} accuracy)
            </span>
            {'\n'}
            <span className="k">Logistic decision boundary </span>
            <span style={{ color: 'var(--c-good)', fontWeight: 700 }}>x = {res.boundLog.toFixed(2)}</span>
            {'\n'}
            <span className="k">Logistic mistakes </span>
            <span style={{ color: 'var(--c-good)', fontWeight: 700 }}>
              {res.logErrors}/12 ({pct1(res.logAcc)} accuracy)
            </span>
          </div>
          <div className="live-note">
            {res.linErrors === 0
              ? `At x = ${outlierX.toFixed(1)}, both classifiers separate the 12 points cleanly (boundaries ${res.boundLin.toFixed(2)} and ${res.boundLog.toFixed(2)}). Now drag the rightmost positive point out to x = 25.`
              : `Because OLS punishes (ŷ − 1)² even when ŷ > 1, pulling one easy positive to x = ${outlierX.toFixed(1)} tilted the line flat and dragged its cut-off to x = ${res.boundLin.toFixed(2)}, misclassifying ${res.linErrors} real positive${res.linErrors === 1 ? '' : 's'}! Logistic regression’s sigmoid saturates at 1 with near-zero gradient, keeping its boundary pinned at x = ${res.boundLog.toFixed(2)}.`}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Thresholds, Precision, Recall & ROC

export function ThresholdRocWidget() {
  const [thresh, setThresh] = useState(0.5);
  const fraud = useMemo(() => getFraudDataset(), []);
  const c = useMemo(() => confusionAt(fraud.probs, fraud.labels, thresh), [fraud, thresh]);

  return (
    <Widget
      title="Thresholds, precision, recall and the ROC curve (5% fraud)"
      wide
      caption={
        <>
          200 held-out transactions (190 legitimate, 10 fraudulent) scored by one trained logistic regression model (ROC AUC = {fraud.auc.toFixed(3)}). Moving the threshold never retrains the model — it moves your operating point along the ROC curve. Compare default <code>t = 0.50</code> (97.0% accuracy, €505 loss) with <code>t = 0.10</code> (91.5% accuracy, €275 loss).
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        {[0.5, 0.2, 0.1, 0.05].map((t) => (
          <button key={t} type="button" className={`btn small${Math.abs(thresh - t) < 0.005 ? ' primary' : ''}`} onClick={() => setThresh(t)}>
            t = {t.toFixed(2)}
          </button>
        ))}
      </div>

      <Slider label="decision threshold t (flag fraud if P ≥ t)" value={thresh} min={0.02} max={0.95} step={0.01} onChange={setThresh} format={(v) => v.toFixed(2)} color="c-a" />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, marginTop: 10, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 10 }}>
          <div style={{ overflowX: 'auto' }}>
            <table className="num-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Confusion Matrix</th>
                  <th>Actual Fraud</th>
                  <th>Actual Legit</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: 600 }}>Flagged (P ≥ t)</td>
                  <td style={{ color: 'var(--c-good)', fontWeight: 700 }}>{c.tp} (TP)</td>
                  <td style={{ color: 'var(--c-b)', fontWeight: 700 }}>{c.fp} (FP)</td>
                </tr>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: 600 }}>Cleared (P &lt; t)</td>
                  <td style={{ color: 'var(--c-bad)', fontWeight: 700 }}>{c.fn} (FN)</td>
                  <td style={{ color: 'var(--text-muted)' }}>{c.tn} (TN)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="readout">
            <span className="k">accuracy (tricky on 5% fraud!) </span>
            {pct1(c.accuracy)}
            {'\n'}
            <span className="k">precision (TP / flagged) </span>
            {pct1(c.precision)} ({c.tp}/{c.tp + c.fp})
            {'\n'}
            <span className="k">recall / TPR (TP / actual fraud) </span>
            <span style={{ color: c.recall >= 0.8 ? 'var(--c-good)' : 'var(--c-b)', fontWeight: 700 }}>{pct1(c.recall)}</span> ({c.tp}/10)
            {'\n'}
            <span className="k">F1 score </span>
            {pct1(c.f1)}
            {'\n'}
            <span className="k">business cost (€100·FN + €5·FP) </span>
            <strong>€{c.costEur}</strong>
          </div>
        </div>

        <div>
          <Plot width={380} height={270} x={[0, 1]} y={[0, 1]} pad={34} xLabel="False Positive Rate (FPR)" yLabel="True Positive Rate (Recall)" ariaLabel="ROC curve for fraud logistic regression">
            {({ x, y }) => {
              const path = fraud.roc.map((pt, i) => `${i === 0 ? 'M' : 'L'}${x(pt.fpr).toFixed(1)},${y(pt.tpr).toFixed(1)}`).join(' ');
              const area = `${path} L${x(1).toFixed(1)},${y(0).toFixed(1)} L${x(0).toFixed(1)},${y(0).toFixed(1)} Z`;
              return (
                <>
                  {/* Chance diagonal */}
                  <line x1={x(0)} x2={x(1)} y1={y(0)} y2={y(1)} stroke="var(--text-muted)" strokeWidth={1.1} strokeDasharray="4 4" opacity={0.55} />
                  {/* AUC shading */}
                  <path d={area} fill="var(--c-a)" opacity={0.12} />
                  <path d={path} fill="none" stroke="var(--c-a)" strokeWidth={2.5} />
                  {/* Current operating point */}
                  <circle cx={x(c.fpr)} cy={y(c.recall)} r={6} fill="var(--c-b)" stroke="var(--bg)" strokeWidth={2} />
                  <text x={x(0.96)} y={y(0.12)} textAnchor="end" fontSize={11.5} fontWeight={650} fill="var(--c-a)">
                    AUC = {fraud.auc.toFixed(3)}
                  </text>
                </>
              );
            }}
          </Plot>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {thresh >= 0.45
          ? `At t = ${thresh.toFixed(2)}, headline accuracy is ${pct1(c.accuracy)}, yet the model misses ${c.fn} of 10 frauds (recall ${pct1(c.recall)}), costing €${c.costEur}. Try lowering the threshold to 0.10.`
          : thresh <= 0.15 && thresh >= 0.08
            ? `At t = ${thresh.toFixed(2)}, accuracy dropped to ${pct1(c.accuracy)}, yet catching ${c.tp}/10 frauds cuts business cost down to €${c.costEur} — nearly half the cost of the default 0.50 threshold!`
            : `Operating at FPR = ${pct1(c.fpr)}, Recall = ${pct1(c.recall)}, Precision = ${pct1(c.precision)}, business cost = €${c.costEur}. The ROC AUC (${fraud.auc.toFixed(3)}) measures ranking quality across all thresholds at once.`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Feature Engineering + Logistic Regression

const LAB_KEY = 'nl-logreg-lab-v1';

type LabEntry = {
  keys: FeatureKey[];
  lambda: number;
  thresh: number;
  solved: boolean;
  bestParams: number | null;
};

type LabState = Record<LabBrief['id'], LabEntry>;

const START: LabEntry = { keys: ['x1', 'x2'], lambda: 0.1, thresh: 0.5, solved: false, bestParams: null };

function loadLab(): LabState {
  const fresh: LabState = {
    ring: { ...START, keys: ['x1', 'x2'] },
    'xor-risk': { ...START, keys: ['x1', 'x2'] },
    imbalanced: { ...START, keys: ['x1', 'x2'] },
  };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return fresh;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    const validKeys = new Set(LAB_FEATURES.map((f) => f.key));
    for (const b of LAB_BRIEFS) {
      const e = parsed[b.id];
      if (!e || !Array.isArray(e.keys) || e.keys.length === 0 || !e.keys.every((k) => validKeys.has(k))) continue;
      if (!(LAB_LAMBDAS as readonly number[]).includes(e.lambda) || !(LAB_THRESHOLDS as readonly number[]).includes(e.thresh)) continue;
      fresh[b.id] = {
        keys: [...e.keys],
        lambda: e.lambda,
        thresh: b.fixedThresh ?? e.thresh,
        solved: !!e.solved,
        bestParams: typeof e.bestParams === 'number' ? e.bestParams : null,
      };
    }
  } catch {
    // ignore storage errors
  }
  return fresh;
}

export function ClassicalLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [showOpt, setShowOpt] = useState(false);

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const thresh = brief.fixedThresh ?? entry.thresh;
  const data = useMemo(() => getLabData(brief.id), [brief.id]);
  const res = useMemo(() => evaluateLab(brief.id, entry.keys, entry.lambda, thresh), [brief.id, entry.keys, entry.lambda, thresh]);
  const pass = briefPasses(brief, res);
  const ex = useMemo(() => (showOpt ? labExhaustive(brief) : null), [showOpt, brief]);

  // Memoised heatmap function (recomputes only when weights/keys/thresh change)
  const heatFn = useMemo(() => {
    // Shift probability so the visual 0.5 boundary aligns with `thresh`
    return (x1: number, x2: number) => {
      const p = res.predictXY(x1, x2);
      if (thresh === 0.5) return p;
      return p >= thresh ? 0.5 + 0.5 * ((p - thresh) / Math.max(1e-6, 1 - thresh)) : 0.5 * (p / Math.max(1e-6, thresh));
    };
  }, [res, thresh]);

  const saved = useRef(lab);
  saved.current = lab;
  const persist = (next: LabState) => {
    setLab(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };
  const update = (patch: Partial<LabEntry>) => persist({ ...lab, [brief.id]: { ...entry, ...patch } });

  const toggleKey = (k: FeatureKey) => {
    const has = entry.keys.includes(k);
    if (has && entry.keys.length === 1) return; // keep at least one feature
    const next = has ? entry.keys.filter((x) => x !== k) : [...entry.keys, k];
    update({ keys: next });
  };

  useEffect(() => {
    if (!pass) return;
    const cur = saved.current[brief.id];
    if (cur.solved && cur.bestParams !== null && cur.bestParams <= res.params) return;
    persist({
      ...saved.current,
      [brief.id]: {
        ...cur,
        solved: true,
        bestParams: cur.bestParams === null ? res.params : Math.min(cur.bestParams, res.params),
      },
    });
  }, [pass, brief.id, res.params]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Widget
      title="Lab: Feature Engineering + Logistic Regression"
      wide
      caption={
        <>
          Toggle engineered features, pick L2 regularisation λ (and decision threshold in Brief 3). Exact Newton–Raphson logistic regression trains live on 220 training points and is scored on 250 held-out test points (plotted). Meet the brief; optionally, use the fewest parameters (1 intercept + number of active features).
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setShowOpt(false);
        }}
        options={LAB_BRIEFS.map((b, i) => ({
          value: i,
          label: `${lab[b.id].solved ? '✓ ' : ''}${b.title.split(' · ')[0]}`,
        }))}
      />

      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 10 }}>
          <div>
            <div className="field-label">Active features ({res.params} params incl. intercept)</div>
            <div className="btn-row" style={{ marginTop: 4 }}>
              {LAB_FEATURES.map((f) => {
                const on = entry.keys.includes(f.key);
                return (
                  <button key={f.key} type="button" className={`btn small${on ? ' primary' : ''}`} onClick={() => toggleKey(f.key)}>
                    {on ? '✓ ' : ''}
                    {f.label}
                  </button>
                );
              })}
            </div>
          </div>

          <Segmented
            label="L2 penalty"
            value={entry.lambda}
            onChange={(lambda) => update({ lambda })}
            options={LAB_LAMBDAS.map((l) => ({ value: l as number, label: String(l) }))}
          />

          {brief.fixedThresh === undefined && (
            <Segmented
              label="Decision threshold"
              value={entry.thresh}
              onChange={(t) => update({ thresh: t })}
              options={LAB_THRESHOLDS.map((t) => ({ value: t as number, label: t.toFixed(2) }))}
            />
          )}

          <div className="btn-row">
            <span className={`status ${pass ? 'good' : 'bad'}`}>{pass ? 'Meets the brief' : 'Not yet'}</span>
            {entry.solved && (
              <span className="status good">Solved{entry.bestParams !== null ? ` · fewest params: ${entry.bestParams}` : ''}</span>
            )}
          </div>

          <div className="readout">
            <span className="k">parameters (1 + features) </span>
            <strong>{res.params}</strong>
            {'\n'}
            {brief.minAcc !== undefined && (
              <>
                <span style={{ color: res.testAcc >= brief.minAcc - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {res.testAcc >= brief.minAcc - 1e-9 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">held-out accuracy </span>
                {pct1(res.testAcc)} (need ≥ {pct1(brief.minAcc)})
                {'\n'}
              </>
            )}
            {brief.minRecall !== undefined && (
              <>
                <span style={{ color: res.testRecall >= brief.minRecall - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {res.testRecall >= brief.minRecall - 1e-9 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">held-out recall </span>
                {pct1(res.testRecall)} (need ≥ {pct1(brief.minRecall)})
                {'\n'}
              </>
            )}
            {brief.minPrecision !== undefined && (
              <>
                <span style={{ color: res.testPrecision >= brief.minPrecision - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                  {res.testPrecision >= brief.minPrecision - 1e-9 ? '✓' : '✗'}
                </span>{' '}
                <span className="k">held-out precision </span>
                {pct1(res.testPrecision)} (need ≥ {pct1(brief.minPrecision)})
                {'\n'}
              </>
            )}
            <span className="k">train accuracy </span>
            {pct1(res.trainAcc)}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <Plot width={320} height={320} x={[-3.1, 3.1]} y={[-3.1, 3.1]} pad={28} xLabel="x₁" yLabel="x₂" ariaLabel="Learned logistic regression decision surface on held-out points">
            <Heatmap f={heatFn} range={[0, 1]} colormap={diverging} resolution={140} opacity={0.55} />
            {({ x, y }) => (
              <>
                {data.test
                  .filter((pt) => Math.abs(pt.x1) <= 3.05 && Math.abs(pt.x2) <= 3.05)
                  .map((pt, i) => (
                    <circle
                      key={i}
                      cx={x(pt.x1)}
                      cy={y(pt.x2)}
                      r={2.8}
                      fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                      stroke="var(--bg)"
                      strokeWidth={0.6}
                      opacity={0.9}
                    />
                  ))}
              </>
            )}
          </Plot>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {pass
          ? `It passes with ${res.params} parameters! Optional: can you drop an unnecessary feature (even x₁ or x₂!) and still meet the brief?`
          : brief.id === 'ring'
            ? 'A straight line can’t enclose a circle. Notice which engineered features turn x₁² + x₂² = r² into a linear equation — and check whether you even need raw x₁ and x₂ afterwards!'
            : brief.id === 'xor-risk'
              ? 'Quadrants I and III are where x₁ and x₂ have the same sign — meaning their product has a single sign everywhere in the positive class.'
              : 'At threshold 0.50, rare positives get missed (low recall). Try adding a non-linear feature for the curved corner AND lowering the threshold.'}
      </div>

      {entry.solved && (
        <div style={{ marginTop: 10 }}>
          <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>
            {showOpt ? 'Hide' : 'Show'} the exhaustive search
          </button>
          {showOpt && ex && (
            <div className="live-note" style={{ marginTop: 10 }}>
              All {ex.total.toLocaleString()} combinations were checked: <strong>{ex.passing}</strong> pass. Fewest parameters is{' '}
              <strong>{ex.bestParams}</strong> (best accuracy {pct1(ex.bestAccAtMin)}), achieved by:{' '}
              {ex.best
                .slice(0, 3)
                .map((w) => `[${w.keys.map((k) => LAB_FEATURES.find((f) => f.key === k)!.label).join(', ')}], λ = ${w.lambda}${brief.fixedThresh === undefined ? `, t = ${w.thresh.toFixed(2)}` : ''}`)
                .join('; or ')}
              .
            </div>
          )}
        </div>
      )}
    </Widget>
  );
}
