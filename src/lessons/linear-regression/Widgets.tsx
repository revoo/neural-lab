// authored-by: Gemini Antigravity
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
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
  labHint,
  linearVsLogistic1D,
  sigmoid,
  type FeatureKey,
  type LabBrief,
} from './regression';
import { sklearnLogRegLines } from './realCode';

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

const OPT_PARAMS: Record<LabBrief['id'], number> = { ring: 3, 'xor-risk': 2, imbalanced: 4 };

const signed = (v: number, digits = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`;

/** Clip a 2D line a*u + b*v = c to the box [x0, x1] x [y0, y1]. */
function clipLine(a: number, b: number, c: number, x0: number, x1: number, y0: number, y1: number): [[number, number], [number, number]] | null {
  const pts: [number, number][] = [];
  const pushUnique = (u: number, v: number) => {
    if (u < x0 - 1e-6 || u > x1 + 1e-6 || v < y0 - 1e-6 || v > y1 + 1e-6) return;
    if (pts.some(([pu, pv]) => Math.hypot(pu - u, pv - v) < 1e-4)) return;
    pts.push([Math.max(x0, Math.min(x1, u)), Math.max(y0, Math.min(y1, v))]);
  };
  if (Math.abs(b) > 1e-9) {
    pushUnique(x0, (c - a * x0) / b);
    pushUnique(x1, (c - a * x1) / b);
  }
  if (Math.abs(a) > 1e-9) {
    pushUnique((c - b * y0) / a, y0);
    pushUnique((c - b * y1) / a, y1);
  }
  return pts.length >= 2 ? [pts[0], pts[1]] : null;
}

function seedShellProgress() {
  const old = loadLab();
  const solved: Record<string, boolean> = {};
  const optimal: Record<string, boolean> = {};
  for (const b of LAB_BRIEFS) {
    if (old[b.id].solved) solved[b.id] = true;
    if (old[b.id].bestParams !== null && old[b.id].bestParams! <= OPT_PARAMS[b.id]) optimal[b.id] = true;
  }
  return { solved, optimal };
}

export function ClassicalLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [selectedPt, setSelectedPt] = useState<number | null>(null);
  const progress = useLabProgress('nl-logreg-lab-v1-shell', seedShellProgress);
  const log = useMachineLog();

  const brief = LAB_BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const thresh = brief.fixedThresh ?? entry.thresh;
  const logitT = Math.log(thresh / (1 - thresh));
  const data = useMemo(() => getLabData(brief.id), [brief.id]);
  const res = useMemo(() => evaluateLab(brief.id, entry.keys, entry.lambda, thresh), [brief.id, entry.keys, entry.lambda, thresh]);
  const pass = briefPasses(brief, res);
  const isOpt = pass && res.params <= OPT_PARAMS[brief.id];
  const ex = useMemo(() => labExhaustive(brief), [brief]);

  // Animate all 6 feature weights (0 if inactive) so bars glide smoothly when toggling
  const fullWeights = useMemo(
    () =>
      LAB_FEATURES.map((f) => {
        const idx = entry.keys.indexOf(f.key);
        return idx >= 0 ? res.w[idx + 1] : 0;
      }),
    [entry.keys, res.w],
  );
  const shownWeights = useTweenedArray(fullWeights, 240);
  const maxAbsW = Math.max(1.5, ...fullWeights.map(Math.abs), Math.abs(res.w[0]));

  // Memoised heatmap function (recomputes only when weights/keys/thresh change)
  const heatFn = useMemo(() => {
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

  const formatKeys = (ks: FeatureKey[]) => `[${ks.map((k) => LAB_FEATURES.find((f) => f.key === k)!.label).join(', ')}]`;

  const logFit = (b: LabBrief, ks: FeatureKey[], lam: number, t: number) => {
    const r = evaluateLab(b.id, ks, lam, t);
    const ok = briefPasses(b, r);
    const tStr = b.fixedThresh === undefined ? `, t=${t.toFixed(2)}` : '';
    const metric =
      b.minRecall !== undefined
        ? `rec ${pct1(r.testRecall)}, prec ${pct1(r.testPrecision)}`
        : `acc ${pct1(r.testAcc)}`;
    log.push(
      `fit ${formatKeys(ks)}, λ=${lam}${tStr} → ${r.iters} Newton iters · loss ${r.finalLoss.toFixed(3)} · ${metric} (TP ${r.tp}, FP ${r.fp}, FN ${r.fn}) ${ok ? '✓' : '✗'}`,
      ok ? 'good' : 'info',
    );
  };

  const update = (patch: Partial<LabEntry>) => {
    const nextEntry = { ...entry, ...patch };
    persist({ ...lab, [brief.id]: nextEntry });
    logFit(brief, nextEntry.keys, nextEntry.lambda, brief.fixedThresh ?? nextEntry.thresh);
  };

  const toggleKey = (k: FeatureKey) => {
    const has = entry.keys.includes(k);
    if (has && entry.keys.length === 1) return; // keep at least one feature
    const next = has ? entry.keys.filter((x) => x !== k) : [...entry.keys, k];
    update({ keys: next });
  };

  const resetBrief = () => {
    const base: LabEntry = { keys: ['x1', 'x2'], lambda: 0.1, thresh: brief.fixedThresh ?? 0.5, solved: entry.solved, bestParams: entry.bestParams };
    persist({ ...lab, [brief.id]: base });
    setSelectedPt(null);
    logFit(brief, base.keys, base.lambda, brief.fixedThresh ?? base.thresh);
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

  // Machine's-eye view geometry (feature space or 1D log-odds sigmoid)
  const machineView = useMemo(() => {
    const nKeys = entry.keys.length;
    if (nKeys === 2) {
      const f1 = LAB_FEATURES.find((f) => f.key === entry.keys[0])!;
      const f2 = LAB_FEATURES.find((f) => f.key === entry.keys[1])!;
      const us = res.Xte.map((row) => row[1]);
      const vs = res.Xte.map((row) => row[2]);
      const clampRange = (arr: number[]): [number, number] => {
        const s = [...arr].sort((a, b) => a - b);
        const lo = s[Math.floor(s.length * 0.02)];
        const hi = s[Math.floor(s.length * 0.98)];
        const pad = Math.max(0.25, (hi - lo) * 0.12);
        return [lo - pad, hi + pad];
      };
      const xRange = clampRange(us);
      const yRange = clampRange(vs);
      const line = clipLine(res.w[1], res.w[2], logitT - res.w[0], xRange[0], xRange[1], yRange[0], yRange[1]);
      return { kind: '2d' as const, xLabel: f1.label, yLabel: f2.label, xRange, yRange, us, vs, line };
    }
    if (nKeys === 1) {
      const f1 = LAB_FEATURES.find((f) => f.key === entry.keys[0])!;
      const us = res.Xte.map((row) => row[1]);
      const s = [...us].sort((a, b) => a - b);
      const lo = s[Math.floor(s.length * 0.02)];
      const hi = s[Math.floor(s.length * 0.98)];
      const pad = Math.max(0.25, (hi - lo) * 0.12);
      const xRange: [number, number] = [lo - pad, hi + pad];
      const cutX = Math.abs(res.w[1]) > 1e-6 ? (logitT - res.w[0]) / res.w[1] : null;
      return { kind: '1d' as const, xLabel: f1.label, xRange, us, cutX };
    }
    const zs = res.Xte.map((row) => row.reduce((acc, v, j) => acc + v * res.w[j], 0));
    const s = [...zs].sort((a, b) => a - b);
    const lo = Math.min(-3, s[Math.floor(s.length * 0.02)] - 0.5);
    const hi = Math.max(3, s[Math.floor(s.length * 0.98)] + 0.5);
    return { kind: 'logit' as const, xLabel: 'log-odds z = w·φ(x)', xRange: [lo, hi] as [number, number], zs };
  }, [entry.keys, res.Xte, res.w, logitT]);

  // Inspected point breakdown
  const inspected = useMemo(() => {
    if (selectedPt === null || selectedPt >= data.test.length) return null;
    const pt = data.test[selectedPt];
    const row = res.Xte[selectedPt];
    const z = row.reduce((acc, v, j) => acc + v * res.w[j], 0);
    const p = res.probsTe[selectedPt];
    const pred: 0 | 1 = p >= thresh ? 1 : 0;
    const verdict = pred === 1 && pt.y === 1 ? 'TP' : pred === 1 && pt.y === 0 ? 'FP' : pred === 0 && pt.y === 1 ? 'FN' : 'TN';
    return { idx: selectedPt, pt, row, z, p, pred, verdict, ok: pred === pt.y };
  }, [selectedPt, data.test, res.Xte, res.w, res.probsTe, thresh]);

  const liveCode = useMemo(() => sklearnLogRegLines(entry.keys, entry.lambda, thresh), [entry.keys, entry.lambda, thresh]);

  const shellBriefs = useMemo(
    () =>
      LAB_BRIEFS.map((b, i) => ({
        id: b.id,
        tab: `Brief ${i + 1}`,
        title: b.title,
        text: b.story,
      })),
    [],
  );

  const statusNode = pass ? (
    <>
      <strong>{isOpt ? 'Optimal — fewest possible parameters!' : 'Meets the brief!'}</strong> {res.params} parameters (1 intercept + {entry.keys.length} feature{entry.keys.length === 1 ? '' : 's'}) · converged in {res.iters} Newton steps.{' '}
      {!isOpt && `Optional: can you drop an unnecessary feature and still pass with ≤ ${OPT_PARAMS[brief.id]} parameters?`}
    </>
  ) : (
    <>
      <strong>Not yet.</strong> {res.params} params · {res.iters} Newton steps ·{' '}
      {brief.minAcc !== undefined
        ? `held-out accuracy ${pct1(res.testAcc)} (need ≥ ${pct1(brief.minAcc)}).`
        : `recall ${pct1(res.testRecall)} (need ≥ ${pct1(brief.minRecall!)}), precision ${pct1(res.testPrecision)} (need ≥ ${pct1(brief.minPrecision!)}).`}{' '}
      Compare the raw input plane on the left with what the model actually sees on the right.
    </>
  );

  const paretoReveal = (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ fontSize: 13.5, lineHeight: 1.55 }}>
        All <strong>{ex.total.toLocaleString()}</strong> combinations in this grid were solved exhaustively: <strong>{ex.passing}</strong> pass. Fewest parameters is{' '}
        <strong>{ex.bestParams}</strong> (best accuracy {pct1(ex.bestAccAtMin)}), achieved by:{' '}
        <span className="mono">
          {ex.best
            .slice(0, 3)
            .map((w) => `[${w.keys.map((k) => LAB_FEATURES.find((f) => f.key === k)!.label).join(', ')}], λ=${w.lambda}${brief.fixedThresh === undefined ? `, t=${w.thresh.toFixed(2)}` : ''}`)
            .join('; or ')}
        </span>
        .
      </div>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <Plot
          width={420}
          height={180}
          x={[1.5, 7.5]}
          y={[0.45, 1.02]}
          pad={34}
          xLabel="parameters (intercept + features)"
          yLabel={brief.minAcc !== undefined ? 'held-out accuracy' : 'held-out F1'}
          ariaLabel="Exhaustive solution space: parameters versus held-out score"
        >
          {({ x, y }) => {
            const metricOf = (pt: (typeof ex.points)[number]) => (brief.minAcc !== undefined ? pt.acc : pt.f1);
            const curMetric = brief.minAcc !== undefined ? res.testAcc : res.testF1;
            const optPt = ex.best[0];
            return (
              <>
                {brief.minAcc !== undefined && (
                  <line x1={x(1.5)} x2={x(7.5)} y1={y(brief.minAcc)} y2={y(brief.minAcc)} stroke="var(--c-good)" strokeDasharray="4 3" strokeWidth={1.2} />
                )}
                {ex.points.map((pt, i) => {
                  // deterministic tiny horizontal jitter so stacked grid points remain visible
                  const jitter = (((i * 37) % 11) - 5) * 0.03;
                  return (
                    <circle
                      key={i}
                      cx={x(pt.params + jitter)}
                      cy={y(Math.max(0.46, metricOf(pt)))}
                      r={pt.pass ? 3 : 2}
                      fill={pt.pass ? 'var(--c-good)' : 'var(--text-muted)'}
                      opacity={pt.pass ? 0.75 : 0.22}
                    />
                  );
                })}
                {optPt && (
                  <circle
                    cx={x(optPt.params)}
                    cy={y(Math.max(0.46, metricOf(optPt)))}
                    r={6.5}
                    fill="none"
                    stroke="var(--c-good)"
                    strokeWidth={2}
                  />
                )}
                <circle
                  cx={x(res.params)}
                  cy={y(Math.max(0.46, curMetric))}
                  r={5}
                  fill="var(--accent)"
                  stroke="var(--bg)"
                  strokeWidth={1.5}
                />
              </>
            );
          }}
        </Plot>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
        Every tested model in the search grid: faint dots fail, highlighted dots pass, ringed dot is the fewest-parameter optimum, and the accent dot is your current model.
      </div>
    </div>
  );

  return (
    <LabShell
      title="Lab: Feature Engineering + Logistic Regression"
      caption={
        <>
          Toggle engineered features, pick L2 regularisation λ (and decision threshold in Brief 3). Exact damped Newton–Raphson logistic regression fits live on 220 training points and scores on 250 held-out test points. Click any point on the left plot to inspect its feature vector and log-odds calculation. Meet each brief — and optionally use the fewest parameters possible.
        </>
      }
      briefs={shellBriefs}
      active={briefIdx}
      onSelect={(i) => {
        setBriefIdx(i);
        setSelectedPt(null);
      }}
      progress={progress}
      passing={pass}
      optimal={isOpt}
      status={statusNode}
      statusTone={pass ? 'good' : 'neutral'}
      hint={labHint(brief, res)}
      reveal={{ label: 'the exhaustive search & solution map', content: paretoReveal }}
      log={{
        lines: log.lines,
        onClear: log.clear,
        title: 'Newton–Raphson solver log',
        empty: 'Toggle a feature or change λ to record a Newton–Raphson solve.',
      }}
      actions={
        <button type="button" className="btn small" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {/* Controls + metrics row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 14, alignItems: 'start', marginBottom: 12 }}>
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

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'end' }}>
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
          </div>
        </div>

        <div className="readout" style={{ margin: 0 }}>
          <span className="k">parameters (1 + features) </span>
          <strong>{res.params}</strong>
          {' · '}
          <span className="k">Newton iters </span>
          <strong>{res.iters}</strong>
          {' · '}
          <span className="k">CE loss </span>
          {res.finalLoss.toFixed(3)}
          {'\n'}
          {brief.minAcc !== undefined && (
            <>
              <span style={{ color: res.testAcc >= brief.minAcc - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {res.testAcc >= brief.minAcc - 1e-9 ? '✓' : '✗'}
              </span>{' '}
              <span className="k">held-out accuracy </span>
              <strong>{pct1(res.testAcc)}</strong> (need ≥ {pct1(brief.minAcc)})
              {'\n'}
            </>
          )}
          {brief.minRecall !== undefined && (
            <>
              <span style={{ color: res.testRecall >= brief.minRecall - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {res.testRecall >= brief.minRecall - 1e-9 ? '✓' : '✗'}
              </span>{' '}
              <span className="k">held-out recall </span>
              <strong>{pct1(res.testRecall)}</strong> (need ≥ {pct1(brief.minRecall)})
              {'\n'}
            </>
          )}
          {brief.minPrecision !== undefined && (
            <>
              <span style={{ color: res.testPrecision >= brief.minPrecision - 1e-9 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {res.testPrecision >= brief.minPrecision - 1e-9 ? '✓' : '✗'}
              </span>{' '}
              <span className="k">held-out precision </span>
              <strong>{pct1(res.testPrecision)}</strong> (need ≥ {pct1(brief.minPrecision)})
              {'\n'}
            </>
          )}
          <span className="k">confusion (250 test) </span>
          TP {res.tp} · FP {res.fp} · FN {res.fn} · TN {res.tn}
          {' · '}
          <span className="k">train </span>
          {pct1(res.trainAcc)}
        </div>
      </div>

      {/* Dual view: Input Space (left) vs What the Model Sees (right) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-muted)' }}>
            Input space (x₁, x₂) · ringed dots are errors · click a dot to inspect
          </div>
          <Plot width={300} height={280} x={[-3.1, 3.1]} y={[-3.1, 3.1]} pad={28} xLabel="x₁" yLabel="x₂" ariaLabel="Learned logistic regression decision surface on held-out points">
            <Heatmap f={heatFn} range={[0, 1]} colormap={diverging} resolution={130} opacity={0.55} />
            {({ x, y }) => (
              <>
                {data.test.map((pt, i) => {
                  if (Math.abs(pt.x1) > 3.05 || Math.abs(pt.x2) > 3.05) return null;
                  const pred = res.probsTe[i] >= thresh ? 1 : 0;
                  const wrong = pred !== pt.y;
                  const isSel = selectedPt === i;
                  return (
                    <g key={i} style={{ cursor: 'pointer' }} onClick={() => setSelectedPt(i)}>
                      {wrong && (
                        <circle cx={x(pt.x1)} cy={y(pt.x2)} r={5.4} fill="none" stroke="var(--c-bad)" strokeWidth={1.3} opacity={0.9} />
                      )}
                      {isSel && (
                        <circle cx={x(pt.x1)} cy={y(pt.x2)} r={7.5} fill="none" stroke="var(--accent)" strokeWidth={2} />
                      )}
                      <circle
                        cx={x(pt.x1)}
                        cy={y(pt.x2)}
                        r={isSel ? 4 : 2.7}
                        fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                        stroke="var(--bg)"
                        strokeWidth={0.6}
                        opacity={0.92}
                      />
                    </g>
                  );
                })}
              </>
            )}
          </Plot>
        </div>

        <div style={{ display: 'grid', gap: 4, justifyItems: 'center' }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-muted)' }}>
            {machineView.kind === '2d'
              ? `What the model sees: straight line in (${machineView.xLabel}, ${machineView.yLabel})`
              : machineView.kind === '1d'
                ? `What the model sees: 1D sigmoid along ${machineView.xLabel}`
                : 'What the model sees: 1D log-odds projection z = w·φ(x)'}
          </div>
          {machineView.kind === '2d' ? (
            <Plot
              width={300}
              height={280}
              x={machineView.xRange}
              y={machineView.yRange}
              pad={30}
              xLabel={machineView.xLabel}
              yLabel={machineView.yLabel}
              ariaLabel="Feature space view showing the linear decision boundary"
            >
              {({ x, y }) => (
                <>
                  {machineView.line && (
                    <line
                      x1={x(machineView.line[0][0])}
                      y1={y(machineView.line[0][1])}
                      x2={x(machineView.line[1][0])}
                      y2={y(machineView.line[1][1])}
                      stroke="var(--accent)"
                      strokeWidth={2.4}
                      strokeDasharray="6 3"
                    />
                  )}
                  {data.test.map((pt, i) => {
                    const u = machineView.us[i];
                    const v = machineView.vs[i];
                    if (u < machineView.xRange[0] || u > machineView.xRange[1] || v < machineView.yRange[0] || v > machineView.yRange[1]) return null;
                    const wrong = (res.probsTe[i] >= thresh ? 1 : 0) !== pt.y;
                    const isSel = selectedPt === i;
                    return (
                      <g key={i} style={{ cursor: 'pointer' }} onClick={() => setSelectedPt(i)}>
                        {wrong && <circle cx={x(u)} cy={y(v)} r={5.2} fill="none" stroke="var(--c-bad)" strokeWidth={1.2} opacity={0.85} />}
                        {isSel && <circle cx={x(u)} cy={y(v)} r={7.2} fill="none" stroke="var(--accent)" strokeWidth={2} />}
                        <circle cx={x(u)} cy={y(v)} r={isSel ? 3.8 : 2.5} fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'} stroke="var(--bg)" strokeWidth={0.5} opacity={0.9} />
                      </g>
                    );
                  })}
                </>
              )}
            </Plot>
          ) : machineView.kind === '1d' ? (
            <Plot
              width={300}
              height={280}
              x={machineView.xRange}
              y={[-0.06, 1.06]}
              pad={30}
              xLabel={machineView.xLabel}
              yLabel="P(y = 1)"
              ariaLabel="1D sigmoid curve over the single engineered feature"
            >
              {({ x, y }) => {
                const [lo, hi] = machineView.xRange;
                const dPath = Array.from({ length: 65 }, (_, k) => {
                  const u = lo + ((hi - lo) * k) / 64;
                  const p = sigmoid(res.w[0] + res.w[1] * u);
                  return `${k === 0 ? 'M' : 'L'}${x(u).toFixed(1)},${y(p).toFixed(1)}`;
                }).join(' ');
                return (
                  <>
                    <line x1={x(lo)} x2={x(hi)} y1={y(thresh)} y2={y(thresh)} stroke="var(--text-muted)" strokeDasharray="4 3" strokeWidth={1.1} />
                    {machineView.cutX !== null && machineView.cutX >= lo && machineView.cutX <= hi && (
                      <line x1={x(machineView.cutX)} x2={x(machineView.cutX)} y1={y(0)} y2={y(1)} stroke="var(--accent)" strokeDasharray="5 3" strokeWidth={1.8} />
                    )}
                    <path d={dPath} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
                    {data.test.map((pt, i) => {
                      const u = machineView.us[i];
                      if (u < lo || u > hi) return null;
                      const p = res.probsTe[i];
                      const wrong = (p >= thresh ? 1 : 0) !== pt.y;
                      const isSel = selectedPt === i;
                      return (
                        <g key={i} style={{ cursor: 'pointer' }} onClick={() => setSelectedPt(i)}>
                          {wrong && <circle cx={x(u)} cy={y(p)} r={5.2} fill="none" stroke="var(--c-bad)" strokeWidth={1.2} />}
                          {isSel && <circle cx={x(u)} cy={y(p)} r={7} fill="none" stroke="var(--accent)" strokeWidth={2} />}
                          <circle cx={x(u)} cy={y(p)} r={isSel ? 3.8 : 2.5} fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'} stroke="var(--bg)" strokeWidth={0.5} opacity={0.85} />
                        </g>
                      );
                    })}
                  </>
                );
              }}
            </Plot>
          ) : (
            <Plot
              width={300}
              height={280}
              x={machineView.xRange}
              y={[-0.06, 1.06]}
              pad={30}
              xLabel={machineView.xLabel}
              yLabel="P(y = 1)"
              ariaLabel="1D log-odds projection and sigmoid threshold"
            >
              {({ x, y }) => {
                const [lo, hi] = machineView.xRange;
                const dPath = Array.from({ length: 65 }, (_, k) => {
                  const z = lo + ((hi - lo) * k) / 64;
                  return `${k === 0 ? 'M' : 'L'}${x(z).toFixed(1)},${y(sigmoid(z)).toFixed(1)}`;
                }).join(' ');
                return (
                  <>
                    <line x1={x(lo)} x2={x(hi)} y1={y(thresh)} y2={y(thresh)} stroke="var(--text-muted)" strokeDasharray="4 3" strokeWidth={1.1} />
                    {logitT >= lo && logitT <= hi && (
                      <line x1={x(logitT)} x2={x(logitT)} y1={y(0)} y2={y(1)} stroke="var(--accent)" strokeDasharray="5 3" strokeWidth={1.8} />
                    )}
                    <path d={dPath} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
                    {data.test.map((pt, i) => {
                      const z = machineView.zs[i];
                      if (z < lo || z > hi) return null;
                      const p = res.probsTe[i];
                      const wrong = (p >= thresh ? 1 : 0) !== pt.y;
                      const isSel = selectedPt === i;
                      return (
                        <g key={i} style={{ cursor: 'pointer' }} onClick={() => setSelectedPt(i)}>
                          {wrong && <circle cx={x(z)} cy={y(p)} r={5.2} fill="none" stroke="var(--c-bad)" strokeWidth={1.2} />}
                          {isSel && <circle cx={x(z)} cy={y(p)} r={7} fill="none" stroke="var(--accent)" strokeWidth={2} />}
                          <circle cx={x(z)} cy={y(p)} r={isSel ? 3.8 : 2.5} fill={pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)'} stroke="var(--bg)" strokeWidth={0.5} opacity={0.85} />
                        </g>
                      );
                    })}
                  </>
                );
              }}
            </Plot>
          )}
        </div>
      </div>

      {/* Inside the machine: live fitted log-odds equation + animated weight bars */}
      <div style={{ marginTop: 12, padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: 6, alignItems: 'baseline' }}>
          <span className="field-label" style={{ margin: 0 }}>Fitted log-odds equation</span>
          <span className="mono" style={{ fontSize: 12.5 }}>
            z = {res.w[0].toFixed(2)}{' '}
            {entry.keys
              .map((k, idx) => `${signed(res.w[idx + 1])}·${LAB_FEATURES.find((f) => f.key === k)!.label}`)
              .join(' ')}
            {brief.fixedThresh === undefined ? `  (alarm if p ≥ ${thresh.toFixed(2)}, z ≥ ${logitT.toFixed(2)})` : ''}
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 8 }}>
          {LAB_FEATURES.map((f, idx) => {
            const active = entry.keys.includes(f.key);
            const wVal = shownWeights[idx];
            const pct = Math.min(100, (Math.abs(wVal) / maxAbsW) * 100);
            return (
              <div key={f.key} style={{ opacity: active ? 1 : 0.35, display: 'grid', gap: 3 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5 }} className="mono">
                  <span>{f.label}</span>
                  <span>{active ? signed(wVal) : 'off'}</span>
                </div>
                <div style={{ height: 8, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${pct}%`,
                      height: '100%',
                      background: wVal >= 0 ? 'var(--c-a)' : 'var(--c-b)',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
        {inspected && (
          <div className="mono" style={{ fontSize: 12, paddingTop: 6, borderTop: '1px solid var(--border)', color: 'var(--text-muted)' }}>
            Point #{inspected.idx + 1} ({inspected.pt.x1.toFixed(2)}, {inspected.pt.x2.toFixed(2)}), true y={inspected.pt.y} →{' '}
            φ=[{entry.keys.map((k, j) => `${LAB_FEATURES.find((f) => f.key === k)!.label}=${inspected.row[j + 1].toFixed(2)}`).join(', ')}] →{' '}
            z={inspected.z.toFixed(2)} → p={(100 * inspected.p).toFixed(1)}% →{' '}
            <strong style={{ color: inspected.ok ? 'var(--c-good)' : 'var(--c-bad)' }}>{inspected.verdict}</strong>
          </div>
        )}
      </div>

      <LiveCode code={liveCode} note="scikit-learn LogisticRegression (C = 1/λ)" />
    </LabShell>
  );
}
