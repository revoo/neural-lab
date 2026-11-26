// authored-by: Gemini Antigravity
import { useCallback, useEffect, useMemo, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { Segmented } from '../../components/controls';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Plot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import { hiddenLineSegment } from './tinyMlp';
import {
  BRIEF1_NET,
  BRIEF1_PROBES,
  BRIEF2_ACTS,
  BRIEF2_INIT_MODES,
  BRIEF2_LRS,
  BRIEF2_STEPS,
  BRIEF2_TARGET_LOSS,
  BRIEF3_LRS,
  BRIEF3_STEPS,
  BRIEF3_TARGET_LOSS,
  CREDIT_XOR,
  brief1Hint,
  brief2Hint,
  brief3Hint,
  checkBrief1,
  evaluateBrief1,
  evaluateBrief2,
  evaluateBrief3,
  mlpPredict,
  type Brief1WeightKey,
  type Brief2Config,
  type Brief3Config,
  type CreditAct,
  type InitMode,
  type SignChoice,
} from './creditLab';
import { BRIEF2_GRID, BRIEF2_SUMMARY, BRIEF3_GRID, BRIEF3_SUMMARY } from './results';
import { brief1PyTorchCode, brief2PyTorchCode, brief3PyTorchCode } from './realCode';

const STORAGE_KEY = 'neural-lab:credit-workbench-v1';

interface SavedState {
  active: number;
  b1Guesses: Record<Brief1WeightKey, SignChoice | null>;
  b1Largest: Brief1WeightKey | null;
  b1Checked: boolean;
  b2: Brief2Config;
  b3: Brief3Config;
}

const DEFAULT_STATE: SavedState = {
  active: 0,
  b1Guesses: { v1: null, w11: null, w21: null, w12: null },
  b1Largest: null,
  b1Checked: false,
  b2: { initMode: 'broken', act: 'tanh', lr: 1.0 },
  b3: { trainH1: false, trainH2: false, trainOut: true, lr: 1.5 },
};

function loadSaved(): SavedState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_STATE;
    const p = JSON.parse(raw) as Partial<SavedState>;
    return {
      active: typeof p.active === 'number' ? p.active : DEFAULT_STATE.active,
      b1Guesses: { ...DEFAULT_STATE.b1Guesses, ...(p.b1Guesses ?? {}) },
      b1Largest: p.b1Largest ?? null,
      b1Checked: Boolean(p.b1Checked),
      b2: { ...DEFAULT_STATE.b2, ...(p.b2 ?? {}) },
      b3: { ...DEFAULT_STATE.b3, ...(p.b3 ?? {}) },
    };
  } catch {
    return DEFAULT_STATE;
  }
}

const INIT_LABELS: Record<InitMode, string> = {
  broken: 'Broken (twins)',
  zeros: 'All zeros',
  'asym-large': 'Asym large (σ=6)',
  'asym-small': 'Asym small (σ=0.8)',
};

const ACT_LABELS: Record<CreditAct, string> = {
  tanh: 'tanh',
  sigmoid: 'sigmoid',
  relu: 'ReLU',
};

const PROBE_META: Record<Brief1WeightKey, { title: string; path: string }> = {
  v1: { title: 'v₁ (h₁ → ŷ)', path: 'output weight, currently −0.8' },
  w11: { title: 'w₁₁ (x₁ → h₁)', path: 'x₁ = +2.0 into active h₁' },
  w21: { title: 'w₂₁ (x₂ → h₁)', path: 'x₂ = −1.0 into active h₁' },
  w12: { title: 'w₁₂ (x₁ → h₂)', path: 'x₁ = +2.0 into h₂ (z₂ = −1.8)' },
};

const fmtS = (v: number, d = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;

export function CreditLab() {
  const [st, setSt] = useState<SavedState>(loadSaved);
  const progress = useLabProgress('backpropagation-credit-lab');
  const log = useMachineLog(10);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const b1Truth = useMemo(() => evaluateBrief1(BRIEF1_NET), []);
  const b1Check = useMemo(
    () => (st.b1Checked ? checkBrief1(st.b1Guesses, st.b1Largest) : null),
    [st.b1Checked, st.b1Guesses, st.b1Largest],
  );
  const b2Res = useMemo(() => evaluateBrief2(st.b2), [st.b2]);
  const b3Res = useMemo(() => evaluateBrief3(st.b3), [st.b3]);

  const b3Key = `${st.b3.trainH1}-${st.b3.trainH2}-${st.b3.trainOut}-${st.b3.lr}`;
  const b3Replay = useReplay(b3Res.snapshots.length - 1, 140, b3Key, false);
  const b3Snap = b3Res.snapshots[b3Replay.i] ?? b3Res.snapshots[b3Res.snapshots.length - 1];

  const b3Predict = useCallback(
    (x1: number, x2: number) => mlpPredict(b3Snap.model, 'tanh', x1, x2),
    [b3Snap.model],
  );

  const briefs: LabBrief[] = [
    {
      id: 'trace-signs',
      tab: '1. Trace signs',
      title: 'Brief 1 — Predict the update signs without a calculator',
      text: (
        <>
          Below is a frozen <span className="mono">2 → 2 → 1</span> ReLU network on one example (
          <span className="mono">x = [+2.0, −1.0]</span>, target <span className="mono">y = +1.0</span>). Its forward
          values are visible, and its prediction <span className="mono">ŷ = −1.0</span> is too low. Predict whether
          one gradient-descent step (<span className="mono">w ← w − η·grad</span>) will{' '}
          <strong>increase (+)</strong>, <strong>decrease (−)</strong>, or leave <strong>unchanged (0)</strong> each
          of the four probed weights. <em>Optional goal:</em> also identify which weight has the largest{' '}
          <span className="mono">|gradient|</span>.
        </>
      ),
    },
    {
      id: 'unstick-hidden',
      tab: '2. Unstick layer',
      title: 'Brief 2 — Diagnose and unstick the broken hidden layer',
      text: (
        <>
          This <span className="mono">2 → 4 → 1</span> XOR network starts broken — inspect its step-0 gradient probe
          and loss curves across 5 fixed seeds. Fix the setup so <strong>all 5 seeds</strong> reach{' '}
          <strong>100% XOR accuracy</strong> and <strong>loss ≤ {BRIEF2_TARGET_LOSS}</strong> within {BRIEF2_STEPS}{' '}
          steps. <em>Optional goal:</em> match the fastest mean solve time over the 48-config grid.
        </>
      ),
    },
    {
      id: 'freeze-hidden',
      tab: '3. Untangle XOR',
      title: 'Brief 3 — Do you actually need hidden-layer backprop?',
      text: (
        <>
          This <span className="mono">2 → 2 → 1</span> tanh network starts with its hidden layer frozen (output-only
          training). Watch the exact 2D hidden plane <span className="mono">(h₁, h₂)</span> on the right as you train
          for {BRIEF3_STEPS} steps: reach <strong>100% XOR accuracy</strong> and{' '}
          <strong>loss ≤ {BRIEF3_TARGET_LOSS}</strong>. <em>Optional goal:</em> pass while unfreezing the{' '}
          <strong>fewest possible hidden neurons</strong>.
        </>
      ),
    },
  ];

  // Passing & optimal flags per active brief
  const passing =
    st.active === 0 ? Boolean(b1Check?.allSignsCorrect) : st.active === 1 ? b2Res.pass : b3Res.pass;

  const optimal =
    st.active === 0
      ? Boolean(b1Check?.allSignsCorrect && b1Check?.largestCorrect)
      : st.active === 1
        ? Boolean(
            b2Res.pass &&
              b2Res.meanSolveStep !== null &&
              Math.abs(b2Res.meanSolveStep - (BRIEF2_SUMMARY.fastest.meanSolveStep ?? 0)) < 0.05,
          )
        : Boolean(b3Res.pass && b3Res.unfrozenHiddenCount === BRIEF3_SUMMARY.minUnfrozenHidden);

  const setB1Guess = (k: Brief1WeightKey, v: SignChoice) => {
    setSt((s) => ({ ...s, b1Guesses: { ...s.b1Guesses, [k]: v }, b1Checked: false }));
  };

  const runB1Check = () => {
    const chk = checkBrief1(st.b1Guesses, st.b1Largest);
    setSt((s) => ({ ...s, b1Checked: true }));
    log.push(
      `Brief 1 check: ${chk.correctCount}/4 signs right${chk.largestCorrect ? ' · largest |grad| right (w11 = +3.20)' : ''}`,
      chk.allSignsCorrect ? 'good' : 'bad',
    );
  };

  const updateB2 = (patch: Partial<Brief2Config>) => {
    const next = { ...st.b2, ...patch };
    setSt((s) => ({ ...s, b2: next }));
    const res = evaluateBrief2(next);
    log.push(
      `Brief 2 [${INIT_LABELS[next.initMode]} · ${ACT_LABELS[next.act]} · η=${next.lr}]: ${res.solvedSeeds}/5 seeds pass · mean loss ${res.meanLoss.toFixed(4)}`,
      res.pass ? 'good' : 'info',
    );
  };

  const updateB3 = (patch: Partial<Brief3Config>) => {
    const next = { ...st.b3, ...patch };
    setSt((s) => ({ ...s, b3: next }));
    const res = evaluateBrief3(next);
    const maskLabel = `h1:${next.trainH1 ? 'ON' : 'frozen'} h2:${next.trainH2 ? 'ON' : 'frozen'} out:${next.trainOut ? 'ON' : 'frozen'}`;
    log.push(
      `Brief 3 [${maskLabel} · η=${next.lr}]: ${res.pass ? `PASS @ step ${res.solvedStep}` : `loss ${res.finalLoss.toFixed(3)} (${(res.finalAcc * 100).toFixed(0)}%)`} · hidden ${res.finalSeparable ? 'untangled' : 'tangled'}`,
      res.pass ? 'good' : 'info',
    );
  };

  // Status text
  let status: React.ReactNode;
  let statusTone: 'good' | 'bad' | 'neutral' = 'neutral';
  if (st.active === 0) {
    if (!b1Check) {
      const picked = BRIEF1_PROBES.filter((k) => st.b1Guesses[k] !== null).length;
      status = `Picked ${picked} / 4 update directions. Press “Check & run backprop” to reveal the exact gradients.`;
    } else if (b1Check.allSignsCorrect && b1Check.largestCorrect) {
      statusTone = 'good';
      status = `All 4 signs right AND you spotted the surprise: hidden weight w₁₁ (|grad| = 3.20) takes a larger step than output weight v₁ (|grad| = 3.00).`;
    } else if (b1Check.allSignsCorrect) {
      statusTone = 'good';
      status = `All 4 update signs right! Optional goal: check the revealed |grad| magnitudes — which weight moves most?`;
    } else {
      statusTone = 'bad';
      status = `${b1Check.correctCount} / 4 signs right. Compare the highlighted forward values and exact gradients below, then adjust your predictions.`;
    }
  } else if (st.active === 1) {
    if (b2Res.pass) {
      statusTone = 'good';
      status = (
        <>
          <strong>Solved!</strong> All 5 seeds reached 100% accuracy and loss ≤ {BRIEF2_TARGET_LOSS} in an average of{' '}
          <strong className="mono">{b2Res.meanSolveStep?.toFixed(1)}</strong> steps (mean final loss{' '}
          <span className="mono">{b2Res.meanLoss.toFixed(4)}</span>).
          {optimal
            ? ' That matches the exhaustive fastest optimum across all 48 settings!'
            : ` Optional goal: exhaustive fastest across the 48 settings is ${BRIEF2_SUMMARY.fastest.meanSolveStep?.toFixed(1)} mean steps.`}
        </>
      );
    } else {
      statusTone = 'bad';
      status = (
        <>
          <strong>Not solved:</strong> {b2Res.solvedSeeds} / 5 seeds passed (mean acc{' '}
          <span className="mono">{(b2Res.meanAcc * 100).toFixed(0)}%</span>, mean loss{' '}
          <span className="mono">{b2Res.meanLoss.toFixed(4)}</span>). Inspect the step-0 hidden gradient probe to see
          why learning stalled.
        </>
      );
    }
  } else {
    if (b3Res.pass) {
      statusTone = 'good';
      status = (
        <>
          <strong>Solved at step {b3Res.solvedStep}!</strong> Hidden space <span className="mono">(h₁, h₂)</span>{' '}
          untangled into linear separability with{' '}
          <strong className="mono">{b3Res.unfrozenHiddenCount}</strong> unfrozen hidden neuron
          {b3Res.unfrozenHiddenCount === 1 ? '' : 's'} (final loss{' '}
          <span className="mono">{b3Res.finalLoss.toFixed(4)}</span>).
          {optimal
            ? ' ★ Minimum possible unfrozen hidden neurons (1)!'
            : ' Optional goal: can you still untangle XOR while freezing one of the hidden neurons?'}
        </>
      );
    } else {
      statusTone = 'bad';
      status = (
        <>
          <strong>Failed after {BRIEF3_STEPS} steps:</strong> accuracy{' '}
          <span className="mono">{(b3Res.finalAcc * 100).toFixed(0)}%</span>, loss{' '}
          <span className="mono">{b3Res.finalLoss.toFixed(4)}</span>. Hidden space is{' '}
          <strong>{b3Res.finalSeparable ? 'separable' : 'still tangled (X crosses)'}</strong>.
        </>
      );
    }
  }

  // Evidence-only hint
  const hint = st.active === 0 ? brief1Hint(b1Check) : st.active === 1 ? brief2Hint(b2Res) : brief3Hint(b3Res);

  // Post-solve reveal
  const reveal =
    st.active === 0
      ? {
          label: 'the exact chain-rule breakdown',
          content: (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ overflowX: 'auto' }}>
                <table className="num-table" style={{ width: '100%', fontSize: 12.5 }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left' }}>Weight</th>
                      <th style={{ textAlign: 'left' }}>Chain-rule product</th>
                      <th style={{ textAlign: 'right' }}>grad</th>
                      <th style={{ textAlign: 'center' }}>Update (−η·grad)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="mono">v₁</td>
                      <td className="mono">error (−2.0) × h₁ (+1.5)</td>
                      <td className="mono" style={{ textAlign: 'right' }}>−3.00</td>
                      <td className="mono" style={{ textAlign: 'center', color: 'var(--c-a)', fontWeight: 700 }}>+ (increase)</td>
                    </tr>
                    <tr>
                      <td className="mono">w₁₁</td>
                      <td className="mono">error (−2.0) × v₁ (−0.8) × 1 × x₁ (+2.0)</td>
                      <td className="mono" style={{ textAlign: 'right', fontWeight: 700 }}>+3.20</td>
                      <td className="mono" style={{ textAlign: 'center', color: 'var(--c-b)', fontWeight: 700 }}>− (decrease) · largest!</td>
                    </tr>
                    <tr>
                      <td className="mono">w₂₁</td>
                      <td className="mono">error (−2.0) × v₁ (−0.8) × 1 × x₂ (−1.0)</td>
                      <td className="mono" style={{ textAlign: 'right' }}>−1.60</td>
                      <td className="mono" style={{ textAlign: 'center', color: 'var(--c-a)', fontWeight: 700 }}>+ (increase)</td>
                    </tr>
                    <tr>
                      <td className="mono">w₁₂</td>
                      <td className="mono">error (−2.0) × v₂ (+0.7) × 0 (dead ReLU) × x₁ (+2.0)</td>
                      <td className="mono" style={{ textAlign: 'right' }}>0.00</td>
                      <td className="mono" style={{ textAlign: 'center', color: 'var(--text-muted)', fontWeight: 700 }}>0 (unchanged)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <LiveCode code={brief1PyTorchCode()} note="Exact PyTorch autograd equivalent for Brief 1." />
            </div>
          ),
        }
      : st.active === 1
        ? {
            label: 'the exhaustive 48-config search',
            content: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ fontSize: 13 }}>
                  Across all <strong>{BRIEF2_SUMMARY.total}</strong> combinations (4 initialisations × 3 activations × 4
                  step sizes, 5 fixed seeds each): <strong>{BRIEF2_SUMMARY.passing} pass</strong>.{' '}
                  <strong>0/12</strong> broken-twin configs and <strong>0/12</strong> all-zero configs pass.{' '}
                  <strong>Honest finding:</strong> large asymmetric weights (<span className="mono">σ = 6</span>) fail{' '}
                  <strong>0/8</strong> on <span className="mono">tanh</span> and <span className="mono">sigmoid</span>{' '}
                  (saturation kills at least one seed), yet pass <strong>3/4</strong> on <span className="mono">ReLU</span>{' '}
                  (which never saturates positively) and produce the fastest solve at{' '}
                  <strong className="mono">{BRIEF2_SUMMARY.fastest.meanSolveStep?.toFixed(1)}</strong> mean steps (
                  <span className="mono">asym-large · ReLU · η=1.0</span>), ahead of{' '}
                  <span className="mono">asym-small · ReLU · η=2.0</span> (<span className="mono">14.4</span> steps) and{' '}
                  <span className="mono">asym-small · tanh · η=2.0</span> (<span className="mono">19.6</span> steps).
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left' }}>Init</th>
                        <th style={{ textAlign: 'left' }}>Act</th>
                        <th style={{ textAlign: 'right' }}>η</th>
                        <th style={{ textAlign: 'right' }}>Seeds</th>
                        <th style={{ textAlign: 'right' }}>Mean step</th>
                        <th style={{ textAlign: 'right' }}>Mean loss</th>
                      </tr>
                    </thead>
                    <tbody>
                      {BRIEF2_GRID.filter((r) => r.pass)
                        .sort((a, b) => (a.meanSolveStep ?? 999) - (b.meanSolveStep ?? 999))
                        .map((r) => (
                          <tr key={`${r.initMode}-${r.act}-${r.lr}`}>
                            <td style={{ textAlign: 'left', whiteSpace: 'nowrap' }}>{INIT_LABELS[r.initMode]}</td>
                            <td className="mono" style={{ textAlign: 'left' }}>{ACT_LABELS[r.act]}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{r.lr.toFixed(1)}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{r.solvedSeeds}/5</td>
                            <td className="mono" style={{ textAlign: 'right', fontWeight: 700 }}>{r.meanSolveStep?.toFixed(1)}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{r.meanLoss.toFixed(4)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
                <LiveCode code={brief2PyTorchCode(st.b2)} note="Live PyTorch snippet matching your active Brief 2 controls." />
              </div>
            ),
          }
        : {
            label: 'the exhaustive 32-config freeze map',
            content: (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ fontSize: 13 }}>
                  Across all <strong>{BRIEF3_SUMMARY.total}</strong> freeze-mask × learning-rate combinations:{' '}
                  <strong>{BRIEF3_SUMMARY.passing} pass</strong>. Output-only training (both hidden units frozen) passes{' '}
                  <strong>0/4</strong> — and stays non-separable even at 2,000 steps. Freezing the output layer passes{' '}
                  <strong>0/16</strong>. Unfreezing just <strong>{BRIEF3_SUMMARY.minUnfrozenHidden}</strong> hidden
                  neuron suffices in <strong>{BRIEF3_SUMMARY.minUnfrozenPassing}</strong> routes (fastest:{' '}
                  <span className="mono">h₂ only + out · η=2.5</span> at step{' '}
                  <strong className="mono">{BRIEF3_SUMMARY.fastestMinUnfrozen.solvedStep}</strong>); unfreezing both
                  hidden neurons passes at all 4 learning rates (fastest step{' '}
                  <strong className="mono">{BRIEF3_SUMMARY.fastestOverall.solvedStep}</strong>).
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left' }}>Mask (h₁ / h₂ / out)</th>
                        <th style={{ textAlign: 'right' }}>Unfrozen hidden</th>
                        <th style={{ textAlign: 'right' }}>η</th>
                        <th style={{ textAlign: 'right' }}>Solve step</th>
                        <th style={{ textAlign: 'right' }}>Final loss</th>
                      </tr>
                    </thead>
                    <tbody>
                      {BRIEF3_GRID.filter((r) => r.pass)
                        .sort((a, b) => a.unfrozenHiddenCount - b.unfrozenHiddenCount || (a.solvedStep ?? 999) - (b.solvedStep ?? 999))
                        .map((r) => (
                          <tr key={`${r.trainH1}-${r.trainH2}-${r.trainOut}-${r.lr}`}>
                            <td className="mono" style={{ textAlign: 'left', whiteSpace: 'nowrap' }}>
                              {r.trainH1 ? 'h₁ ON' : 'h₁ off'} · {r.trainH2 ? 'h₂ ON' : 'h₂ off'} · {r.trainOut ? 'out ON' : 'out off'}
                            </td>
                            <td className="mono" style={{ textAlign: 'right', fontWeight: r.unfrozenHiddenCount === 1 ? 700 : 400 }}>
                              {r.unfrozenHiddenCount}
                            </td>
                            <td className="mono" style={{ textAlign: 'right' }}>{r.lr.toFixed(1)}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{r.solvedStep}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{r.finalLoss.toFixed(4)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
                <LiveCode code={brief3PyTorchCode(st.b3)} note="Live PyTorch requires_grad / register_hook code matching your active Brief 3 toggles." />
              </div>
            ),
          };

  // Hidden-space output boundary line for Brief 3: v1*h1 + v2*h2 + c = 0
  const b3OutLine = useMemo(
    () => hiddenLineSegment(b3Snap.model.W2[0], b3Snap.model.W2[1], b3Snap.model.b2, [-1.15, 1.15, -1.15, 1.15]),
    [b3Snap.model],
  );
  const b3H1Line = useMemo(
    () => hiddenLineSegment(b3Snap.model.W1[0][0], b3Snap.model.W1[0][1], b3Snap.model.b1[0], [-1.5, 1.5, -1.5, 1.5]),
    [b3Snap.model],
  );
  const b3H2Line = useMemo(
    () => hiddenLineSegment(b3Snap.model.W1[1][0], b3Snap.model.W1[1][1], b3Snap.model.b1[1], [-1.5, 1.5, -1.5, 1.5]),
    [b3Snap.model],
  );

  return (
    <LabShell
      title="Lab: The Credit-Assignment Workbench"
      caption="Three hands-on investigations into multi-layer backpropagation: trace signs by hand, unstick a broken hidden layer across 5 seeds, and prove why hidden-layer backprop is required to untangle XOR."
      briefs={briefs}
      active={st.active}
      onSelect={(i) => setSt((s) => ({ ...s, active: i }))}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={status}
      statusTone={statusTone}
      hint={hint}
      reveal={reveal}
      log={{ lines: log.lines, onClear: log.clear, title: 'Workbench log' }}
      actions={
        <button
          type="button"
          className="btn"
          onClick={() => {
            if (st.active === 0) setSt((s) => ({ ...s, b1Guesses: { v1: null, w11: null, w21: null, w12: null }, b1Largest: null, b1Checked: false }));
            if (st.active === 1) setSt((s) => ({ ...s, b2: { ...DEFAULT_STATE.b2 } }));
            if (st.active === 2) setSt((s) => ({ ...s, b3: { ...DEFAULT_STATE.b3 } }));
          }}
        >
          Reset brief
        </button>
      }
    >
      {/* ===================== BRIEF 1 ===================== */}
      {st.active === 0 && (
        <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          {/* Frozen network forward diagram */}
          <div style={{ flex: '1 1 340px', minWidth: 270 }}>
            <svg
              viewBox="0 0 500 245"
              style={{
                width: '100%',
                height: 'auto',
                display: 'block',
                background: 'var(--bg-sunken)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)',
              }}
              role="img"
              aria-label="Frozen 2-2-1 ReLU network for sign-tracing brief"
            >
              {/* Edges */}
              <line x1={74} y1={70} x2={180} y2={70} stroke="var(--c-a)" strokeWidth={2.2} />
              <line x1={74} y1={175} x2={180} y2={70} stroke="var(--c-b)" strokeWidth={2.2} />
              <line x1={74} y1={70} x2={180} y2={175} stroke="var(--c-b)" strokeWidth={1.8} strokeDasharray="4 3" />
              <line x1={74} y1={175} x2={180} y2={175} stroke="var(--border-strong)" strokeWidth={1.4} />
              <line x1={276} y1={70} x2={366} y2={122} stroke="var(--c-b)" strokeWidth={2.4} />
              <line x1={276} y1={175} x2={366} y2={122} stroke="var(--c-a)" strokeWidth={1.8} strokeDasharray="4 3" />

              {/* Weight labels */}
              <text x={127} y={60} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--c-a)" style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-sunken)', strokeWidth: 4 }}>
                w₁₁ = +0.5
              </text>
              <text x={112} y={138} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--c-b)" style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-sunken)', strokeWidth: 4 }}>
                w₂₁ = −0.5
              </text>
              <text x={144} y={114} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-sunken)', strokeWidth: 4 }}>
                w₁₂ = −0.6
              </text>
              <text x={127} y={191} textAnchor="middle" fontSize={10} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                w₂₂ = +0.4
              </text>
              <text x={324} y={84} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--c-b)" style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-sunken)', strokeWidth: 4 }}>
                v₁ = −0.8
              </text>
              <text x={324} y={164} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                v₂ = +0.7
              </text>

              {/* Inputs */}
              <rect x={16} y={48} width={58} height={44} rx={8} fill="var(--bg-elev)" stroke="var(--border-strong)" strokeWidth={1.4} />
              <text x={45} y={66} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>x₁</text>
              <text x={45} y={83} textAnchor="middle" fontSize={12.5} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>+2.0</text>

              <rect x={16} y={153} width={58} height={44} rx={8} fill="var(--bg-elev)" stroke="var(--border-strong)" strokeWidth={1.4} />
              <text x={45} y={171} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>x₂</text>
              <text x={45} y={188} textAnchor="middle" fontSize={12.5} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>−1.0</text>

              {/* Hidden h1 (active) & h2 (dead) */}
              <rect x={180} y={40} width={96} height={60} rx={9} fill="var(--bg-elev)" stroke="var(--accent)" strokeWidth={1.8} />
              <text x={228} y={58} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>h₁ (ReLU)</text>
              <text x={228} y={74} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>z₁ = +1.5</text>
              <text x={228} y={90} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--c-c)" style={{ fontFamily: 'var(--font-mono)' }}>h₁ = +1.5</text>

              <rect x={180} y={145} width={96} height={60} rx={9} fill="var(--bg-elev)" stroke="var(--c-warn)" strokeWidth={1.8} strokeDasharray="4 2" />
              <text x={228} y={163} textAnchor="middle" fontSize={10.5} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>h₂ (ReLU)</text>
              <text x={228} y={179} textAnchor="middle" fontSize={10.5} fill="var(--c-warn)" style={{ fontFamily: 'var(--font-mono)' }}>z₂ = −1.8</text>
              <text x={228} y={195} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>h₂ = 0.0</text>

              {/* Output */}
              <rect x={366} y={84} width={118} height={76} rx={10} fill="var(--bg-elev)" stroke="var(--c-grad)" strokeWidth={2} />
              <text x={425} y={104} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>
                ŷ = −1.0
              </text>
              <text x={425} y={121} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                target y = +1.0
              </text>
              <text x={425} y={139} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--c-grad)" style={{ fontFamily: 'var(--font-mono)' }}>
                ŷ − y = −2.0
              </text>
              <text x={425} y={154} textAnchor="middle" fontSize={9.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                (too low!)
              </text>
            </svg>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>
              Biases: <span className="mono">b₁ = 0.0</span>, <span className="mono">b₂ = −0.2</span>,{' '}
              <span className="mono">c = +0.2</span>. Squared error <span className="mono">L = ½(ŷ − y)²</span>.
            </div>
          </div>

          {/* 4 probe selectors + optional largest |grad| */}
          <div style={{ flex: '1 1 280px', minWidth: 260, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {BRIEF1_PROBES.map((k) => {
              const cur = st.b1Guesses[k];
              const ok = b1Check ? b1Check.perWeight[k] : null;
              return (
                <div
                  key={k}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-sunken)',
                    border: `1.5px solid ${ok === null ? 'var(--border)' : ok ? 'var(--c-good)' : 'var(--c-bad)'}`,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <div>
                      <strong className="mono" style={{ fontSize: 13 }}>{PROBE_META[k].title}</strong>
                      <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>{PROBE_META[k].path}</div>
                    </div>
                    <div className="btn-row" role="group" aria-label={`Update direction for ${k}`}>
                      {(['+', '-', '0'] as const).map((sign) => (
                        <button
                          key={sign}
                          type="button"
                          className={`btn small ${cur === sign ? 'primary' : ''}`}
                          onClick={() => setB1Guess(k, sign)}
                          data-probe={`${k}:${sign}`}
                        >
                          {sign === '+' ? '+ up' : sign === '-' ? '− down' : '0 same'}
                        </button>
                      ))}
                    </div>
                  </div>
                  {b1Check && (
                    <div className="mono" style={{ fontSize: 11.5, marginTop: 5, color: ok ? 'var(--c-good)' : 'var(--c-bad)' }}>
                      {ok ? '✓' : '✗'} exact grad = {fmtS(b1Truth.grads[k], 2)} → update{' '}
                      <strong>
                        {b1Truth.updateSigns[k] === '+'
                          ? '+ (increase)'
                          : b1Truth.updateSigns[k] === '-'
                            ? '− (decrease)'
                            : '0 (unchanged)'}
                      </strong>
                    </div>
                  )}
                </div>
              );
            })}

            <div
              style={{
                padding: '8px 10px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--bg-sunken)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                Optional goal: which weight has the largest <span className="mono">|grad|</span>?
              </div>
              <div className="btn-row">
                {BRIEF1_PROBES.map((k) => (
                  <button
                    key={k}
                    type="button"
                    className={`btn small ${st.b1Largest === k ? 'primary' : ''}`}
                    onClick={() => setSt((s) => ({ ...s, b1Largest: k, b1Checked: false }))}
                    data-largest={k}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>

            <div className="btn-row">
              <button
                type="button"
                className="btn primary"
                onClick={runB1Check}
                disabled={BRIEF1_PROBES.some((k) => st.b1Guesses[k] === null)}
              >
                Check &amp; run backprop
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ===================== BRIEF 2 ===================== */}
      {st.active === 1 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="controls inline" style={{ gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <Segmented
              options={BRIEF2_INIT_MODES.map((m) => ({ value: m, label: INIT_LABELS[m] }))}
              value={st.b2.initMode}
              onChange={(initMode) => updateB2({ initMode })}
            />
            <Segmented
              options={BRIEF2_ACTS.map((a) => ({ value: a, label: ACT_LABELS[a] }))}
              value={st.b2.act}
              onChange={(act) => updateB2({ act })}
            />
            <Segmented
              options={BRIEF2_LRS.map((lr) => ({ value: String(lr), label: `η=${lr}` }))}
              value={String(st.b2.lr)}
              onChange={(v) => updateB2({ lr: Number(v) })}
            />
          </div>

          <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            {/* Left: 5-seed loss trajectories */}
            <div style={{ flex: '1 1 340px', minWidth: 270 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
                <span>Loss across 5 fixed seeds (0 → {BRIEF2_STEPS} steps)</span>
                <span className="mono">
                  solved: <strong style={{ color: b2Res.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{b2Res.solvedSeeds}/5</strong>
                </span>
              </div>
              <svg
                viewBox="0 0 380 200"
                style={{
                  width: '100%',
                  height: 'auto',
                  display: 'block',
                  background: 'var(--bg-sunken)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                }}
                role="img"
                aria-label="Training loss curves across 5 fixed seeds on XOR"
              >
                {/* Target loss line at 0.08 */}
                {(() => {
                  const padL = 38;
                  const padR = 14;
                  const padT = 14;
                  const padB = 26;
                  const W = 380 - padL - padR;
                  const H = 200 - padT - padB;
                  const maxL = 1.0;
                  const sx = (idx: number, len: number) => padL + (idx / Math.max(1, len - 1)) * W;
                  const sy = (loss: number) => padT + (1 - Math.min(maxL, Math.max(0, loss)) / maxL) * H;
                  const targetY = sy(BRIEF2_TARGET_LOSS);
                  const seedColors = ['var(--c-a)', 'var(--c-b)', 'var(--c-c)', 'var(--c-grad)', 'var(--accent)'];
                  return (
                    <g>
                      {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
                        <g key={tick}>
                          <line x1={padL} x2={380 - padR} y1={sy(tick)} y2={sy(tick)} stroke="var(--c-grid)" strokeWidth={1} />
                          <text x={padL - 5} y={sy(tick) + 3.5} textAnchor="end" fontSize={10} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                            {tick.toFixed(2)}
                          </text>
                        </g>
                      ))}
                      <line x1={padL} x2={380 - padR} y1={targetY} y2={targetY} stroke="var(--c-good)" strokeWidth={1.4} strokeDasharray="5 3" />
                      <text x={380 - padR - 2} y={targetY - 4} textAnchor="end" fontSize={9.5} fill="var(--c-good)" style={{ fontFamily: 'var(--font-mono)' }}>
                        target ≤ {BRIEF2_TARGET_LOSS}
                      </text>
                      {b2Res.seeds.map((sr, si) => {
                        const d = sr.lossCurve
                          .map((v, idx) => `${idx === 0 ? 'M' : 'L'}${sx(idx, sr.lossCurve.length).toFixed(1)},${sy(v).toFixed(1)}`)
                          .join(' ');
                        return <path key={sr.seed} d={d} fill="none" stroke={seedColors[si]} strokeWidth={2} strokeOpacity={0.9} />;
                      })}
                      <text x={padL} y={194} fontSize={10} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>step 0</text>
                      <text x={380 - padR} y={194} textAnchor="end" fontSize={10} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>step {BRIEF2_STEPS}</text>
                    </g>
                  );
                })()}
              </svg>

              {/* Seed pills */}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                {b2Res.seeds.map((sr) => (
                  <span
                    key={sr.seed}
                    className="mono"
                    style={{
                      fontSize: 11.5,
                      padding: '3px 7px',
                      borderRadius: 5,
                      border: `1px solid ${sr.solvedStep !== null ? 'var(--c-good)' : 'var(--border-strong)'}`,
                      background: 'var(--bg-sunken)',
                      color: sr.solvedStep !== null ? 'var(--c-good)' : 'var(--text-muted)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    s{sr.seed}: {sr.solvedStep !== null ? `✓ @${sr.solvedStep}` : `L=${sr.finalLoss.toFixed(2)} (${(sr.finalAcc * 100).toFixed(0)}%)`}
                  </span>
                ))}
              </div>
            </div>

            {/* Right: Step-0 hidden gradient probe */}
            <div
              style={{
                flex: '1 1 250px',
                minWidth: 240,
                padding: '10px 12px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--bg-sunken)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ fontSize: 12.5, fontWeight: 700, marginBottom: 6 }}>
                Step-0 hidden gradient probe (seed 1)
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {b2Res.seeds[0].initHiddenGradNorms.map((gn, j) => {
                  const maxG = Math.max(0.25, ...b2Res.seeds[0].initHiddenGradNorms);
                  const pct = (gn / maxG) * 100;
                  return (
                    <div key={j} style={{ display: 'grid', gridTemplateColumns: '34px 1fr 64px', gap: 8, alignItems: 'center', fontSize: 12 }}>
                      <span className="mono">h{j + 1}</span>
                      <div style={{ height: 12, background: 'var(--bg-elev)', borderRadius: 3, overflow: 'hidden' }}>
                        <div
                          style={{
                            width: `${gn > 0 ? Math.max(3, pct) : 0}%`,
                            height: '100%',
                            background: gn < 1e-4 ? 'var(--c-bad)' : 'var(--c-grad)',
                          }}
                        />
                      </div>
                      <span className="mono" style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                        {gn.toFixed(4)}
                      </span>
                    </div>
                  );
                })}
              </div>
              <div
                className="mono"
                style={{
                  marginTop: 10,
                  paddingTop: 8,
                  borderTop: '1px solid var(--border)',
                  fontSize: 12,
                  color: 'var(--text-muted)',
                }}
              >
                Final |w(h₁) − w(h₂)| ={' '}
                <strong style={{ color: b2Res.seeds[0].finalTwinGap < 1e-6 ? 'var(--c-bad)' : 'var(--text)' }}>
                  {b2Res.seeds[0].finalTwinGap.toFixed(4)}
                </strong>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===================== BRIEF 3 ===================== */}
      {st.active === 2 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="controls inline" style={{ gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <div className="btn-row" role="group" aria-label="Freeze or unfreeze units">
              <button
                type="button"
                className={`btn small ${st.b3.trainH1 ? 'primary' : ''}`}
                onClick={() => updateB3({ trainH1: !st.b3.trainH1 })}
                data-toggle="h1"
              >
                h₁: {st.b3.trainH1 ? 'Trainable' : 'Frozen'}
              </button>
              <button
                type="button"
                className={`btn small ${st.b3.trainH2 ? 'primary' : ''}`}
                onClick={() => updateB3({ trainH2: !st.b3.trainH2 })}
                data-toggle="h2"
              >
                h₂: {st.b3.trainH2 ? 'Trainable' : 'Frozen'}
              </button>
              <button
                type="button"
                className={`btn small ${st.b3.trainOut ? 'primary' : ''}`}
                onClick={() => updateB3({ trainOut: !st.b3.trainOut })}
                data-toggle="out"
              >
                out: {st.b3.trainOut ? 'Trainable' : 'Frozen'}
              </button>
            </div>

            <Segmented
              options={BRIEF3_LRS.map((lr) => ({ value: String(lr), label: `η=${lr}` }))}
              value={String(st.b3.lr)}
              onChange={(v) => updateB3({ lr: Number(v) })}
            />
          </div>

          <ReplayControls r={b3Replay} label={`snap (step ${b3Snap.step})`} />

          {/* Dual side-by-side geometry: Input space (x1, x2) vs Hidden space (h1, h2) */}
          <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            {/* Input space */}
            <div style={{ flex: '1 1 280px', maxWidth: 330 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <strong>Input space (x₁, x₂)</strong>
                <span className="mono" style={{ color: 'var(--text-muted)' }}>
                  loss {b3Snap.loss.toFixed(3)} · {(b3Snap.acc * 100).toFixed(0)}%
                </span>
              </div>
              <Plot
                width={310}
                height={310}
                x={[-1.5, 1.5]}
                y={[-1.5, 1.5]}
                xLabel="x₁"
                yLabel="x₂"
                ariaLabel="Input space decision boundary for Brief 3"
                style={{ width: '100%', height: 'auto', display: 'block' }}
              >
                <Heatmap f={b3Predict} range={[0, 1]} colormap={diverging} resolution={76} opacity={0.44} />
                {({ x, y }) => (
                  <g>
                    {b3H1Line && (
                      <line
                        x1={x(b3H1Line.x1)}
                        y1={y(b3H1Line.y1)}
                        x2={x(b3H1Line.x2)}
                        y2={y(b3H1Line.y2)}
                        stroke="var(--c-a)"
                        strokeWidth={2.2}
                        strokeDasharray={st.b3.trainH1 ? undefined : '5 4'}
                      />
                    )}
                    {b3H2Line && (
                      <line
                        x1={x(b3H2Line.x1)}
                        y1={y(b3H2Line.y1)}
                        x2={x(b3H2Line.x2)}
                        y2={y(b3H2Line.y2)}
                        stroke="var(--c-c)"
                        strokeWidth={2.2}
                        strokeDasharray={st.b3.trainH2 ? undefined : '5 4'}
                      />
                    )}
                    {CREDIT_XOR.map(([x1, x2, label], idx) => (
                      <circle
                        key={idx}
                        cx={x(x1)}
                        cy={y(x2)}
                        r={7}
                        fill={label === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                        stroke="var(--bg-elev)"
                        strokeWidth={2}
                      />
                    ))}
                  </g>
                )}
              </Plot>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }}>
                Solid line = trainable hidden unit; dashed = frozen in place.
              </div>
            </div>

            {/* Exact 2D Hidden space (h1, h2) */}
            <div style={{ flex: '1 1 280px', maxWidth: 330 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4, gap: 6 }}>
                <strong>Hidden space (h₁, h₂)</strong>
                <span
                  className="mono"
                  style={{ fontWeight: 700, color: b3Snap.separable ? 'var(--c-good)' : 'var(--c-bad)' }}
                >
                  {b3Snap.separable ? '✓ Untangled' : '✗ Tangled (X crosses)'}
                </span>
              </div>
              <Plot
                width={310}
                height={310}
                x={[-1.15, 1.15]}
                y={[-1.15, 1.15]}
                xLabel="h₁"
                yLabel="h₂"
                ariaLabel="2D hidden representation (h1, h2) showing XOR untangling"
                style={{ width: '100%', height: 'auto', display: 'block' }}
              >
                {({ x, y }) => {
                  const pts = b3Snap.hiddenCoords;
                  return (
                    <g>
                      {/* Class-0 segment (P0-P3) and Class-1 segment (P1-P2) to make X-crossing unmistakable */}
                      <line
                        x1={x(pts[0][0])}
                        y1={y(pts[0][1])}
                        x2={x(pts[3][0])}
                        y2={y(pts[3][1])}
                        stroke="var(--c-b)"
                        strokeWidth={1.6}
                        strokeDasharray="4 3"
                        strokeOpacity={0.7}
                      />
                      <line
                        x1={x(pts[1][0])}
                        y1={y(pts[1][1])}
                        x2={x(pts[2][0])}
                        y2={y(pts[2][1])}
                        stroke="var(--c-a)"
                        strokeWidth={1.6}
                        strokeDasharray="4 3"
                        strokeOpacity={0.7}
                      />

                      {/* Output layer's straight separating line v1*h1 + v2*h2 + c = 0 */}
                      {b3OutLine && (
                        <line
                          x1={x(b3OutLine.x1)}
                          y1={y(b3OutLine.y1)}
                          x2={x(b3OutLine.x2)}
                          y2={y(b3OutLine.y2)}
                          stroke="var(--c-grad)"
                          strokeWidth={2.6}
                          strokeDasharray={st.b3.trainOut ? undefined : '5 4'}
                        />
                      )}

                      {/* 4 XOR points in (h1, h2) space */}
                      {pts.map(([h1, h2], idx) => {
                        const label = CREDIT_XOR[idx][2];
                        return (
                          <g key={idx}>
                            <circle
                              cx={x(h1)}
                              cy={y(h2)}
                              r={7.5}
                              fill={label === 1 ? 'var(--c-a)' : 'var(--c-b)'}
                              stroke="var(--bg-elev)"
                              strokeWidth={2}
                            />
                            <text
                              x={x(h1)}
                              y={y(h2) - 10}
                              textAnchor="middle"
                              fontSize={10}
                              fontWeight={700}
                              fill="var(--text)"
                              style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-elev)', strokeWidth: 3.5 }}
                            >
                              P{idx}
                            </text>
                          </g>
                        );
                      })}
                    </g>
                  );
                }}
              </Plot>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }}>
                The magenta line is the output neuron (<span className="mono">v₁h₁ + v₂h₂ + c = 0</span>). While the
                dashed segments cross, no straight line can separate them.
              </div>
            </div>
          </div>
        </div>
      )}
    </LabShell>
  );
}
