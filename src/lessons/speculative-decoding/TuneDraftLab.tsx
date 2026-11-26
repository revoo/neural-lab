// authored-by: Gemini Antigravity
/**
 * Lab 2: Tune the draft — speculative decoding configuration lab on LabShell.
 * Uses the exhaustive 48-cell grid measured on our real Grimm tiny GPTs (`specResults.ts`)
 * plus live in-browser verification and CPU stopwatch measurements via `useSpecModels()`.
 */
import { useEffect, useMemo, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Segmented } from '../../components/controls';
import { SPEC_BRIEFS, specBriefSummary, specCellFor, specHint } from './lab';
import { vllmSpecCommand } from './realCode';
import { BigramDrafter, GptDrafter, runPlainTarget, runSpeculative, type DrafterId } from './spec';
import { SPEC_GRID, SPEC_PROMPTS } from './specResults';
import { useSpecModels } from './SpecWidget';

const SETTING_KEY = 'llm-serving-draft-cfg-v1';
const IMPOSSIBLE_KEY = 'llm-serving-draft-imp-v1';

function loadDraftSetting(): { drafter: DrafterId; k: number } {
  try {
    const raw = JSON.parse(localStorage.getItem(SETTING_KEY) ?? 'null');
    if (raw && (raw.drafter === 'bigram' || raw.drafter === 'small-gpt') && raw.k >= 1 && raw.k <= 8) {
      return raw;
    }
  } catch {
    /* ignore */
  }
  return { drafter: 'small-gpt', k: 1 };
}

export function TuneDraftLab() {
  const progress = useLabProgress('llm-serving-draft-v1');
  const log = useMachineLog();
  const { models } = useSpecModels();

  const [active, setActive] = useState(0);
  const brief = SPEC_BRIEFS[active];

  const [setting, setSetting] = useState<{ drafter: DrafterId; k: number }>(loadDraftSetting);
  const [declaredImpossible, setDeclaredImpossible] = useState<boolean>(() => {
    try {
      return localStorage.getItem(IMPOSSIBLE_KEY) === '1';
    } catch {
      return false;
    }
  });
  const [liveCheck, setLiveCheck] = useState<{
    key: string;
    exactGreedy: boolean | null;
    specMs: number;
    plainMs: number;
    cpuRatio: number;
    emitted: number;
    passes: number;
  } | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(SETTING_KEY, JSON.stringify(setting));
    } catch {
      /* ignore */
    }
  }, [setting]);

  const cell = specCellFor(setting.drafter, brief.temperature, setting.k);
  const summary = useMemo(() => specBriefSummary(brief), [brief]);
  const curveCells = useMemo(
    () => SPEC_GRID.filter((c) => c.drafter === setting.drafter && c.temperature === brief.temperature),
    [setting.drafter, brief.temperature],
  );

  const curKey = `${brief.id}|${setting.drafter}|${setting.k}`;
  const liveForCur = liveCheck && liveCheck.key === curKey ? liveCheck : null;

  const passing = brief.impossible
    ? declaredImpossible
    : cell.modelledSpeedup >= brief.targetSpeedup - 1e-9;
  const optimal = brief.impossible
    ? declaredImpossible
    : passing && cell.modelledSpeedup >= summary.bestSpeedup - 0.005;

  const runLiveStopwatch = () => {
    if (!models) return;
    // Measure across 4 held-out tale openings (144 generated characters) so browser timing is crisp
    const demoPrompts = [SPEC_PROMPTS[0], SPEC_PROMPTS[3], SPEC_PROMPTS[6], SPEC_PROMPTS[9]];
    let specMs = 0;
    let plainMs = 0;
    let exactAll = true;
    let emitted = 0;
    let passes = 0;
    for (const p of demoPrompts) {
      const dr =
        setting.drafter === 'bigram'
          ? new BigramDrafter(models.bigramLogProbs)
          : new GptDrafter(models.draft);
      const sp = runSpeculative({
        targetModel: models.target,
        drafter: dr,
        prompt: p.prompt,
        maxNewTokens: 36,
        k: setting.k,
        temperature: brief.temperature,
        seed: 11,
      });
      const pl = runPlainTarget({
        targetModel: models.target,
        prompt: p.prompt,
        maxNewTokens: 36,
        temperature: brief.temperature,
        seed: 11,
      });
      specMs += sp.wallMs;
      plainMs += pl.wallMs;
      emitted += sp.totalEmitted;
      passes += sp.targetPasses;
      if (brief.temperature === 0 && sp.generatedText !== pl.generatedText) exactAll = false;
    }
    const cpuRatio = plainMs / Math.max(0.05, specMs);
    setLiveCheck({
      key: curKey,
      exactGreedy: brief.temperature === 0 ? exactAll : null,
      specMs,
      plainMs,
      cpuRatio,
      emitted,
      passes,
    });
    log.push(
      `[${brief.tab}] ${setting.drafter}, k=${setting.k}: ${emitted} chars in ${passes} passes (${(emitted / passes).toFixed(2)}/pass), browser CPU ${specMs.toFixed(1)} ms vs plain ${plainMs.toFixed(1)} ms (${cpuRatio.toFixed(2)}×)${brief.temperature === 0 ? (exactAll ? ', greedy exact ✓' : ', MISMATCH') : ''}`,
      brief.impossible ? (cpuRatio < 1 ? 'info' : 'bad') : passing ? 'good' : 'info',
    );
  };

  const declareImpossible = () => {
    setDeclaredImpossible(true);
    try {
      localStorage.setItem(IMPOSSIBLE_KEY, '1');
    } catch {
      /* ignore */
    }
    log.push(
      `[Brief 3] Declared impossible on this CPU: exhaustive Node measurement peaks at ${summary.bestSpeedup.toFixed(2)}× (< 1.00×) across all 48 cells because a 116 KiB model in CPU cache is compute-bound, not memory-bound.`,
      'good',
    );
  };

  const briefs: LabBrief[] = SPEC_BRIEFS.map((b) => ({
    id: b.id,
    tab: b.tab,
    title: b.title,
    text: (
      <>
        {b.description} <strong>Optional goal:</strong> {b.optGoal}
      </>
    ),
  }));

  const statusText = brief.impossible
    ? declaredImpossible
      ? `Confirmed impossible on this CPU! Across all 48 (drafter × k × temperature) cells, the fastest CPU wall-clock ratio is ${summary.bestSpeedup.toFixed(2)}× — speculation only wins when target verification is memory-bound.`
      : `Current setting (${setting.drafter}, k = ${setting.k}) runs at ${cell.cpuWallSpeedup.toFixed(2)}× real CPU speed in Node${liveForCur ? ` (${liveForCur.cpuRatio.toFixed(2)}× on your browser just now)` : ''}. Try other settings with “Measure on my CPU”, or click “Declare impossible” once you see why none can beat 1.00×.`
    : passing
      ? optimal
        ? `Optimal! ${setting.drafter} at k = ${setting.k} reaches ${cell.modelledSpeedup.toFixed(2)}× modelled speed-up (best of all 16 settings at T = ${brief.temperature}).`
        : `Brief passed! Modelled speed-up is ${cell.modelledSpeedup.toFixed(2)}× (target ≥ ${brief.targetSpeedup.toFixed(2)}×). Optional goal: reach the exhaustive best of ${summary.bestSpeedup.toFixed(2)}×.`
      : `Not yet passing: modelled speed-up is ${cell.modelledSpeedup.toFixed(2)}× (need ≥ ${brief.targetSpeedup.toFixed(2)}×). Measured E = ${cell.tokensPerPass.toFixed(2)} chars/pass, draft cost k·c = ${(setting.k * cell.costRatio).toFixed(2)}.`;

  // Chart dimensions for E(k) vs i.i.d. formula & modelled speed-up
  const maxE = Math.max(4.6, ...curveCells.map((c) => Math.max(c.tokensPerPass, c.formulaTokensPerPass)));
  const maxSpd = Math.max(2.6, ...curveCells.map((c) => c.modelledSpeedup));

  return (
    <LabShell
      title="Lab · Tune the Draft"
      caption="Measured across 12 held-out Grimm’s Fairy Tales prompts (real 29.7k target GPT, 5.4k draft GPT, and corpus bigram table). Production speed-up uses the RTX 4090 roofline (Llama-3 8B FP16 target vs 1B-class FP16 draft, c = 0.127, or bigram lookup, c = 0.01)."
      briefs={briefs}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={statusText}
      statusTone={passing ? 'good' : 'neutral'}
      hint={specHint(brief, setting.drafter, setting.k)}
      reveal={{
        label: brief.impossible
          ? 'all 48 CPU wall-clock ratios'
          : `all 16 settings at T = ${brief.temperature}`,
        content: (
          <div>
            <p style={{ margin: '0 0 8px', fontSize: 13 }}>
              {brief.impossible
                ? `Exhaustive check over all 48 (drafter × k × temperature) cells in Node: 0 of 48 beat plain decoding on CPU (range 0.34× to ${summary.bestSpeedup.toFixed(2)}×). Because the 116 KiB target sits inside CPU cache and verification is compute-bound, checking k tokens costs roughly k× one token.`
                : `${summary.passingCount} of 16 settings reach ≥ ${brief.targetSpeedup.toFixed(2)}× at T = ${brief.temperature}. Best: ${summary.bestCells.map((c) => `${c.drafter} at k = ${c.k} (${c.modelledSpeedup.toFixed(2)}×)`).join(', ')}. Notice the twist: even though the 5.4k GPT has ~15–20% higher acceptance than the bigram table, charging it a real 1B-scale draft cost (c = 0.127) lets the nearly-free bigram table (c = 0.01) edge ahead!`}
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table className="num-table">
                <thead>
                  <tr>
                    <th>Drafter</th>
                    <th>Lookahead</th>
                    <th>Acceptance</th>
                    <th>Measured E</th>
                    <th>Formula E</th>
                    <th>GPU speedup</th>
                    <th>CPU clock</th>
                  </tr>
                </thead>
                <tbody>
                  {SPEC_GRID.filter((c) => c.temperature === brief.temperature).map((c) => {
                    const isBest = !brief.impossible && Math.abs(c.modelledSpeedup - summary.bestSpeedup) < 0.005;
                    const isCur = c.drafter === setting.drafter && c.k === setting.k;
                    return (
                      <tr key={`${c.drafter}-${c.k}`} style={{ background: isCur ? 'var(--accent-soft)' : undefined }}>
                        <td>{c.drafter === 'bigram' ? 'Bigram' : 'Small GPT'}</td>
                        <td className="mono">{c.k}</td>
                        <td className="mono">{(c.alpha * 100).toFixed(1)}%</td>
                        <td className="mono">{c.tokensPerPass.toFixed(2)}</td>
                        <td className="mono">{c.formulaTokensPerPass.toFixed(2)}</td>
                        <td className="mono" style={{ color: isBest ? 'var(--c-good)' : undefined, fontWeight: isBest ? 700 : 400 }}>
                          {c.modelledSpeedup.toFixed(2)}×
                        </td>
                        <td className="mono" style={{ color: 'var(--c-warn)' }}>{c.cpuWallSpeedup.toFixed(2)}×</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear, title: 'Draft tuning log', empty: 'Press “Measure live on my CPU” to run the real models in your browser.' }}
      actions={
        <>
          <button type="button" className="btn primary" disabled={!models} onClick={runLiveStopwatch}>
            {models ? 'Measure live on my CPU' : 'Loading models…'}
          </button>
          {brief.impossible && !declaredImpossible && (
            <button type="button" className="btn" onClick={declareImpossible}>
              Declare impossible on this CPU
            </button>
          )}
        </>
      }
    >
      {/* Controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 14 }}>
        <Segmented
          label="Draft model"
          value={setting.drafter}
          onChange={(v) => setSetting((s) => ({ ...s, drafter: v as DrafterId }))}
          options={[
            { value: 'small-gpt', label: 'Small GPT (5.4k · c = 0.127)' },
            { value: 'bigram', label: 'Bigram table (c = 0.01)' },
          ]}
        />
        <Segmented
          label="Lookahead tokens"
          value={String(setting.k)}
          onChange={(v) => setSetting((s) => ({ ...s, k: Number(v) }))}
          options={[1, 2, 3, 4, 5, 6, 7, 8].map((n) => ({ value: String(n), label: String(n) }))}
        />
      </div>

      {/* KPI cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(135px, 100%), 1fr))', gap: 8, marginBottom: 12 }}>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>GPU speedup</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: !brief.impossible && passing ? 'var(--c-good)' : 'var(--c-a)' }}>
            {cell.modelledSpeedup.toFixed(2)}×
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            E / (k·c + 1)
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Measured E</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-c)' }}>
            {cell.tokensPerPass.toFixed(2)}
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            i.i.d. formula: {cell.formulaTokensPerPass.toFixed(2)}
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Draft cost tax</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-b)' }}>
            +{(setting.k * cell.costRatio * 100).toFixed(0)}%
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {setting.k} × {cell.costRatio.toFixed(3)}
          </div>
        </div>
        <div style={{ padding: '7px 9px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>CPU wall clock</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-warn)' }}>
            {liveForCur ? `${liveForCur.cpuRatio.toFixed(2)}×` : `${cell.cpuWallSpeedup.toFixed(2)}×`}
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {liveForCur ? 'measured live' : 'Node baseline'}
          </div>
        </div>
      </div>

      {/* Charts: Measured vs Formula E(k) + Speedup over k */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 12, marginBottom: 12 }}>
        {/* Left: E(k) measured vs i.i.d. formula */}
        <div style={{ padding: 10, background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
            Tokens per target pass: measured (solid) vs i.i.d. formula (dashed)
          </div>
          <svg viewBox="0 0 340 155" style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="Measured vs formula tokens per pass">
            <line x1={34} y1={16} x2={34} y2={126} stroke="var(--c-axis)" />
            <line x1={34} y1={126} x2={326} y2={126} stroke="var(--c-axis)" />
            {(() => {
              const xOf = (kv: number) => 46 + ((kv - 1) / 7) * 268;
              const yOf = (val: number) => 126 - (val / maxE) * 106;
              return (
                <>
                  <polyline
                    fill="none"
                    stroke="var(--text-muted)"
                    strokeDasharray="4 3"
                    strokeWidth={1.6}
                    points={curveCells.map((c) => `${xOf(c.k).toFixed(1)},${yOf(c.formulaTokensPerPass).toFixed(1)}`).join(' ')}
                  />
                  <polyline
                    fill="none"
                    stroke="var(--c-c)"
                    strokeWidth={2.2}
                    points={curveCells.map((c) => `${xOf(c.k).toFixed(1)},${yOf(c.tokensPerPass).toFixed(1)}`).join(' ')}
                  />
                  {curveCells.map((c) => {
                    const activeDot = c.k === setting.k;
                    return (
                      <g key={c.k}>
                        <circle
                          cx={xOf(c.k)}
                          cy={yOf(c.tokensPerPass)}
                          r={activeDot ? 5 : 3}
                          fill={activeDot ? 'var(--accent)' : 'var(--c-c)'}
                          stroke="var(--bg-elev)"
                          strokeWidth={1.4}
                        />
                        <text x={xOf(c.k)} y={142} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill={activeDot ? 'var(--accent-text)' : 'var(--text-muted)'}>
                          {c.k}
                        </text>
                      </g>
                    );
                  })}
                  <text x={30} y={22} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">{maxE.toFixed(1)}</text>
                  <text x={30} y={129} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">0</text>
                </>
              );
            })()}
          </svg>
        </div>

        {/* Right: Modelled GPU speedup vs k + word-start vs mid-word bars */}
        <div style={{ padding: 10, background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
            Modelled RTX 4090 speedup over lookahead (dashed = target)
          </div>
          <svg viewBox="0 0 340 155" style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label="Modelled GPU speedup across lookahead values">
            <line x1={34} y1={16} x2={34} y2={126} stroke="var(--c-axis)" />
            <line x1={34} y1={126} x2={326} y2={126} stroke="var(--c-axis)" />
            {(() => {
              const xOf = (kv: number) => 46 + ((kv - 1) / 7) * 268;
              const yOf = (val: number) => 126 - (val / maxSpd) * 106;
              const yTarget = yOf(brief.impossible ? 1.0 : brief.targetSpeedup);
              return (
                <>
                  <line x1={34} y1={yTarget} x2={326} y2={yTarget} stroke="var(--c-warn)" strokeDasharray="4 3" strokeWidth={1.3} />
                  <text x={324} y={yTarget - 4} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--c-warn)">
                    {brief.impossible ? '1.00× break-even' : `${brief.targetSpeedup.toFixed(2)}× target`}
                  </text>
                  <polyline
                    fill="none"
                    stroke="var(--c-a)"
                    strokeWidth={2.2}
                    points={curveCells.map((c) => `${xOf(c.k).toFixed(1)},${yOf(c.modelledSpeedup).toFixed(1)}`).join(' ')}
                  />
                  {curveCells.map((c) => {
                    const activeDot = c.k === setting.k;
                    return (
                      <g key={c.k}>
                        <circle
                          cx={xOf(c.k)}
                          cy={yOf(c.modelledSpeedup)}
                          r={activeDot ? 5 : 3}
                          fill={activeDot ? 'var(--accent)' : 'var(--c-a)'}
                          stroke="var(--bg-elev)"
                          strokeWidth={1.4}
                        />
                        <text x={xOf(c.k)} y={142} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill={activeDot ? 'var(--accent-text)' : 'var(--text-muted)'}>
                          {c.k}
                        </text>
                      </g>
                    );
                  })}
                  <text x={30} y={22} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">{maxSpd.toFixed(1)}×</text>
                  <text x={30} y={129} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-muted)">0</text>
                </>
              );
            })()}
          </svg>
        </div>
      </div>

      {/* Position acceptance breakdown: word starts vs mid-word */}
      <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', marginBottom: 12, fontSize: 12 }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>
          Why measured E differs from the i.i.d. formula: acceptance by position
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(220px, 100%), 1fr))', gap: 10 }}>
          {[
            { label: 'Word starts (after space/punct)', val: cell.alphaWordStart, col: 'var(--c-b)' },
            { label: 'Mid-word (inside a word)', val: cell.alphaMidWord, col: 'var(--c-c)' },
          ].map((row) => (
            <div key={row.label}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, marginBottom: 3 }}>
                <span>{row.label}</span>
                <strong className="mono">{(row.val * 100).toFixed(1)}%</strong>
              </div>
              <div style={{ height: 9, background: 'var(--bg-elev)', borderRadius: 4, overflow: 'hidden', border: '1px solid var(--border)' }}>
                <div style={{ width: `${row.val * 100}%`, height: '100%', background: row.col }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Live vLLM speculative config */}
      <LiveCode
        code={vllmSpecCommand(setting.drafter, setting.k, brief.temperature)}
        note="vLLM --speculative-config JSON verified against docs.vllm.ai/en/stable (Nov 2026)"
      />
    </LabShell>
  );
}
