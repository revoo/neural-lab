// authored-by: Gemini Antigravity
/**
 * Widget E: SpeculativeStepper — step through live speculative decoding rounds on our real
 * trained Grimm's Fairy Tales character models (`tinyGpt.cpu.bin` target + `draftGpt.bin` or
 * corpus bigram drafter). Also exports `useSpecModels()` for Lab 2 (TuneDraftLab).
 */
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import { Gpt } from '../../lib/compute/gpt';
import { loadGrimm } from '../../lib/corpora/grimm';
import targetBinUrl from '../../lib/compute/tinyGpt.cpu.bin?url';
import draftBinUrl from './draftGpt.bin?url';
import {
  BigramDrafter,
  DRAFT_CFG,
  GptDrafter,
  TARGET_CFG,
  buildBigramLogProbs,
  modelledSpeedup,
  runPlainTarget,
  runSpeculative,
  type DrafterId,
} from './spec';
import { SPEC_PROMPTS } from './specResults';

export interface LoadedSpecModels {
  target: Gpt;
  draft: Gpt;
  bigramLogProbs: Float64Array;
}

let cachedPromise: Promise<LoadedSpecModels> | null = null;

export function loadSpecModels(): Promise<LoadedSpecModels> {
  cachedPromise ??= Promise.all([
    fetch(targetBinUrl).then((r) => r.arrayBuffer()),
    fetch(draftBinUrl).then((r) => r.arrayBuffer()),
    loadGrimm(),
  ]).then(([tBuf, dBuf, corpus]) => {
    const target = new Gpt(TARGET_CFG);
    target.load(new Float32Array(tBuf));
    const draft = new Gpt(DRAFT_CFG);
    draft.load(new Float32Array(dBuf));
    const bigramLogProbs = buildBigramLogProbs(corpus.train);
    return { target, draft, bigramLogProbs };
  });
  return cachedPromise;
}

export function useSpecModels() {
  const [models, setModels] = useState<LoadedSpecModels | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    loadSpecModels()
      .then((m) => {
        if (alive) setModels(m);
      })
      .catch((e) => {
        if (alive) setError(String(e));
      });
    return () => {
      alive = false;
    };
  }, []);
  return { models, error };
}

export function showChar(ch: string): string {
  if (ch === ' ') return '␣';
  if (ch === '\n') return '↵';
  return ch;
}

const DEMO_PROMPT_INDICES = [0, 3, 6, 9] as const; // one opening from each held-out tale

export function SpeculativeStepper() {
  const { models, error } = useSpecModels();
  const [drafterId, setDrafterId] = useState<DrafterId>('small-gpt');
  const [k, setK] = useState(5);
  const [temperature, setTemperature] = useState<0 | 0.7 | 1.0>(0);
  const [promptSlot, setPromptSlot] = useState(0);
  const [seed, setSeed] = useState(11);

  const promptEntry = SPEC_PROMPTS[DEMO_PROMPT_INDICES[promptSlot]];

  const runData = useMemo(() => {
    if (!models) return null;
    const drafter =
      drafterId === 'bigram' ? new BigramDrafter(models.bigramLogProbs) : new GptDrafter(models.draft);
    const spec = runSpeculative({
      targetModel: models.target,
      drafter,
      prompt: promptEntry.prompt,
      maxNewTokens: 36,
      k,
      temperature,
      seed,
    });
    const plain = runPlainTarget({
      targetModel: models.target,
      prompt: promptEntry.prompt,
      maxNewTokens: 36,
      temperature,
      seed,
    });
    const prod = modelledSpeedup(spec.tokensPerTargetPass, drafterId, k);
    return { spec, plain, prod };
  }, [models, drafterId, k, temperature, promptEntry.prompt, seed]);

  const totalRounds = Math.max(0, (runData?.spec.rounds.length ?? 1) - 1);
  const resetKey = `${drafterId}|${k}|${temperature}|${promptSlot}|${seed}`;
  const replay = useReplay(totalRounds, 700, resetKey, false);
  const round = runData?.spec.rounds[Math.min(replay.i, totalRounds)] ?? null;

  const greedyMatches =
    runData && temperature === 0 ? runData.spec.generatedText === runData.plain.generatedText : null;
  const cpuRatio = runData ? runData.plain.wallMs / Math.max(0.05, runData.spec.wallMs) : 0;

  return (
    <Widget
      title="Speculative decoding stepper: draft, verify in one pass, and correct"
      wide
      caption="Runs two real character-level models trained on Grimm’s Fairy Tales: a 29,696-parameter target GPT (from the capstone) paired with either a 5,424-parameter draft GPT or a zero-cost bigram table. Step through rounds to inspect q(x), p(x), the accept test u < p/q, and the residual correction distribution max(0, p − q)."
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Draft model"
          value={drafterId}
          onChange={(v) => setDrafterId(v as DrafterId)}
          options={[
            { value: 'small-gpt', label: 'Small GPT (5.4k)' },
            { value: 'bigram', label: 'Bigram table' },
          ]}
        />
        <Segmented
          label="Lookahead tokens"
          value={String(k)}
          onChange={(v) => setK(Number(v))}
          options={[1, 2, 3, 4, 5, 6, 7, 8].map((n) => ({ value: String(n), label: String(n) }))}
        />
        <Segmented
          label="Temperature"
          value={String(temperature)}
          onChange={(v) => setTemperature(Number(v) as 0 | 0.7 | 1.0)}
          options={[
            { value: '0', label: '0 (greedy)' },
            { value: '0.7', label: '0.7' },
            { value: '1', label: '1.0' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Held-out prompt:</span>
        {DEMO_PROMPT_INDICES.map((pIdx, i) => (
          <button
            key={pIdx}
            type="button"
            className={`btn small ${promptSlot === i ? 'primary' : ''}`}
            onClick={() => setPromptSlot(i)}
          >
            {SPEC_PROMPTS[pIdx].title.split(' ')[0]}
          </button>
        ))}
        {temperature > 0 && (
          <button type="button" className="btn small" onClick={() => setSeed((s) => s + 1)}>
            New seed (#{seed})
          </button>
        )}
      </div>

      {error && <div className="live-note tone-bad">Could not load model weights: {error}</div>}
      {!runData && !error && <div className="live-note">Loading real Grimm tiny-GPT weights…</div>}

      {runData && round && (
        <>
          <ReplayControls r={replay} label="round" />

          {/* Running text strip */}
          <div
            className="mono"
            style={{
              marginTop: 10,
              padding: '10px 12px',
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              fontSize: 13,
              lineHeight: 1.6,
              overflowX: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            <span style={{ color: 'var(--text-muted)' }}>{round.prefixBefore}</span>
            {round.emittedTokens.map((_, idx) => {
              const isCorr = idx === round.emittedTokens.length - 1 && (round.rejectedAt >= 0 || round.wasBonus);
              const ch = round.emittedText[idx];
              return (
                <span
                  key={idx}
                  style={{
                    background: isCorr ? 'var(--accent-soft)' : 'transparent',
                    color: isCorr ? 'var(--accent-text)' : 'var(--c-good)',
                    fontWeight: 700,
                    padding: '0 1px',
                    borderBottom: isCorr ? '2px solid var(--accent)' : '2px solid var(--c-good)',
                  }}
                >
                  {showChar(ch)}
                </span>
              );
            })}
          </div>

          {/* Round inspection cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))', gap: 14, marginTop: 12 }}>
            {/* Left: Drafted token accept/reject strip */}
            <div style={{ padding: 10, background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6 }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                Round {round.round + 1}: draft proposes up to {k} · target checks all in 1 pass
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {round.drafted.map((d) => (
                  <div
                    key={d.index}
                    style={{
                      flex: '1 1 78px',
                      maxWidth: 112,
                      padding: '6px 8px',
                      borderRadius: 6,
                      border: `1.5px solid ${d.accepted ? 'var(--c-good)' : 'var(--c-bad)'}`,
                      background: 'var(--bg-elev)',
                      fontSize: 11,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
                      <span className="mono" style={{ fontSize: 15, fontWeight: 700 }}>
                        {showChar(d.char)}
                      </span>
                      <span
                        className="mono"
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          color: d.accepted ? 'var(--c-good)' : 'var(--c-bad)',
                        }}
                      >
                        {d.accepted ? 'ACCEPT' : 'REJECT'}
                      </span>
                    </div>
                    <div className="mono" style={{ color: 'var(--text-muted)', fontSize: 10, lineHeight: 1.4 }}>
                      q = {(d.qProb * 100).toFixed(0)}%<br />
                      p = {(d.pProb * 100).toFixed(0)}%<br />
                      p/q: {(d.acceptProb * 100).toFixed(0)}%<br />
                      u = {d.u.toFixed(2)}
                    </div>
                    <div style={{ fontSize: 9.5, color: 'var(--text-faint)', marginTop: 2 }}>
                      {d.isWordStart ? 'word start' : 'mid-word'}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
                {round.rejectedAt >= 0
                  ? `Rejected at draft #${round.rejectedAt + 1}: rolled back ${round.rolledBack} KV ${round.rolledBack === 1 ? 'slot' : 'slots'} (decoder.len = ${round.prefixBefore.length + round.acceptedCount}), emitted ${round.emittedTokens.length} chars.`
                  : `All ${round.acceptedCount} accepted! Emitted ${round.emittedTokens.length} chars (including bonus token from p).`}
              </div>
            </div>

            {/* Right: Residual correction distribution norm(max(0, p - q)) or Bonus p */}
            <div style={{ padding: 10, background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 6 }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                {round.rejectedAt >= 0
                  ? `Residual correction norm(max(0, p − q)) → drew “${showChar(round.correctionChar)}”`
                  : `Bonus draw from target p → drew “${showChar(round.correctionChar)}”`}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {round.correctionDistTop.map((row) => {
                  const chosen = row.token === round.correctionToken;
                  return (
                    <div key={row.token} style={{ display: 'grid', gridTemplateColumns: '28px 1fr 48px', alignItems: 'center', gap: 8, fontSize: 11.5 }}>
                      <span className="mono" style={{ fontWeight: chosen ? 700 : 500, color: chosen ? 'var(--accent-text)' : 'var(--text)' }}>
                        {showChar(row.char)}
                      </span>
                      <div style={{ height: 12, background: 'var(--bg-elev)', borderRadius: 3, overflow: 'hidden', border: '1px solid var(--border)' }}>
                        <div
                          style={{
                            width: `${Math.max(2, row.prob * 100)}%`,
                            height: '100%',
                            background: chosen ? 'var(--accent)' : 'var(--c-a)',
                          }}
                        />
                      </div>
                      <span className="mono" style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
                        {(row.prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
              {round.rejectedAt >= 0 && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
                  Any character where q already exceeded p gets zero residual mass — keeping the exact output distribution equal to p.
                </div>
              )}
            </div>
          </div>

          {/* Running tallies + Honest CPU wall-clock readout */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(150px, 100%), 1fr))', gap: 10, marginTop: 12, marginBottom: 10 }}>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Chars / target pass</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-c)' }}>
                {runData.spec.tokensPerTargetPass.toFixed(2)}
              </div>
              <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                {runData.spec.totalEmitted} chars in {runData.spec.targetPasses} passes
              </div>
            </div>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Acceptance rate</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--c-a)' }}>
                {(runData.spec.acceptanceRate * 100).toFixed(1)}%
              </div>
              <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                {runData.spec.totalAccepted} / {runData.spec.totalDrafted} drafts
              </div>
            </div>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Modelled GPU gain</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: runData.prod.speedup >= 1 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {runData.prod.speedup.toFixed(2)}×
              </div>
              <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                RTX 4090 (c = {runData.prod.c.toFixed(3)})
              </div>
            </div>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Real CPU clock</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: cpuRatio >= 1 ? 'var(--c-good)' : 'var(--c-warn)' }}>
                {cpuRatio.toFixed(2)}×
              </div>
              <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
                {runData.spec.wallMs.toFixed(1)} ms vs {runData.plain.wallMs.toFixed(1)} ms plain
              </div>
            </div>
          </div>

          <div className={`live-note ${greedyMatches === true ? 'tone-good' : ''}`}>
            {greedyMatches === true
              ? `Exact greedy check: all ${runData.spec.totalEmitted}/${runData.spec.totalEmitted} generated characters match plain target decoding character for character. Notice the CPU stopwatch above (${cpuRatio.toFixed(2)}×): on a tiny 116 KiB model in CPU cache, verification is compute-bound, so wall-clock time gets worse — the gain only appears on memory-bound GPUs.`
              : `At temperature ${temperature}, speculative decoding emits ${runData.spec.tokensPerTargetPass.toFixed(2)} chars per target pass (acceptance ${(runData.spec.acceptanceRate * 100).toFixed(1)}%). Your browser’s CPU stopwatch measured ${cpuRatio.toFixed(2)}× (${runData.spec.wallMs.toFixed(1)} ms vs ${runData.plain.wallMs.toFixed(1)} ms plain) because this tiny model is compute-bound on CPU, not memory-bound.`}
          </div>
        </>
      )}
    </Widget>
  );
}
