// authored-by: Claude Opus 5.5
/**
 * The capstone's centrepiece: train the tiny GPT live and scrub back through its life.
 * Every panel is computed from the learner's own checkpoints (parameters read back from the worker
 * or the GPU). On the CPU tier the run is bit-identical to the recorded reference run.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { BackendChip, useComputeTier } from '../../components/BackendChip';
import { usePrefersReducedMotion } from '../../components/motion';
import { Gpt } from '../../lib/compute/gpt';
import { CPU_RUN, GPU_RUN, PROMPT, samplesAt } from '../../lib/compute/gptReference';
import { CPU_REFERENCE } from '../../lib/compute/gptReferenceResults';
import { GPU_REFERENCE } from '../../lib/compute/gptReferenceGpuResults';
import { initialParams, runTraining, type RunSpec } from '../../lib/compute/gptRun';
import { createGptTrainer, type GptTrainer } from '../../lib/compute/gptTrainer';
import { charFrequencies, dictionary, evalWindows, loadGrimm, type Corpus } from '../../lib/corpora/grimm';
import { births, continueText, landmarks, statsOf, type Birth, type LiveCheckpoint } from './tinyGpt';
import { runStore } from './runStore';

const REF = CPU_REFERENCE;
const CAL = REF.calibration;
const W = 520;
const H = 230;
const PAD = { l: 40, r: 14, t: 14, b: 34 };

type Lang = { corpus: Corpus; dict: Set<string>; freq: Float64Array };

export function useCorpus(): Lang | null {
  const [lang, setLang] = useState<Lang | null>(null);
  useEffect(() => {
    let live = true;
    loadGrimm().then((corpus) => {
      if (!live) return;
      setLang({ corpus, dict: dictionary(corpus.text.slice(0, corpus.split)), freq: charFrequencies(corpus.train) });
    });
    return () => {
      live = false;
    };
  }, []);
  return lang;
}

/** Keep the prompt inside the model's 68 characters. */
export function cleanPrompt(s: string) {
  return s.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[—–]/g, '-').replace(/[^A-Za-z0-9 .,;:!?'"()\-\n]/g, '');
}

const fmtStep = (s: number) => s.toLocaleString('en-GB');
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;

function StatLine({ c }: { c: LiveCheckpoint }) {
  return (
    <div className="mono" style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
      val loss {c.valLoss.toFixed(3)} · real words {pct(c.stats.realShare)} · letter-mix gap {c.stats.kl.toFixed(3)} · longest real run {c.stats.longestRun}
    </div>
  );
}

export function TrainWidget() {
  const tier = useComputeTier();
  const lang = useCorpus();
  const reduced = usePrefersReducedMotion();
  const [size, setSize] = useState<'small' | 'large'>('small');
  const spec: RunSpec = size === 'large' ? GPU_RUN : CPU_RUN;
  const [cps, setCps] = useState<LiveCheckpoint[]>([]);
  const [sel, setSel] = useState(0);
  const [follow, setFollow] = useState(true);
  const [running, setRunning] = useState(false);
  const [timing, setTiming] = useState('');
  const [flash, setFlash] = useState<Birth | null>(null);
  const [where, setWhere] = useState('Web Worker');
  const [prompt, setPrompt] = useState(PROMPT);
  const [asked, setAsked] = useState(PROMPT);
  const stop = useRef(false);
  const trainer = useRef<GptTrainer | null>(null);
  const seen = useRef(new Set<string>());

  // the untrained network, shown before anything is trained
  useEffect(() => {
    if (!lang || cps.length) return;
    const p = initialParams(spec);
    const samples = samplesAt(spec, p);
    const m = new Gpt(spec.cfg);
    m.load(p);
    const valLoss = m.evalLoss(evalWindows(lang.corpus.val, spec.evalRows, spec.T), spec.evalRows, spec.T);
    setCps([{ step: 0, params: p, trainLoss: NaN, valLoss, samples, stats: statsOf(samples, lang.dict, lang.freq), sec: 0 }]);
    setSel(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang, size]);

  useEffect(
    () => () => {
      stop.current = true;
      trainer.current?.destroy();
    },
    [],
  );

  const born = useMemo(() => births(cps, CAL), [cps]);
  useEffect(() => {
    for (const b of born)
      if (!seen.current.has(b.id) && running) {
        seen.current.add(b.id);
        setFlash(b);
      }
  }, [born, running]);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), reduced ? 6000 : 4500);
    return () => clearTimeout(t);
  }, [flash, reduced]);

  const run = async () => {
    if (!lang) return;
    stop.current = false;
    seen.current = new Set();
    setRunning(true);
    setFollow(true);
    trainer.current?.destroy();
    const tr = await createGptTrainer(spec.cfg);
    trainer.current = tr;
    setWhere(tr.where === 'worker' ? 'Web Worker' : tr.where);
    const acc: LiveCheckpoint[] = [];
    const label = size === 'large' ? 'larger model' : 'tiny model';
    try {
      await runTraining(spec, lang.corpus, tr, async (c) => {
        const samples = samplesAt(spec, c.params);
        acc.push({ step: c.step, params: c.params, trainLoss: c.trainLoss, valLoss: c.valLoss, samples, stats: statsOf(samples, lang.dict, lang.freq), sec: c.ms / 1000 });
        setCps([...acc]);
        runStore.set({ spec, label, tier: tr.kind, checkpoints: acc, done: c.step === spec.steps });
        const rate = c.step / Math.max(1e-3, c.ms / 1000);
        const left = c.step ? (spec.steps - c.step) / rate : NaN;
        setTiming(`step ${fmtStep(c.step)} / ${fmtStep(spec.steps)} · ${(c.ms / 1000).toFixed(0)} s${Number.isFinite(left) && left > 1 ? ` · about ${Math.ceil(left / 60)} min left` : ''}`);
        await new Promise((r) => requestAnimationFrame(() => r(null)));
        return !stop.current;
      });
    } catch {
      /* trainer destroyed mid-run */
    }
    setRunning(false);
  };

  // follow the newest checkpoint while training, unless the learner grabbed the scrubber
  useEffect(() => {
    if (follow) setSel(Math.max(0, cps.length - 1));
  }, [cps.length, follow]);

  // ▶ replay the timeline
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      setSel(cps.length - 1);
      setPlaying(false);
      return;
    }
    const t = setInterval(() => {
      setSel((s) => {
        if (s >= cps.length - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 420);
    return () => clearInterval(t);
  }, [playing, reduced, cps.length]);

  const cur = cps[Math.min(sel, cps.length - 1)];
  const first = cps[0];
  const model = useMemo(() => {
    if (!cur) return null;
    const m = new Gpt(spec.cfg);
    m.load(cur.params);
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cur]);
  const model0 = useMemo(() => {
    if (!first) return null;
    const m = new Gpt(spec.cfg);
    m.load(first.params);
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [first]);
  const cleanAsked = cleanPrompt(asked) || PROMPT;
  const reply = useMemo(() => (model ? continueText(model, cleanAsked, 160, 7) : ''), [model, cleanAsked]);
  const reply0 = useMemo(() => (model0 ? continueText(model0, cleanAsked, 160, 7) : ''), [model0, cleanAsked]);

  // ---- chart
  const total = spec.steps;
  const x = (s: number) => PAD.l + ((W - PAD.l - PAD.r) * Math.log10(s + 1)) / Math.log10(total + 1);
  const lo = 1.4;
  const hi = 4.4;
  const y = (v: number) => PAD.t + ((H - PAD.t - PAD.b) * (hi - Math.min(hi, Math.max(lo, v)))) / (hi - lo);
  const line = (pts: { step: number; v: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.v).toFixed(1)}`).join('');
  const refPath = line((size === 'small' ? REF : GPU_REFERENCE).checkpoints.map((c) => ({ step: c.step, v: c.valLoss })));
  const livePath = line(cps.map((c) => ({ step: c.step, v: c.valLoss })));
  const base = REF.baselines;
  const stepTicks = [0, 10, 100, 1000].filter((v) => v <= total);

  const marks = landmarks(cps.map((c) => c.step), born);
  const recordedBorn = births(REF.checkpoints.map((c) => ({ step: c.step, stats: c.stats })), CAL);
  const recordedMarks = landmarks(REF.checkpoints.map((c) => c.step), recordedBorn);
  const haveRun = cps.length > 1;

  let note = 'Loading the fairy tales…';
  if (lang && cur) {
    if (!haveRun)
      note = `This is the network before any training: ${REF.params.toLocaleString('en-GB')} random numbers. Its letters are almost uniform (a validation loss of ${cur.valLoss.toFixed(3)}, close to ln 68 = ${base.uniform.toFixed(3)}, the loss of a blind guess). Press Train.`;
    else {
      const b = born.filter((bb) => bb.step <= cur.step);
      note = `Step ${fmtStep(cur.step)}: validation loss ${cur.valLoss.toFixed(3)} nats per character (bigram counts: ${base.bigram.toFixed(3)}). ${pct(cur.stats.realShare)} of the words in its four samples are real words from the tales.`;
      if (b.length) note += ` Reached so far: ${b.map((bb) => bb.label).join('; ')}.`;
      if (cur.step === spec.steps && size === 'small') {
        const missing = recordedBorn.length < 5 ? ' It never gets to four real words in five, or to six real words in a row.' : '';
        note += missing;
      }
    }
  }

  const sizeOptions = [
    { value: 'small' as const, label: `tiny (${REF.params.toLocaleString('en-GB')} parameters)` },
    { value: 'large' as const, label: 'larger (GPU only)' },
  ];

  return (
    <Widget
      title="Train a tiny GPT, and watch it learn to write"
      wide
      actions={<BackendChip tier={tier} timing={timing} cpuWhere={where} />}
      caption={
        <>
          A real {spec.cfg.layers}-block GPT (width {spec.cfg.d}, {spec.cfg.heads} heads, {spec.cfg.ctx} characters of context) learning to predict the next character of
          Grimms’ fairy tales. The chart’s <span className="c-p">solid line</span> is your run’s validation loss on four tales it never trains on; dotted lines are
          the baselines (guessing blindly, letter counts, pair counts), and the faint dashed line is the recorded reference run. The numbered ticks are
          measured milestones; click one, or drag the timeline, to see what the model wrote at that step. On the CPU your run reproduces the recorded one exactly; on a GPU
          the last bits of each sum differ, so the samples can drift apart.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8, flexWrap: 'wrap', gap: 8 }}>
        <button
          className={`btn${running ? '' : ' primary'}`}
          disabled={!lang || tier.kind === 'checking' || (size === 'large' && tier.kind !== 'webgpu')}
          onClick={() => (running ? (stop.current = true) : run())}
        >
          {running ? '❚❚ Stop' : haveRun ? '↻ Train again' : '▶ Train'}
        </button>
        <Segmented
          options={sizeOptions}
          value={size}
          onChange={(v) => {
            if (running) return;
            setSize(v);
            setCps([]);
            setTiming('');
          }}
        />
        {haveRun && (
          <button className="btn" onClick={() => (setFollow(false), setSel(0), setPlaying(true))} disabled={running}>
            ▶ Replay its life
          </button>
        )}
      </div>
      {size === 'large' && tier.kind !== 'webgpu' && (
        <p className="live-note" style={{ marginTop: 0 }}>
          The larger model has about {Math.round((GPU_RUN.cfg.d / CPU_RUN.cfg.d) ** 2)} times as many weights per layer. Trained on one CPU core its recorded reference run took about an hour
          and a quarter, so here it is offered only when this browser exposes a GPU.
        </p>
      )}
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, minWidth: 320, display: 'block' }} role="img" aria-label="Validation loss against training steps">
          {[1.5, 2, 2.5, 3, 3.5, 4].map((v) => (
            <g key={v}>
              <line className="plot-grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
              <text className="plot-tick" x={PAD.l - 5} y={y(v) + 3.5} textAnchor="end">
                {v.toFixed(1)}
              </text>
            </g>
          ))}
          {stepTicks.map((t) => (
            <text key={t} className="plot-tick" x={x(t)} y={H - PAD.b + 14} textAnchor={t === 0 ? 'start' : 'middle'}>
              {fmtStep(t)}
            </text>
          ))}
          <text className="plot-tick" x={(W + PAD.l) / 2} y={H - 3} textAnchor="middle">
            training steps (log scale)
          </text>
          {(
            [
              ['blind guess', base.uniform],
              ['letter counts', base.unigram],
              ['pair counts', base.bigram],
            ] as const
          ).map(([name, v]) => (
            <g key={name}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="var(--text-faint)" strokeDasharray="2 3" />
              <text className="plot-tick" x={W - PAD.r} y={y(v) - 4} textAnchor="end">
                {name} {v.toFixed(2)}
              </text>
            </g>
          ))}
          {refPath && <path d={refPath} fill="none" stroke="var(--text-faint)" strokeWidth={1.5} strokeDasharray="5 4" />}
          {haveRun && <path d={livePath} fill="none" stroke="var(--accent)" strokeWidth={2.6} />}
          {cps.map((c, i) => (
            <circle key={c.step} cx={x(c.step)} cy={y(c.valLoss)} r={i === sel ? 5 : 2.4} fill={i === sel ? 'var(--accent)' : 'var(--text-faint)'} style={{ cursor: 'pointer' }} onClick={() => (setFollow(false), setSel(i))} />
          ))}
          {cur && <line x1={x(cur.step)} x2={x(cur.step)} y1={PAD.t} y2={H - PAD.b} stroke="var(--accent)" strokeOpacity={0.35} />}
          {born.map((b, k) => {
            const i = cps.findIndex((c) => c.step === b.step);
            return (
              <g key={b.id} style={{ cursor: 'pointer' }} onClick={() => (setFollow(false), setSel(i))} data-birth={b.id}>
                <line x1={x(b.step)} x2={x(b.step)} y1={H - PAD.b - 10} y2={H - PAD.b} stroke="var(--c-c)" strokeWidth={2} />
                <circle cx={x(b.step)} cy={H - PAD.b - 17} r={7} fill="var(--c-c)" />
                <text x={x(b.step)} y={H - PAD.b - 13.5} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--bg)">
                  {k + 1}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      {flash && (
        <div className="birth-flash" role="status">
          Step {fmtStep(flash.step)}: {flash.label}.
        </div>
      )}
      {cps.length > 0 && (
        <label className="slider" style={{ display: 'block', margin: '6px 0 10px' }}>
          <span className="slider-top">
            <span className="slider-label">timeline</span>
            <span className="slider-value">step {cur ? fmtStep(cur.step) : 0}</span>
          </span>
          <input
            type="range"
            min={0}
            max={Math.max(0, cps.length - 1)}
            step={1}
            value={Math.min(sel, cps.length - 1)}
            onChange={(e) => (setFollow(false), setSel(+e.target.value))}
            aria-label="Training timeline"
          />
        </label>
      )}
      {born.length > 0 && (
        <div className="btn-row" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
          {born.map((b, k) => (
            <button key={b.id} className="btn" style={{ fontSize: 12.5 }} onClick={() => (setFollow(false), setSel(cps.findIndex((c) => c.step === b.step)))}>
              {k + 1}. step {fmtStep(b.step)}: {b.label}
            </button>
          ))}
        </div>
      )}
      {cur && first && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(240px, 100%), 1fr))', gap: 12 }}>
          {(haveRun
            ? [
                ['Step 0 (random weights)', first],
                [`Step ${fmtStep(cur.step)}`, cur],
              ]
            : [['Step 0 (random weights)', first]]
          ).map(([title, c]) => (
            <div key={title as string} className="gpt-panel">
              <div className="field-label" style={{ textTransform: 'none' }}>
                {title as string}: “{PROMPT}…”
              </div>
              <div className="mono gpt-sample">{PROMPT}<b>{(c as LiveCheckpoint).samples[0]}</b></div>
              <StatLine c={c as LiveCheckpoint} />
            </div>
          ))}
        </div>
      )}
      <p className="live-note">{note}</p>
      {cur && (
        <div className="gpt-panel" style={{ marginTop: 10 }}>
          <div className="field-label" style={{ textTransform: 'none' }}>Talk to it: your prompt, continued by the model at the selected step (and by the random one)</div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setAsked(prompt);
            }}
            style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '6px 0' }}
          >
            <input className="text-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={60} style={{ flex: '1 1 200px' }} aria-label="Prompt" />
            <button className="btn primary" type="submit">
              Continue
            </button>
          </form>
          <div className="mono gpt-sample">
            <span style={{ color: 'var(--text-muted)' }}>step {fmtStep(cur.step)}: </span>
            {cleanAsked}
            <b>{reply}</b>
          </div>
          {cur.step > 0 && (
            <div className="mono gpt-sample" style={{ opacity: 0.75 }}>
              <span style={{ color: 'var(--text-muted)' }}>step 0: </span>
              {cleanAsked}
              <b>{reply0}</b>
            </div>
          )}
          {cleanPrompt(asked) !== asked && <div className="plot-tick">Characters the model has never seen were left out of the prompt.</div>}
        </div>
      )}
      <details className="gpt-panel" style={{ marginTop: 10 }} open>
        <summary className="field-label" style={{ textTransform: 'none', cursor: 'pointer' }}>
          Same prompt, same seed, at the landmark steps {haveRun ? 'of your run' : 'of the recorded run'}
        </summary>
        {(haveRun ? marks.map((s) => cps.find((c) => c.step === s)!) : recordedMarks.map((s) => REF.checkpoints.find((c) => c.step === s)!)).map((c) => {
          const b = (haveRun ? born : recordedBorn).filter((bb) => bb.step === c.step);
          return (
            <div key={c.step} style={{ margin: '8px 0' }}>
              <div className="plot-tick">
                step {fmtStep(c.step)} · val loss {c.valLoss.toFixed(3)} · {pct(c.stats.realShare)} real words{b.length ? ` · ${b.map((bb) => bb.label).join('; ')}` : ''}
              </div>
              <div className="mono gpt-sample">{PROMPT}<b>{c.samples[0].slice(0, 120)}</b></div>
            </div>
          );
        })}
      </details>
    </Widget>
  );
}
