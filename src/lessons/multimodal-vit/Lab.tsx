// authored-by: Claude Opus 5.5
import { useEffect, useRef, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Segmented } from '../../components/controls';
import { trainClip, zeroShot } from './clip';
import { attnCost, ATTN_CAP, CLIP_BAR, PATCH_BAR, PATCH_N, RESCUE_BAR, RESCUE_N, runOne, SEEDS, type Arch } from './visionLab';
import { CLIP_RESULTS, VISION_RESULTS } from './visionResults';

type State = { P: 1 | 2 | 4; d: 4 | 8 | 16; pos: boolean; augment: boolean; tau: number; k: number; batch: number };
const INIT: State = { P: 4, d: 4, pos: true, augment: false, tau: 1, k: 2, batch: 4 };
const KEY = 'vit-lab-state-v1';
const BRIEFS = [
  { id: 'patch', tab: '1 · Patch budget' },
  { id: 'rescue', tab: '2 · Small data' },
  { id: 'clip', tab: '3 · Zero-shot' },
] as const;
type BriefId = (typeof BRIEFS)[number]['id'];
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;

function load(): State {
  try {
    const s = JSON.parse(localStorage.getItem(KEY) ?? 'null');
    if (s && typeof s.tau === 'number') return { ...INIT, ...s };
  } catch {
    /* ignore */
  }
  return INIT;
}

type Result = { brief: BriefId; key: string; accs: number[]; cost?: number; over?: boolean };

const keyOf = (b: BriefId, s: State) => (b === 'patch' ? `${b}:${s.P}:${s.d}` : b === 'rescue' ? `${b}:${s.P}:${s.d}:${s.pos}:${s.augment}` : `${b}:${s.tau}:${s.k}:${s.batch}`);

/** Evidence-only hints. */
export function vitHint(b: BriefId, r: Result | null): string {
  if (!r) return 'Train a setting first: the hint reads its results.';
  const m = mean(r.accs);
  if (b === 'patch') {
    if (r.over) return `This layout needs ${r.cost!.toLocaleString()} attention multiply-adds per image, over the ${ATTN_CAP.toLocaleString()} budget: attention compares every token with every other, so the cost grows with the square of the number of tokens.`;
    if (m < PATCH_BAR) return `${pct(m)} on new images. A 3×3 sprite is small: ask how much of it lands in one token, and how many unrelated pixels each token mixes in with it.`;
    return `Within budget at ${r.cost!.toLocaleString()} multiply-adds per image.`;
  }
  if (b === 'rescue') {
    const spread = Math.max(...r.accs) - Math.min(...r.accs);
    return m < RESCUE_BAR
      ? `${pct(m)} from 32 training images (seeds range over ${(100 * spread).toFixed(1)} points). The plus and the X can be anywhere; with 32 examples the model has seen each sprite at only a few of its 36 possible positions. What would it need to have seen?`
      : 'It generalises from 32 images.';
  }
  return m < CLIP_BAR
    ? `${pct(m)} zero-shot. Each training step can only push an image away from the other captions in its batch, and the similarity is divided by the temperature before the softmax. Think about how far apart 8 classes can sit in the space you gave them.`
    : 'Images and captions line up.';
}

export function WireLab() {
  const progress = useLabProgress('vit-lab-v1');
  const log = useMachineLog();
  const [active, setActive] = useState(0);
  const brief = BRIEFS[active].id as BriefId;
  const [s, setS] = useState<State>(load);
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(s));
    } catch {
      /* ignore */
    }
  }, [s]);
  const effP = brief === 'rescue' && s.P === 1 ? 2 : s.P;
  const key = keyOf(brief, { ...s, P: effP as 1 | 2 | 4, d: brief === 'rescue' && s.d === 16 ? 8 : s.d });
  const cur = res && res.key === key ? res : null;
  const m = cur ? mean(cur.accs) : 0;
  const passing = !!cur && !cur.over && (brief === 'patch' ? m >= PATCH_BAR : brief === 'rescue' ? m >= RESCUE_BAR : m >= CLIP_BAR);
  const optimal = passing && (brief === 'patch' ? cur!.cost! <= 2048 : brief === 'clip' ? s.batch <= 8 : false);

  const run = async () => {
    setBusy(true);
    await new Promise((r) => setTimeout(r, 30));
    let out: Result;
    if (brief === 'patch') {
      const cost = attnCost(s.P, s.d);
      if (cost > ATTN_CAP) out = { brief, key, accs: [], cost, over: true };
      else {
        const arch: Arch = { kind: 'vit', P: s.P as 2 | 4, d: s.d, pos: true, augment: false };
        const accs: number[] = [];
        for (const seed of SEEDS) {
          accs.push(runOne(arch, PATCH_N, seed));
          await new Promise((r) => setTimeout(r, 0));
        }
        out = { brief, key, accs, cost };
      }
    } else if (brief === 'rescue') {
      const arch: Arch = { kind: 'vit', P: effP as 2 | 4, d: s.d === 16 ? 8 : s.d, pos: s.pos, augment: s.augment };
      const accs: number[] = [];
      for (const seed of SEEDS) {
        accs.push(runOne(arch, RESCUE_N, seed));
        await new Promise((r) => setTimeout(r, 0));
      }
      out = { brief, key, accs };
    } else {
      out = { brief, key, accs: SEEDS.map((seed) => zeroShot(trainClip({ tau: s.tau, k: s.k, batch: s.batch }, seed))) };
    }
    if (!alive.current) return;
    setRes(out);
    setBusy(false);
    log.push(`${key.replace(/:/g, ' ')} → ${out.over ? 'over budget, not trained' : `${pct(mean(out.accs))} (seeds ${out.accs.map((a) => pct(a)).join(', ')})`}`, out.over ? 'bad' : undefined);
  };

  const briefs: LabBrief[] = [
    {
      id: 'patch',
      tab: BRIEFS[0].tab,
      title: 'Brief 1 — The patch budget',
      text: `Your attention layer may spend at most ${ATTN_CAP.toLocaleString()} multiply-adds per 8×8 image (tokens² × width). Choose the patch size and width so that a ViT with position embeddings, trained on ${PATCH_N} images, tells plus from X at least ${pct(PATCH_BAR)} of the time (mean of seeds 1–3). Optional: the cheapest layout that does it.`,
    },
    {
      id: 'rescue',
      tab: BRIEFS[1].tab,
      title: 'Brief 2 — Rescue the small-data transformer',
      text: `The team must ship a transformer, and there are only ${RESCUE_N} labelled images. Choose patch size, width, whether to add position embeddings, and whether to augment the training images with random shifts. Reach ${pct(RESCUE_BAR)} mean test accuracy on seeds 1–3.`,
    },
    {
      id: 'clip',
      tab: BRIEFS[2].tab,
      title: 'Brief 3 — Zero-shot by caption',
      text: `Train the mini-CLIP encoders (400 steps) so that classifying new images by their most similar caption reaches ${pct(CLIP_BAR)} on average over seeds 1–3. You choose the temperature, the size of the shared space, and the batch size (how many captions each image is contrasted with). Optional: get there with the smallest batch.`,
    },
  ];

  let status = 'Choose the settings, then press “Train 3 seeds”.';
  if (busy) status = 'Training seeds 1–3…';
  else if (cur) status = cur.over ? `Over budget (${cur.cost!.toLocaleString()} multiply-adds per image), so not trained.` : `${passing ? '✓ ' : ''}Mean ${pct(m)} (seeds ${cur.accs.map(pct).join(', ')})${cur.cost ? ` · ${cur.cost.toLocaleString()} attention multiply-adds per image` : ''}${optimal ? ' · the cheapest possible' : ''}.`;

  const vrows = VISION_RESULTS.filter((r) => r.group === (brief === 'patch' ? 'patch' : 'rescue'));
  const reveal =
    brief === 'clip' ? (
      <div style={{ overflowX: 'auto' }}>
        <p style={{ margin: '0 0 6px' }}>All 36 settings, each on seeds 1–3: {CLIP_RESULTS.filter((r) => r.acc >= CLIP_BAR).length} reach 90%; the smallest batch that does is 8.</p>
        <table className="num-table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>temperature</th>
              <th>space</th>
              <th>batch</th>
              <th>zero-shot</th>
            </tr>
          </thead>
          <tbody>
            {CLIP_RESULTS.filter((r) => r.acc >= 0.85).map((r, i) => (
              <tr key={i} style={r.acc >= CLIP_BAR ? { background: 'var(--accent-soft)' } : undefined}>
                <td>{r.tau}</td>
                <td>{r.k}</td>
                <td>{r.batch}</td>
                <td className="mono">{pct(r.acc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ) : (
      <div style={{ overflowX: 'auto' }}>
        <p style={{ margin: '0 0 6px' }}>
          Every setting, each on seeds 1–3{brief === 'patch' ? '; one-pixel patches need 16,384 multiply-adds or more, over budget at any width' : ''}.
        </p>
        <table className="num-table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>patch</th>
              <th>width</th>
              {brief === 'rescue' && <th>positions</th>}
              {brief === 'rescue' && <th>shifts</th>}
              <th>mean accuracy</th>
            </tr>
          </thead>
          <tbody>
            {vrows.map((r, i) =>
              r.arch.kind === 'vit' ? (
                <tr key={i} style={mean(r.accs) >= (brief === 'patch' ? PATCH_BAR : RESCUE_BAR) ? { background: 'var(--accent-soft)' } : undefined}>
                  <td>{r.arch.P}×{r.arch.P}</td>
                  <td>{r.arch.d}</td>
                  {brief === 'rescue' && <td>{r.arch.pos ? 'yes' : 'no'}</td>}
                  {brief === 'rescue' && <td>{r.arch.augment ? 'yes' : 'no'}</td>}
                  <td className="mono">{pct(mean(r.accs))}</td>
                </tr>
              ) : null,
            )}
          </tbody>
        </table>
      </div>
    );

  const code =
    brief === 'clip'
      ? [
          `img = F.normalize(image_encoder(images), dim=-1)   # [${s.batch}, ${s.k}]`,
          `txt = F.normalize(text_encoder(captions), dim=-1)  # [${s.batch}, ${s.k}]`,
          `logits = img @ txt.T / ${s.tau}`,
          `labels = torch.arange(${s.batch})`,
          `loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2`,
        ].join('\n')
      : [
          `patches = image.unfold(0, ${effP}, ${effP}).unfold(1, ${effP}, ${effP}).reshape(-1, ${effP * effP})   # ${(8 / effP) ** 2} tokens`,
          `tokens = patches @ W_embed${brief === 'rescue' && !s.pos ? '' : ' + pos_embed'}          # width ${s.d}`,
          ...(brief === 'rescue' && s.augment ? ['image = random_shift(image, max=2)   # training only'] : []),
          `x = tokens + attention(tokens)`,
          `logit = x.mean(0) @ w_out`,
        ].join('\n');

  return (
    <LabShell
      title="Lab: Wire the Multimodal Model"
      caption="Real models, trained in your browser on seeds 1–3 when you press the button (a second or two). Each brief changes the controls. Your browser rounds exp() slightly differently from the Node.js run that recorded the table, so a live result can differ from the record by a point or so."
      briefs={briefs}
      active={active}
      onSelect={(i) => {
        setActive(i);
        setRes(null);
      }}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={status}
      hint={vitHint(brief, cur)}
      reveal={{ label: 'every setting of this brief, as recorded', content: reveal }}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Press “Train 3 seeds” to log a setting.' }}
      actions={
        <button type="button" className="btn" onClick={() => setS(INIT)}>
          Reset
        </button>
      }
    >
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div style={{ flex: '1 1 280px', display: 'grid', gap: 10 }}>
          {brief !== 'clip' && (
            <>
              <Segmented label="Patch size" options={(brief === 'patch' ? [1, 2, 4] : [2, 4]).map((p) => ({ value: p, label: `${p} × ${p}` }))} value={effP} onChange={(v) => setS((x) => ({ ...x, P: v as 1 | 2 | 4 }))} />
              <Segmented label="Token width" options={brief === 'patch' ? [4, 8, 16] : [4, 8]} value={brief === 'rescue' && s.d === 16 ? 8 : s.d} onChange={(v) => setS((x) => ({ ...x, d: v as 4 | 8 | 16 }))} />
            </>
          )}
          {brief === 'rescue' && (
            <>
              <Segmented label="Position embeddings" options={[{ value: 'no', label: 'none' }, { value: 'yes', label: 'learned' }]} value={s.pos ? 'yes' : 'no'} onChange={(v) => setS((x) => ({ ...x, pos: v === 'yes' }))} />
              <Segmented label="Random shifts while training" options={[{ value: 'no', label: 'off' }, { value: 'yes', label: 'on' }]} value={s.augment ? 'yes' : 'no'} onChange={(v) => setS((x) => ({ ...x, augment: v === 'yes' }))} />
            </>
          )}
          {brief === 'clip' && (
            <>
              <Segmented label="Temperature" options={[0.05, 0.1, 0.3, 1].map((t) => ({ value: t, label: `τ = ${t}` }))} value={s.tau} onChange={(v) => setS((x) => ({ ...x, tau: v as number }))} />
              <Segmented label="Size of the shared space" options={[2, 4, 8]} value={s.k} onChange={(v) => setS((x) => ({ ...x, k: v as number }))} />
              <Segmented label="Batch size" options={[4, 8, 16]} value={s.batch} onChange={(v) => setS((x) => ({ ...x, batch: v as number }))} />
            </>
          )}
          {brief === 'patch' && <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>Attention cost of this layout: {attnCost(s.P, s.d).toLocaleString()} multiply-adds per image (budget {ATTN_CAP.toLocaleString()}).</div>}
          <div className="btn-row">
            <button type="button" className="btn primary" disabled={busy} onClick={run}>
              {busy ? 'Training…' : 'Train 3 seeds'}
            </button>
          </div>
        </div>
        <div style={{ flex: '1 1 280px', minWidth: 0 }}>
          <LiveCode note="The same model in PyTorch (sketch)" code={code} />
        </div>
      </div>
    </LabShell>
  );
}
