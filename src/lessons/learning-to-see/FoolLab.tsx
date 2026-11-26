// authored-by: Claude Opus 5.5
/**
 * Lab: Fool the classifier. Click pixels of a correctly classified photo to set them to full ink or
 * blank until the trained network changes its answer. The optional goal is the fewest changes, against
 * the best found by `fool.run.ts` (every single change, every pair, then greedy).
 */
import { useEffect, useMemo, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Toggle } from '../../components/controls';
import netUrl from './seeNet.bin?url';
import { Cnn } from './cnn';
import { CLASSES } from './fashion';
import { apply, lead, topClass, type Change } from './fool';
import { FOOL_RESULTS } from './foolResults';
import { saliency, SEE_RUN } from './see';
import { useFashion } from './SeeWidget';
import { PixelGrid } from './views';

let netPromise: Promise<Cnn> | null = null;
export function loadSeeNet() {
  netPromise ??= fetch(netUrl)
    .then((r) => r.arrayBuffer())
    .then((b) => {
      const n = new Cnn(SEE_RUN.cfg);
      n.p.set(new Float32Array(b));
      return n;
    });
  return netPromise;
}

type Recorded = { image: number; label: number; method: 'single' | 'pair' | 'greedy'; singles: number; pairsChecked: number; best: Change[] };
const RESULTS = FOOL_RESULTS as unknown as Recorded[];
const KEY = 'learning-to-see-fool';
const BRIEFS: LabBrief[] = RESULTS.map((r, i) => ({
  id: `img${i}`,
  tab: `${i + 1} · ${CLASSES[r.label]}`,
  title: `Make it stop seeing a ${CLASSES[r.label].toLowerCase()}`,
  text: (
    <>
      The network correctly calls this test photo a {CLASSES[r.label].toLowerCase()}. Change pixels (each click sets a pixel to full ink, then to blank, then back) until its top answer is
      anything else. Optional goal: as few changed pixels as the best attack our search found.
    </>
  ),
}));

export function FoolLab() {
  const data = useFashion();
  const [net, setNet] = useState<Cnn | null>(null);
  useEffect(() => {
    let ok = true;
    loadSeeNet().then((n) => ok && setNet(n));
    return () => {
      ok = false;
    };
  }, []);
  const progress = useLabProgress(KEY);
  const [active, setActive] = useState(0);
  const [changes, setChanges] = useState<Change[][]>(() => {
    try {
      const s = JSON.parse(localStorage.getItem(KEY + '-changes') ?? '');
      if (Array.isArray(s) && s.length === BRIEFS.length) return s;
    } catch {
      /* ignore */
    }
    return BRIEFS.map(() => []);
  });
  useEffect(() => {
    try {
      localStorage.setItem(KEY + '-changes', JSON.stringify(changes));
    } catch {
      /* ignore */
    }
  }, [changes]);
  const [showGrad, setShowGrad] = useState(false);
  const log = useMachineLog();
  const rec = RESULTS[active];
  const mine = changes[active];
  const orig = data ? data.test[rec.image] : null;
  const img = useMemo(() => (orig ? apply(orig, mine) : null), [orig, mine]);
  const state = useMemo(() => {
    if (!net || !img) return null;
    const tr = net.newTrace();
    const p = Array.from(net.forward(img, tr));
    return { p, top: topClass(net, img), lead: lead(net, img, rec.label) };
  }, [net, img, rec.label]);
  const grad = useMemo(() => (net && img && showGrad ? saliency(net, img, rec.label) : null), [net, img, rec.label, showGrad]);
  const fooled = !!state && state.top !== rec.label;
  const optimal = fooled && mine.length <= rec.best.length;

  const click = (px: number) => {
    if (!orig) return;
    const cur = mine.find((c) => c.px === px);
    let next: Change[];
    if (!cur) next = [...mine, { px, to: 1 }];
    else if (cur.to === 1) next = mine.map((c) => (c.px === px ? { px, to: 0 } : c));
    else next = mine.filter((c) => c.px !== px);
    // a change that sets a pixel to its own value isn't a change
    next = next.filter((c) => orig[c.px] !== c.to);
    setChanges(changes.map((c, i) => (i === active ? next : c)));
    if (net) {
      const im = apply(orig, next);
      const t = topClass(net, im);
      const l = lead(net, im, rec.label);
      log.push(`${next.length} pixel${next.length === 1 ? '' : 's'} changed: top answer ${CLASSES[t]}; ${CLASSES[rec.label]} leads the next class by ${l.toFixed(2)}`, t !== rec.label ? 'good' : 'info');
    }
  };

  const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
  return (
    <LabShell
      title="Lab: Fool the classifier"
      caption={
        <>
          The trained network from this lesson (the recorded run’s final weights, the same numbers your browser computes), judging one test photo. Pixel values are amounts of ink, drawn in the
          page’s text colour. Click a pixel to give it full ink; click again to blank it; a third click restores it. The gradient view shows where a little more ink would raise (first colour) or lower (second
          colour) the true class’s score: evidence, not an answer.
        </>
      }
      briefs={BRIEFS}
      active={active}
      onSelect={(i) => setActive(i)}
      progress={progress}
      passing={fooled}
      optimal={optimal}
      status={
        !state
          ? 'Loading the network…'
          : fooled
            ? `Fooled with ${mine.length} changed pixel${mine.length === 1 ? '' : 's'}: it now says ${CLASSES[state.top]} (${pct(state.p[state.top])}).${optimal ? ' That matches the best attack the search found.' : ` The best attack the search found uses ${rec.best.length}.`}`
            : `${mine.length} pixel${mine.length === 1 ? '' : 's'} changed. It still says ${CLASSES[rec.label]} (${pct(state.p[rec.label])}); its lead over the next class is ${state.lead.toFixed(2)} in raw scores.`
      }
      hint={
        state && !fooled ? (
          <>Watch the lead number after each click: changes that shrink it are working. Turn on the gradient view: it marks the pixels where the {CLASSES[rec.label].toLowerCase()} score is most sensitive right now.</>
        ) : undefined
      }
      reveal={{
        label: 'Show the search’s best attack',
        content: (
          <div>
            <p style={{ marginTop: 0 }}>
              {rec.method === 'single'
                ? `One pixel is enough: the search tried every single change and this was the first that worked, so ${rec.best.length} is the true minimum.`
                : rec.method === 'pair'
                  ? `No single pixel works (all ${rec.singles.toLocaleString('en-GB')} single changes were tried), but this pair does, so 2 is the true minimum.`
                  : `No single change and none of the ${rec.pairsChecked.toLocaleString('en-GB')} pairs works. Adding, one at a time, whichever change lowers the ${CLASSES[rec.label].toLowerCase()} lead most found this attack with ${rec.best.length} pixels: the best found, not a proven minimum.`}
            </p>
            {orig && <PixelGrid v={apply(orig, rec.best)} n={28} mode="grey" size={140} label="The search's attack" />}
          </div>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear, title: 'attack.log', empty: 'Each click is logged here.' }}
      actions={
        <button className="btn" onClick={() => setChanges(changes.map((c, i) => (i === active ? [] : c)))}>
          Reset this photo
        </button>
      }
    >
      {img && state && (
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <div>
            <PixelGrid v={img} n={28} mode="grey" size={252} label="Photo: click pixels to change them" onPixel={(i, e) => e.type === 'pointerdown' && click(i)} highlight />
            <div className="plot-tick">changed pixels: {mine.length}</div>
          </div>
          <div style={{ flex: '1 1 200px' }}>
            {state.p
              .map((p, k) => ({ p, k }))
              .sort((a, b) => b.p - a.p)
              .slice(0, 5)
              .map(({ p, k }) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
                  <span style={{ width: 82 }}>{CLASSES[k]}</span>
                  <span style={{ height: 10, width: `${55 * p}%`, background: `var(--cls-${k})`, borderRadius: 3 }} />
                  <span className="mono">{pct(p)}</span>
                </div>
              ))}
            <div style={{ marginTop: 8 }}>
              <Toggle label="gradient view" checked={showGrad} onChange={setShowGrad} />
            </div>
            {grad && <PixelGrid v={grad} n={28} mode="signed" size={140} label="Gradient of the true class's score" />}
          </div>
        </div>
      )}
    </LabShell>
  );
}
