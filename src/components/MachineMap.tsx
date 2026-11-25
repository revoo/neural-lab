// authored-by: Claude Opus 5.5
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { lessonHref } from '../router';

/**
 * The learning machine that the Maths Toolkit builds, as one diagram: the
 * forward stages (measurements → weighted sum → layer → stack → probabilities
 * → loss) along the top, and the learning stages (slope → gradient → chain
 * rule) coming back along the bottom. Each box links to the lesson that
 * builds it; `highlight` marks the current lesson's stage.
 */
export type MachineStage = { id: string; label: string; caption: string; lesson: string };

export const MACHINE_STAGES: MachineStage[] = [
  { id: 'features', label: 'Email → [3, 6]', caption: 'measurements as a vector', lesson: 'vectors' },
  { id: 'weighted-sum', label: 'w · x', caption: 'weigh the evidence', lesson: 'dot-product' },
  { id: 'layer', label: 'W x', caption: 'many neurons at once', lesson: 'matrices' },
  { id: 'stack', label: 'W₂(W₁ x)', caption: 'layers in a pipeline', lesson: 'matrix-multiplication' },
  { id: 'probabilities', label: 'softmax', caption: 'scores → probabilities', lesson: 'probability' },
  { id: 'loss', label: '−log p', caption: 'how surprised?', lesson: 'information' },
  { id: 'slope', label: 'f′(x)', caption: 'which way is downhill?', lesson: 'derivatives' },
  { id: 'gradient', label: '∇L', caption: 'every knob’s slope', lesson: 'gradients' },
  { id: 'backward', label: 'chain rule', caption: 'blame flows backwards', lesson: 'chain-rule' },
];

const FORWARD = MACHINE_STAGES.slice(0, 6);
const LEARNING = MACHINE_STAGES.slice(6); // slope, gradient, backward: drawn right to left

type Box = { s: MachineStage; x: number; y: number };
type Layout = { w: number; h: number; bw: number; bh: number; boxes: Box[]; arrows: { d: string; label?: { x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end' } }[]; rowLabels: { x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end' }[] };

/** Wide layout: forward row on top, learning row underneath running right to left. */
function wideLayout(): Layout {
  const bw = 146, bh = 58, gap = 16, x0 = 4, top = 26, bottom = 150;
  const fx = (i: number) => x0 + i * (bw + gap);
  const boxes: Box[] = [
    ...FORWARD.map((s, i) => ({ s, x: fx(i), y: top })),
    // slope under loss (col 5), gradient under probabilities (4), backward under stack (3)
    ...LEARNING.map((s, i) => ({ s, x: fx(5 - i), y: bottom })),
  ];
  const w = fx(5) + bw + x0;
  const arrows: Layout['arrows'] = [];
  for (let i = 0; i < 5; i++) arrows.push({ d: `M${fx(i) + bw + 2},${top + bh / 2} L${fx(i + 1) - 4},${top + bh / 2}` });
  for (let i = 5; i > 3; i--) arrows.push({ d: `M${fx(i) - 2},${bottom + bh / 2} L${fx(i - 1) + bw + 4},${bottom + bh / 2}` });
  // loss down to slope (right edge), backward up to the weights (layer box)
  const lx = fx(5) + bw / 2;
  arrows.push({ d: `M${lx},${top + bh + 2} L${lx},${bottom - 4}`, label: { x: lx + 6, y: (top + bh + bottom) / 2 + 4, text: 'learn', anchor: 'start' } });
  const bx = fx(3) + 16, wx = fx(2) + bw / 2;
  arrows.push({ d: `M${bx - 18},${bottom + bh / 2} C${wx},${bottom + bh / 2} ${wx},${bottom + bh / 2} ${wx},${top + bh + 4}`, label: { x: wx - 8, y: (top + bh + bottom) / 2 + 14, text: 'update the weights', anchor: 'end' } });
  return {
    w, h: bottom + bh + 20, bw, bh, boxes, arrows,
    rowLabels: [{ x: x0, y: 14, text: 'FORWARD: what the machine computes' }],
  };
}

/** Phone layout: forward stages down the left column, learning stages up the right. */
function narrowLayout(): Layout {
  const bw = 160, bh = 50, gap = 16, top = 22, lx = 2, rx = 206;
  const fy = (i: number) => top + i * (bh + gap);
  const boxes: Box[] = [
    ...FORWARD.map((s, i) => ({ s, x: lx, y: fy(i) })),
    ...LEARNING.map((s, i) => ({ s, x: rx, y: fy(5 - i) })),
  ];
  const arrows: Layout['arrows'] = [];
  for (let i = 0; i < 5; i++) arrows.push({ d: `M${lx + bw / 2},${fy(i) + bh + 2} L${lx + bw / 2},${fy(i + 1) - 4}` });
  for (let i = 5; i > 3; i--) arrows.push({ d: `M${rx + bw / 2},${fy(i) - 2} L${rx + bw / 2},${fy(i - 1) + bh + 4}` });
  arrows.push({ d: `M${lx + bw + 2},${fy(5) + bh / 2} L${rx - 4},${fy(5) + bh / 2}`, label: { x: (lx + bw + rx) / 2, y: fy(5) + bh / 2 - 9, text: 'learn' } });
  const ry = fy(3);
  arrows.push({ d: `M${rx + bw / 2},${ry - 2} C${rx + bw / 2},${fy(2) + bh / 2} ${rx + bw / 2},${fy(2) + bh / 2} ${lx + bw + 4},${fy(2) + bh / 2}`, label: { x: rx + bw / 2, y: fy(2) + bh / 2 - 8, text: 'update the weights' } });
  return { w: rx + bw + 2, h: fy(5) + bh + 4, bw, bh, boxes, arrows, rowLabels: [{ x: lx, y: 13, text: 'FORWARD' }, { x: rx, y: 13, text: 'LEARNING' }] };
}

export function MachineMap({ highlight, caption }: { highlight?: string; caption?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setNarrow(e.contentRect.width < 560));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const L = narrow ? narrowLayout() : wideLayout();
  const here = MACHINE_STAGES.find((s) => s.id === highlight);
  return (
    <figure className="machine-map" ref={ref}>
      <svg viewBox={`0 0 ${L.w} ${L.h}`} role="img" aria-label="The learning machine: forward stages from measurements to loss, then learning stages from slope back to the weights">
        <defs>
          <marker id="mm-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--text-muted)" />
          </marker>
        </defs>
        {L.rowLabels.map((r) => (
          <text key={r.text} x={r.x} y={r.y} className="mm-row" textAnchor={r.anchor ?? 'start'}>{r.text}</text>
        ))}
        {!narrow && (
          <text x={L.w - 4} y={L.h - 2} className="mm-row" textAnchor="end">
            ← LEARNING: how it improves
          </text>
        )}
        {L.arrows.map((a, i) => (
          <g key={i}>
            <path d={a.d} fill="none" stroke="var(--text-muted)" strokeWidth={1.4} markerEnd="url(#mm-arrow)" />
            {a.label && <text x={a.label.x} y={a.label.y} className="mm-arrow-label" textAnchor={a.label.anchor ?? 'middle'}>{a.label.text}</text>}
          </g>
        ))}
        {L.boxes.map(({ s, x, y }) => {
          const on = s.id === highlight;
          return (
            <a key={s.id} href={lessonHref(s.lesson)} className={`mm-box${on ? ' on' : ''}`} aria-current={on ? 'step' : undefined}>
              <title>{`${s.label}: ${s.caption}`}</title>
              <rect x={x} y={y} width={L.bw} height={L.bh} rx={8} />
              <text x={x + L.bw / 2} y={y + L.bh / 2 - 4} className="mm-label" textAnchor="middle">{s.label}</text>
              <text x={x + L.bw / 2} y={y + L.bh / 2 + 14} className="mm-caption" textAnchor="middle">{s.caption}</text>
            </a>
          );
        })}
      </svg>
      <figcaption className="live-note">
        {caption ?? (here ? <>You are here: <strong>{here.caption}</strong>.</> : <>Each box is one lesson of the Maths Toolkit. Click one to go there.</>)}
      </figcaption>
    </figure>
  );
}
