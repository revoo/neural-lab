import { useState, type ReactElement } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { Plot, usePlot } from '../../components/plot';
import { Tex } from '../../components/Tex';
import { identity, lerpMat, matMul, type Mat } from '../../lib/linalg';
import { useFramePlayer } from '../../components/motion';

/** Morph timeline: 20 frames through the first matrix, 8 frames of pause, 20 frames through the second. */
export const MORPH = { first: 20, pause: 8, second: 20 };
export const MORPH_FRAMES = MORPH.first + MORPH.pause + MORPH.second;
/** The matrix shown at frame i when `first` is applied, then `second`. */
export function morphAt(first: Mat, second: Mat, i: number): Mat {
  const end1 = MORPH.first;
  const start2 = MORPH.first + MORPH.pause;
  if (i <= end1) return lerpMat(identity(2), first, i / end1);
  if (i <= start2) return first;
  return lerpMat(first, matMul(second, first), (i - start2) / MORPH.second);
}

/** The square grid, carried along by the matrix m. */
function Grid({ m }: { m: Mat }) {
  const { x, y } = usePlot();
  const lines: ReactElement[] = [];
  const ap = (p: [number, number]) => [m[0][0] * p[0] + m[0][1] * p[1], m[1][0] * p[0] + m[1][1] * p[1]];
  for (let k = -3; k <= 3; k++) {
    for (const [o, [p, q]] of ([[[k, -3], [k, 3]], [[-3, k], [3, k]]] as [number, number][][]).entries()) {
      const a = ap(p);
      const b = ap(q);
      lines.push(<line key={`${o}:${k}`} x1={x(a[0])} y1={y(a[1])} x2={x(b[0])} y2={y(b[1])} stroke={k === 0 ? 'var(--text-muted)' : 'var(--border-strong)'} strokeWidth={k === 0 ? 1.5 : 1} />);
    }
  }
  return <g>{lines}</g>;
}

const T: Record<string, { label: string; m: Mat }> = {
  rot: { label: 'Rotate 90°', m: [[0, -1], [1, 0]] },
  shear: { label: 'Shear →', m: [[1, 1], [0, 1]] },
  stretch: { label: 'Stretch x ×2', m: [[2, 0], [0, 1]] },
  flip: { label: 'Mirror', m: [[-1, 0], [0, 1]] },
};
type Key = keyof typeof T;

// An asymmetric "F" so that rotations and reflections are visible.
const F: [number, number][] = [
  [0, 0], [0.35, 0], [0.35, 0.8], [0.85, 0.8], [0.85, 1.1], [0.35, 1.1], [0.35, 1.4], [1.1, 1.4], [1.1, 1.75], [0, 1.75],
];

function Shape({ m, color, dashed }: { m: Mat; color: string; dashed?: boolean }) {
  const { x, y } = usePlot();
  const pts = F.map(([a, b]) => [m[0][0] * a + m[0][1] * b, m[1][0] * a + m[1][1] * b]);
  return (
    <polygon
      points={pts.map((p) => `${x(p[0])},${y(p[1])}`).join(' ')}
      fill={dashed ? 'none' : color}
      fillOpacity={0.35}
      stroke={color}
      strokeWidth={2}
      strokeDasharray={dashed ? '4 3' : undefined}
    />
  );
}

function Panel({ first, second, title }: { first: Key; second: Key; title: string }) {
  const prod = matMul(T[second].m, T[first].m); // apply `first`, then `second`
  return (
    <div style={{ flex: '1 1 240px', minWidth: 0 }}>
      <div style={{ fontWeight: 650, fontSize: 13.5, marginBottom: 4 }}>{title}</div>
      <Plot x={[-3, 3]} y={[-3, 3]} width={260} height={260} tickLabels={false} ariaLabel={title}>
        <Shape m={[[1, 0], [0, 1]]} color="var(--text-faint)" dashed />
        <Shape m={T[first].m} color="var(--c-b)" dashed />
        <Shape m={prod} color="var(--c-c)" />
      </Plot>
      <div style={{ fontSize: 14 }}>
        <Tex block>{`\\underbrace{${tex(T[second].m)}}_{\\text{${T[second].label}}}\\underbrace{${tex(T[first].m)}}_{\\text{${T[first].label}}} = ${tex(prod)}`}</Tex>
      </div>
    </div>
  );
}

const tex = (m: Mat) => `\\begin{bmatrix}${m.map((r) => r.join('&')).join('\\\\')}\\end{bmatrix}`;

export function CompositionDemo() {
  const [a, setA] = useState<Key>('rot');
  const [b, setB] = useState<Key>('shear');
  const opts = (Object.keys(T) as Key[]).map((k) => ({ value: k, label: T[k].label }));
  const same = JSON.stringify(matMul(T[a].m, T[b].m)) === JSON.stringify(matMul(T[b].m, T[a].m));
  const [anim, setAnim] = useState<{ order: 'AB' | 'BA'; i: number } | null>(null);
  const [ghost, setGhost] = useState<{ order: 'AB' | 'BA'; m: Mat } | null>(null);
  const [first, second] = anim?.order === 'BA' ? [b, a] : [a, b];
  const player = useFramePlayer(MORPH_FRAMES, 45, (i) => setAnim((cur) => (cur ? { ...cur, i } : cur)));
  const play = (order: 'AB' | 'BA') => {
    if (anim && anim.i === MORPH_FRAMES && anim.order !== order) {
      const [f0, s0] = anim.order === 'BA' ? [b, a] : [a, b];
      setGhost({ order: anim.order, m: matMul(T[s0].m, T[f0].m) });
    }
    setAnim({ order, i: 0 });
    player.start();
  };
  const shown = anim ? morphAt(T[first].m, T[second].m, anim.i) : identity(2);
  const fmtM = (m: Mat) => `[${m.map((r) => `[${r.map((v) => Number(v.toFixed(2))).join(', ')}]`).join(', ')}]`;
  const morphNote = !anim
    ? 'Press a play button: the grid and the F are carried through the first transformation, pause, then through the second.'
    : anim.i <= MORPH.first
      ? `Applying ${T[first].label}…`
      : anim.i <= MORPH.first + MORPH.pause
        ? `After ${T[first].label}: ${fmtM(T[first].m)}. Now ${T[second].label}…`
        : `After ${T[first].label}: ${fmtM(T[first].m)}; after ${T[second].label}: ${fmtM(matMul(T[second].m, T[first].m))}.` +
          (ghost && ghost.order !== anim.order && anim.i === MORPH_FRAMES ? (same ? ' The dashed ghost, from the other order, lands in the same place: this pair commutes.' : ' The dashed ghost is where the other order ended: a different place.') : '');
  return (
    <Widget
      title="Order matters: A then B is not B then A"
      wide
      caption={
        <>
          The grey outline is the original “F”. The dashed <span className="c-b">orange</span> F shows the result after the first transformation; the
          solid <span className="c-c">green</span> F is the final result. Matrices are written right-to-left: in <strong>BA</strong>, A acts first. Try
          to find a pair where the order <em>doesn’t</em> matter — they exist, but they’re the exception.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 12 }}>
        <Segmented label="Transformation A" options={opts} value={a} onChange={(k) => { setAnim(null); setGhost(null); setA(k); }} />
        <Segmented label="Transformation B" options={opts} value={b} onChange={(k) => { setAnim(null); setGhost(null); setB(k); }} />
      </div>
      <div className="widget-row" style={{ alignItems: 'center', marginBottom: 12 }}>
        <div style={{ flex: '0 1 300px', minWidth: 0 }}>
          <Plot x={[-3, 3]} y={[-3, 3]} width={300} height={300} tickLabels={false} ariaLabel="The grid morphing through the two transformations">
            <Grid m={shown} />
            <Shape m={[[1, 0], [0, 1]]} color="var(--text-faint)" dashed />
            {ghost && <Shape m={ghost.m} color="var(--c-b)" dashed />}
            <Shape m={shown} color="var(--c-c)" />
          </Plot>
        </div>
        <div className="controls" style={{ flex: '1 1 220px' }}>
          <div className="btn-row">
            <button type="button" className="btn primary" onClick={() => play('AB')} disabled={player.running || a === b}>
              ▶ Play: first A, then B
            </button>
            <button type="button" className="btn" onClick={() => play('BA')} disabled={player.running || a === b}>
              ▶ Play: first B, then A
            </button>
          </div>
          <div className="live-note">{morphNote}</div>
        </div>
      </div>
      <div className="widget-row">
        <Panel first={a} second={b} title="First A, then B   (the matrix BA)" />
        <Panel first={b} second={a} title="First B, then A   (the matrix AB)" />
      </div>
      <div className="live-note" style={{ marginTop: 8 }}>
        {a === b ? (
          <>Same transformation twice — of course the order can’t matter. Pick two different ones.</>
        ) : same ? (
          <>
            <strong>These two commute:</strong> AB = BA. Both results are identical. That happens for special pairs (for example, two stretches along
            the axes), but it’s not something you can assume.
          </>
        ) : (
          <>
            <strong>Different results.</strong> AB ≠ BA. Matrix multiplication composes transformations, and — just like “put on socks, then shoes” — the
            order changes the outcome.
          </>
        )}
      </div>
    </Widget>
  );
}
