import { useId } from 'react';

/**
 * A feed-forward network drawn as columns of nodes joined by weighted edges.
 * Layer indices: node column 0 is the input; column l (≥ 1) is the output of weight layer l − 1.
 * Edges are W[l][j][i]: from node i in column l to node j in column l + 1 (same convention as lib/mlp).
 * Edge colour = sign (blue positive, orange negative), thickness = size.
 */
export type NetSelection = { kind: 'neuron'; col: number; j: number } | { kind: 'edge'; l: number; j: number; i: number } | null;

export function NetDiagram({
  sizes,
  W,
  values,
  showValues = true,
  thumbs,
  selected = null,
  onSelect,
  width = 520,
  height = 300,
  node = 30,
  inputLabels,
  activeCols,
  dead,
  fill,
  ariaLabel = 'Network diagram',
  maxAbsW,
  badges,
}: {
  sizes: number[];
  W: number[][][];
  /** values[col][j] shown under (and used to fill) each node. Missing columns are drawn empty. */
  values?: (number[] | undefined)[];
  showValues?: boolean;
  /** thumbs[col][j]: image data URL drawn inside the node. */
  thumbs?: (string[] | undefined)[];
  selected?: NetSelection;
  onSelect?: (s: NetSelection) => void;
  width?: number;
  height?: number;
  /** Node size in px. */
  node?: number;
  inputLabels?: string[];
  /** Columns to highlight (others' edges fade). */
  activeCols?: number[];
  /** dead[col][j]: node is switched off (drawn hatched). */
  dead?: (boolean[] | undefined)[];
  /** Fill colour for a node value; default shades the accent by value in [0, 1]. */
  fill?: (v: number, col: number) => string;
  ariaLabel?: string;
  maxAbsW?: number;
  /** badges[col][j]: short text drawn just above the node (e.g. an importance score). */
  badges?: (( string | undefined)[] | undefined)[];
}) {
  const uid = useId().replace(/:/g, '');
  const padX = node / 2 + (inputLabels ? 46 : 8);
  const padY = node / 2 + (badges ? 16 : 6);
  const cols = sizes.length;
  const cx = (c: number) => (cols === 1 ? width / 2 : padX + (c * (width - 2 * padX + (inputLabels ? 38 : 0))) / (cols - 1));
  const maxN = Math.max(...sizes);
  const gap = Math.min((height - 2 * padY - (showValues ? 14 : 0)) / Math.max(1, maxN - 1), node * 2.4);
  const cy = (c: number, j: number) => height / 2 - (showValues ? 7 : 0) + (j - (sizes[c] - 1) / 2) * gap;
  const mw = maxAbsW ?? Math.max(1e-9, ...W.flat(2).map(Math.abs));
  const sel = selected;
  const touches = (l: number, j: number, i: number) => {
    if (!sel) return false;
    if (sel.kind === 'edge') return sel.l === l && sel.j === j && sel.i === i;
    return (sel.col === l + 1 && sel.j === j) || (sel.col === l && sel.j === i);
  };
  const defaultFill = (v: number) => `color-mix(in srgb, var(--accent) ${Math.round(Math.max(0, Math.min(1, v)) * 85)}%, var(--bg-elev))`;
  const fl = fill ?? defaultFill;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={ariaLabel} style={{ maxWidth: '100%', height: 'auto' }}>
      <defs>
        <pattern id={`hatch${uid}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="var(--bg-sunken)" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--text-faint)" strokeWidth="2" />
        </pattern>
      </defs>
      {W.map((L, l) =>
        L.map((row, j) =>
          row.map((w, i) => {
            const x1 = cx(l) + node / 2;
            const y1 = cy(l, i);
            const x2 = cx(l + 1) - node / 2;
            const y2 = cy(l + 1, j);
            const hot = touches(l, j, i);
            const faded = (sel && !hot) || (activeCols && !activeCols.includes(l + 1));
            const isSelEdge = sel?.kind === 'edge' && sel.l === l && sel.j === j && sel.i === i;
            return (
              <g key={`${l}-${j}-${i}`} data-edge={`${l}-${j}-${i}`} style={{ cursor: onSelect ? 'pointer' : undefined }} onClick={onSelect ? () => onSelect(isSelEdge ? null : { kind: 'edge', l, j, i }) : undefined}>
                {onSelect && <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" strokeWidth={9} />}
                {isSelEdge && <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--c-grad)" strokeWidth={0.8 + (Math.abs(w) / mw) * 5 + 5} opacity={0.35} strokeLinecap="round" />}
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={w === 0 ? 'var(--border-strong)' : w > 0 ? 'var(--c-pos)' : 'var(--c-neg)'}
                  strokeWidth={w === 0 ? 1 : 0.8 + (Math.abs(w) / mw) * 5}
                  strokeDasharray={w === 0 ? '3 3' : undefined}
                  opacity={faded ? 0.18 : 0.85}
                />
              </g>
            );
          }),
        ),
      )}
      {sizes.map((n, c) =>
        Array.from({ length: n }, (_, j) => {
          const x = cx(c) - node / 2;
          const y = cy(c, j) - node / 2;
          const v = values?.[c]?.[j];
          const thumb = thumbs?.[c]?.[j];
          const isSel = sel?.kind === 'neuron' && sel.col === c && sel.j === j;
          const isDead = dead?.[c]?.[j];
          return (
            <g key={`n${c}-${j}`} data-node={`${c}-${j}`} style={{ cursor: onSelect ? 'pointer' : undefined }} onClick={onSelect ? () => onSelect(isSel ? null : { kind: 'neuron', col: c, j }) : undefined}>
              <clipPath id={`clip${uid}-${c}-${j}`}>
                <rect x={x} y={y} width={node} height={node} rx={c === 0 ? node / 2 : 6} />
              </clipPath>
              <rect x={x} y={y} width={node} height={node} rx={c === 0 ? node / 2 : 6} fill={isDead ? `url(#hatch${uid})` : v !== undefined && !thumb ? fl(v, c) : 'var(--bg-elev)'} />
              {thumb && !isDead && <image href={thumb} x={x} y={y} width={node} height={node} clipPath={`url(#clip${uid}-${c}-${j})`} preserveAspectRatio="none" />}
              <rect
                x={x}
                y={y}
                width={node}
                height={node}
                rx={c === 0 ? node / 2 : 6}
                fill="none"
                stroke={isSel ? 'var(--c-grad)' : 'var(--border-strong)'}
                strokeWidth={isSel ? 3 : 1.5}
              />
              {showValues && v !== undefined && (
                <text x={cx(c)} y={y + node + 12} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                  {Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)}
                </text>
              )}
              {badges?.[c]?.[j] && (
                <text x={cx(c)} y={y - 4} textAnchor="middle" fontSize={10.5} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--c-grad)">
                  {badges[c]![j]}
                </text>
              )}
              {c === 0 && inputLabels?.[j] && (
                <text x={x - 6} y={cy(c, j) + 4} textAnchor="end" fontSize={12} fontWeight={600} fill="var(--text-muted)" fontFamily="var(--font-ui)">
                  {inputLabels[j]}
                </text>
              )}
            </g>
          );
        }),
      )}
    </svg>
  );
}
