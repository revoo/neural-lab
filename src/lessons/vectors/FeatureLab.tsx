// authored-by: Claude Opus 5.5
import { useMemo, useState } from 'react';
import { Segmented } from '../../components/controls';
import { Plot, Seg, usePlot } from '../../components/plot';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { BRIEFS, EMAILS, FEATURES, UNITS, UNIT_LABEL, nearest, points, score, solve, type Config, type FeatureId, type Unit } from './featureLab';

const LAB_KEY = 'neural-lab:vectors-lab:configs';
const PLOT_W = 440;
const PLOT_H = 330;

/** Domains for a plot whose two axes share one scale (data units per pixel), with a margin. */
export function equalScale(xr: [number, number], yr: [number, number], w: number, h: number): [[number, number], [number, number]] {
  const sx = Math.max(xr[1] - xr[0], 1e-9);
  const sy = Math.max(yr[1] - yr[0], 1e-9);
  const k = Math.max(sx / w, sy / h) * 1.2;
  const cx = (xr[0] + xr[1]) / 2;
  const cy = (yr[0] + yr[1]) / 2;
  return [
    [cx - (k * w) / 2, cx + (k * w) / 2],
    [cy - (k * h) / 2, cy + (k * h) / 2],
  ];
}
const START: Config = { x: 'links', y: 'length', ux: '1', uy: '1' };
const fname = (id: FeatureId) => FEATURES.find((f) => f.id === id)!.label;
const describe = (c: Config) => `x = ${fname(c.x)} ${UNIT_LABEL[c.ux]}, y = ${fname(c.y)} ${UNIT_LABEL[c.uy]}`;

function load(): Record<string, Config> {
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return {};
}

/** The twelve emails as points; wrong-neighbour emails ringed; the selected email joined to its nearest neighbour. */
function Points({ pts, wrong, sel, nn }: { pts: [number, number][]; wrong: boolean[]; sel: number; nn: { j: number; d: number }[] }) {
  const { x, y } = usePlot();
  return (
    <g>
      {pts.map((p, i) => (
        <g key={i}>
          {wrong[i] && <circle cx={x(p[0])} cy={y(p[1])} r={11} fill="none" stroke="var(--c-bad)" strokeWidth={2} strokeDasharray="3 2" />}
          <circle cx={x(p[0])} cy={y(p[1])} r={i === sel ? 8 : 6} fill={EMAILS[i].spam ? 'var(--c-b)' : 'var(--c-a)'} stroke={i === sel ? 'var(--text)' : 'var(--bg-elev)'} strokeWidth={2}>
            <title>{`${EMAILS[i].subject} (${EMAILS[i].spam ? 'spam' : 'not spam'})`}</title>
          </circle>
        </g>
      ))}
      {sel >= 0 && <circle cx={x(pts[nn[sel].j][0])} cy={y(pts[nn[sel].j][1])} r={10} fill="none" stroke="var(--text)" strokeWidth={1.5} />}
    </g>
  );
}

/** Post-solve map: every pair of measurements × every pair of units, shaded by score. */
function SolutionMap() {
  const s = useMemo(() => solve(), []);
  const pairs: [FeatureId, FeatureId][] = [];
  for (let a = 0; a < FEATURES.length; a++) for (let b = a + 1; b < FEATURES.length; b++) pairs.push([FEATURES[a].id, FEATURES[b].id]);
  const at = (x: FeatureId, y: FeatureId, ux: Unit, uy: Unit) => s.all.find((r) => r.c.x === x && r.c.y === y && r.c.ux === ux && r.c.uy === uy)!.s;
  return (
    <>
      <p style={{ margin: '0 0 8px' }}>
        All 375 settings, scored (15 pairs of measurements × 5 units for each axis). <strong>{s.perfect.length}</strong> reach 12 / 12, from only{' '}
        {new Set(s.perfect.map((r) => r.c.x + r.c.y)).size} pairs; {s.atLeast10.length} reach at least 10. Darker means more neighbours agree; ✓ marks 12 / 12. Columns run through the x unit, then the y unit (× 0.01, × 0.1, × 1, × 10, standardise).
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="num-table" style={{ fontSize: 11, borderCollapse: 'collapse' }}>
          <tbody>
            {pairs.map(([x, y]) => (
              <tr key={x + y}>
                <td style={{ whiteSpace: 'nowrap', paddingRight: 8 }}>
                  {fname(x)} / {fname(y)}
                </td>
                {UNITS.map((ux) =>
                  UNITS.map((uy) => {
                    const v = at(x, y, ux, uy);
                    return (
                      <td
                        key={ux + uy}
                        title={`${fname(x)} ${UNIT_LABEL[ux]}, ${fname(y)} ${UNIT_LABEL[uy]}: ${v} / 12`}
                        style={{ width: 14, height: 14, padding: 0, textAlign: 'center', background: `color-mix(in srgb, var(--c-good) ${Math.round((Math.max(0, v - 4) / 8) * 85)}%, transparent)`, borderLeft: uy === '0.01' ? '2px solid var(--border-strong)' : undefined, color: 'var(--text)' }}
                      >
                        {v === 12 ? '✓' : ''}
                      </td>
                    );
                  }),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function FeatureLab() {
  const [saved, setSaved] = useState<Record<string, Config>>(load);
  const [bi, setBi] = useState(0);
  const [sel, setSel] = useState(0);
  const progress = useLabProgress('vectors-lab');
  const log = useMachineLog();
  const brief = BRIEFS[bi];
  const c: Config = { ...START, ...saved[brief.id], ...(brief.locked ?? {}) };
  const pts = points(c);
  const nn = nearest(pts);
  const s = score(c);
  const wrong = nn.map((n, i) => c.x !== c.y && EMAILS[n.j].spam !== EMAILS[i].spam);
  const passing = brief.pass(c, s);

  const update = (next: Config) => {
    const all = { ...saved, [brief.id]: next };
    setSaved(all);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(all));
    } catch {
      /* ignore */
    }
    const ns = score(next);
    log.push(`${describe(next)} → ${next.x === next.y ? 'same measurement twice' : `${ns} / 12 neighbours agree`}${brief.pass(next, ns) ? ' · meets the brief' : ''}`, brief.pass(next, ns) ? 'good' : 'info');
  };

  // Both axes drawn to the same scale, so a distance on screen is the real distance.
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const [xd, yd] = equalScale([Math.min(...xs), Math.max(...xs)], [Math.min(...ys), Math.max(...ys)], PLOT_W - 36, PLOT_H - 36);
  const selN = nn[sel];
  const dx = pts[selN.j][0] - pts[sel][0];
  const dy = pts[selN.j][1] - pts[sel][1];
  const featOpts = FEATURES.map((f) => ({ value: f.id, label: f.label }));
  const unitOpts = UNITS.map((u) => ({ value: u, label: UNIT_LABEL[u] }));

  const spread = (v: number[]) => Math.sqrt(v.reduce((a, b) => a + (b - v.reduce((p, q) => p + q, 0) / v.length) ** 2, 0) / v.length);
  const ratio = spread(xs) / Math.max(spread(ys), 1e-12);
  const yLead = ratio < 0.1;
  const xLead = ratio > 10;
  const status =
    c.x === c.y
      ? 'Both axes show the same measurement. Pick two different ones.'
      : passing
        ? brief.id === 'nolinks'
          ? `12 / 12 without links: a second route. ${describe(c)}.`
          : `${s} / 12 neighbours agree. ${describe(c)}.`
        : `${s} / 12 neighbours agree. The ringed emails have a nearest neighbour with the other label.`;

  return (
    <LabShell
      title="Lab: The Feature Workbench"
      caption={
        <>
          Each dot is an email (orange: spam; blue: not spam; hover for its subject). Choose which two measurements become the axes, and the units of each. Both axes are drawn to the same scale, so what looks close is close. Click any email to see its nearest neighbour: the line joins them, and the readout shows how much of that distance comes from each axis. Your settings are saved in this browser.
        </>
      }
      briefs={BRIEFS.map((b) => ({ id: b.id, tab: b.tab, title: b.title, text: b.text }))}
      active={bi}
      onSelect={setBi}
      progress={progress}
      passing={passing}
      status={status}
      hint={
        <>
          Click one of the ringed emails and look at the readout: which axis supplies most of the distance to its nearest neighbour? If one measurement is counted in thousands and the other in single digits, the big one decides every distance on its own.
        </>
      }
      reveal={{ label: 'the map of all 375 settings', content: <SolutionMap /> }}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Change an axis or a unit: each change is scored here.' }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div className="controls">
          {!brief.locked && <Segmented label="x axis" options={featOpts} value={c.x} onChange={(v) => update({ ...c, x: v })} />}
          <Segmented label={`x units (${fname(c.x)})`} options={unitOpts} value={c.ux} onChange={(v) => update({ ...c, ux: v })} />
          {!brief.locked && <Segmented label="y axis" options={featOpts} value={c.y} onChange={(v) => update({ ...c, y: v })} />}
          <Segmented label={`y units (${fname(c.y)})`} options={unitOpts} value={c.uy} onChange={(v) => update({ ...c, uy: v })} />
          {brief.locked && <div className="muted" style={{ fontSize: 13 }}>Axes locked for this brief: x = {fname(brief.locked.x)}, y = {fname(brief.locked.y)}.</div>}
        </div>
        <div>
          <Plot
            x={xd}
            y={yd}
            width={PLOT_W}
            height={PLOT_H}
            xLabel={`${fname(c.x)} ${UNIT_LABEL[c.ux]}`}
            yLabel={`${fname(c.y)} ${UNIT_LABEL[c.uy]}`}
            ariaLabel="Twelve emails plotted by the chosen measurements"
            onPointerDown={(p) => {
              let best = 0;
              let bd = Infinity;
              pts.forEach((q, i) => {
                const d = Math.hypot((q[0] - p.x) / (xd[1] - xd[0]), (q[1] - p.y) / (yd[1] - yd[0]));
                if (d < bd) {
                  bd = d;
                  best = i;
                }
              });
              setSel(best);
            }}
          >
            <Seg from={pts[sel]} to={pts[selN.j]} color="var(--text)" width={2} dash="5 4" />
            <Points pts={pts} wrong={wrong} sel={sel} nn={nn} />
          </Plot>
          <div className="readout">
            <span className="k">selected </span>“{EMAILS[sel].subject}” ({EMAILS[sel].spam ? 'spam' : 'not spam'}){'\n'}
            <span className="k">nearest </span>“{EMAILS[selN.j].subject}” ({EMAILS[selN.j].spam ? 'spam' : 'not spam'}){' '}
            {EMAILS[selN.j].spam === EMAILS[sel].spam ? '✓' : '✗'}
            {'\n'}
            <span className="k">distance </span>
            {selN.d.toFixed(3)} = √(({dx.toFixed(3)})² + ({dy.toFixed(3)})²)
          </div>
          {c.x !== c.y && (yLead || xLead) && (
            <div className="live-note" style={{ marginTop: 8 }}>
              In these units the {yLead ? 'y' : 'x'} axis ({fname(yLead ? c.y : c.x)}) is spread about {Math.round(yLead ? 1 / ratio : ratio).toLocaleString()} times wider than the other, so it decides almost every distance on its own.
            </div>
          )}
        </div>
      </div>
    </LabShell>
  );
}
