import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Arrow, Clip, Dot, FnPath, Handle, Label, Plot, Seg } from '../../components/plot';
import { bceGrad, bceLoss, escape, trajectory, fitMAE, fitMSE, lineMAE, lineMSE, maeConst, mean, median, mseConst, sigmoid, sqGrad, sqLoss, type Pt } from './losses';

const f1 = (v: number) => v.toFixed(1);

// ---------------------------------------------------------------- tug of war (best constant)

const START_TIMES = [38, 34, 31, 30, 27, 25, 22];

export function TugOfWar() {
  const [ys, setYs] = useState(START_TIMES);
  const [c, setC] = useState(60);
  const [mode, setMode] = useState<'mse' | 'mae'>('mse');
  const mu = mean(ys);
  const med = median(ys);
  const n = ys.length;
  const ROPE = 14; // drawn length of a constant-force rope
  const pulls = ys.map((y) => (mode === 'mse' ? y - c : Math.sign(y - c) * ROPE));
  const net = mean(pulls);
  const loss = mode === 'mse' ? mseConst(ys, c) : maeConst(ys, c);
  const best = mode === 'mse' ? mu : med;
  const above = ys.filter((y) => y > c).length;
  const below = ys.filter((y) => y < c).length;
  const balanced = Math.abs(c - best) < 0.6;

  // Loss curves (each scaled to its own maximum so both fit).
  const maxMse = Math.max(mseConst(ys, 0), mseConst(ys, 200));
  const maxMae = Math.max(maeConst(ys, 0), maeConst(ys, 200));

  let note: React.ReactNode;
  if (mode === 'mse')
    note = balanced ? (
      <>
        <strong>Balanced.</strong> Every spring pulls in proportion to its length, and they cancel exactly at the <strong>mean</strong>, {f1(mu)} minutes.{' '}
        {Math.max(...ys) > 100 && 'Notice how far the single slow delivery has dragged it: its spring is so long that it outweighs all the others.'}
      </>
    ) : (
      <>
        Each data point is attached to your guess by a <strong>spring</strong>: the further away, the harder it pulls. The net pull (bottom arrow) points from your guess to{' '}
        {f1(mu)} — the mean. Drag the guess there.
      </>
    );
  else
    note = balanced ? (
      <>
        <strong>Balanced.</strong> Every rope pulls with the same force, so the guess settles where as many points lie above as below: the <strong>median</strong>, {f1(med)}{' '}
        minutes. {Math.max(...ys) > 100 && 'The slow delivery pulls no harder than any other point, however far away you drag it.'}
      </>
    ) : (
      <>
        Each point pulls with a <strong>rope of fixed strength</strong>, however far away it is. {above} point{above === 1 ? '' : 's'} above your guess, {below} below: the
        guess is pulled towards the bigger crowd until the two sides are equal.
      </>
    );

  return (
    <Widget
      title="A tug of war over one number"
      wide
      caption={
        <>
          Seven delivery times, in minutes. The model is as simple as possible: it predicts the <strong>same number</strong> for every delivery (the purple line). Drag the
          purple line to change the guess, and drag any orange point to change the data. Switch between the two losses and watch how each one pulls. Then drag the bottom
          delivery out to 180 minutes — a van broke down.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 6 }}>
        <Segmented
          label="Loss"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'mse', label: 'squared error (MSE)' },
            { value: 'mae', label: 'absolute error (MAE)' },
          ]}
        />
        <button className="btn small" onClick={() => setC(Math.round(best * 10) / 10)}>
          Jump to the best guess
        </button>
        <button className="btn small" onClick={() => setYs(START_TIMES)}>
          Reset data
        </button>
      </div>
      <Plot x={[-8, 208]} y={[-1.6, n - 0.4]} width={880} height={300} grid={false} axes={false} pad={14} ariaLabel="Data points pulling on a single guess">
        {({ x, y }) => (
          <g>
            {[0, 50, 100, 150, 200].map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={y(-1.6)} y2={y(n - 0.4)} className="plot-grid" />
                <text x={x(t)} y={y(-1.6) - 2} className="plot-tick" textAnchor="middle">
                  {t} min
                </text>
              </g>
            ))}
          </g>
        )}
        <Seg from={[mu, -1.4]} to={[mu, n - 0.5]} color="var(--c-a)" dash="4 4" width={1.5} />
        <Seg from={[med, -1.4]} to={[med, n - 0.5]} color="var(--c-c)" dash="4 4" width={1.5} />
        <Label at={[mu, n - 0.6]} dx={4} color="var(--c-a)" size={11.5}>mean</Label>
        <Label at={[med, n - 0.6]} dx={4} dy={13} color="var(--c-c)" size={11.5}>median</Label>
        {ys.map((_, i) =>
          Math.abs(pulls[i]) > 0.5 ? <Arrow key={`a${i}`} from={[c, i]} to={[c + pulls[i], i]} color={mode === 'mse' ? 'var(--c-a)' : 'var(--c-c)'} width={2} head={7} opacity={0.75} /> : null,
        )}
        {Math.abs(net) > 0.4 && <Arrow from={[c, -1]} to={[c + net * (mode === 'mse' ? 1 : 3), -1]} color="var(--c-grad)" width={3.5} head={10} />}
        <Label at={[c, -1]} dx={net >= 0 ? -6 : 6} dy={4} anchor={net >= 0 ? 'end' : 'start'} color="var(--c-grad)" size={11.5}>
          {balanced ? 'balanced' : 'net pull'}
        </Label>
        <Seg from={[c, -1.5]} to={[c, n - 0.5]} color="var(--accent)" width={3} />
        <Handle x={c} y={n - 0.5} onDrag={(a) => setC(Math.round(a * 10) / 10)} color="var(--accent)" label="the guess" />
        {ys.map((v, i) => (
          <Handle key={i} x={v} y={i} onDrag={(a) => setYs(ys.map((u, k) => (k === i ? Math.round(a) : u)))} color="var(--c-b)" r={6} label={`delivery ${i + 1}`} />
        ))}
      </Plot>
      <div className="widget-row" style={{ marginTop: 8 }}>
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={[0, 200]} y={[0, 1.05]} width={460} height={150} grid={false} tickLabels={false} xLabel="guess" ariaLabel="Both losses as the guess changes">
            <FnPath f={(t) => mseConst(ys, t) / maxMse} color="var(--c-a)" width={2.2} />
            <FnPath f={(t) => maeConst(ys, t) / maxMae} color="var(--c-c)" width={2.2} />
            <Dot at={[mu, mseConst(ys, mu) / maxMse]} r={4} color="var(--c-a)" />
            <Dot at={[med, maeConst(ys, med) / maxMae]} r={4} color="var(--c-c)" />
            <Seg from={[c, 0]} to={[c, 1.05]} color="var(--accent)" width={1.5} />
          </Plot>
          <div className="muted" style={{ fontSize: 12.5 }}>
            <span className="c-a">MSE</span> and <span className="c-c">MAE</span> for every possible guess (each scaled to fit); dots mark their lowest points.
          </div>
        </div>
        <div className="widget-col">
          <div className="readout">
            <span className="k">your guess   </span> {f1(c)} min{'\n'}
            <span className="k">{mode === 'mse' ? 'MSE' : 'MAE'} there    </span> <strong>{mode === 'mse' ? mseConst(ys, c).toFixed(1) : loss.toFixed(2)}</strong>
            {mode === 'mse' ? ' min²' : ' min'}
            {'\n'}
            <span className="k">mean         </span> <span className="c-a">{f1(mu)}</span>
            {'\n'}
            <span className="k">median       </span> <span className="c-c">{f1(med)}</span>
          </div>
          <div className="live-note" style={{ marginTop: 8 }}>
            {note}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ---------------------------------------------------------------- line fit with an outlier

const BASE_PTS: Pt[] = [
  [1, 15],
  [2, 21],
  [3, 24],
  [4, 31],
  [5, 34],
  [6, 41],
  [7, 44],
  [8, 51],
  [9, 55],
];

function FitLine({ w, b, color, dash }: { w: number; b: number; color: string; dash?: string }) {
  return <Seg from={[0, b]} to={[10, w * 10 + b]} color={color} width={2.5} dash={dash} />;
}

export function LineFit() {
  const [pts, setPts] = useState<Pt[]>(BASE_PTS);
  const [glitch, setGlitch] = useState(false);
  const [g, setG] = useState<Pt>([7.5, 5]);
  const all = useMemo(() => (glitch ? [...pts, g] : pts), [pts, glitch, g]);
  const l2 = useMemo(() => fitMSE(all), [all]);
  const l1 = useMemo(() => fitMAE(all), [all]);
  const clean2 = useMemo(() => fitMSE(pts), [pts]);
  const at = (l: { w: number; b: number }, x: number) => l.w * x + l.b;
  const shift2 = Math.abs(at(l2, 5) - at(clean2, 5));
  const shift1 = Math.abs(at(l1, 5) - at(fitMAE(pts), 5));
  return (
    <Widget
      title="One bad reading"
      wide
      caption={
        <>
          Delivery time against distance. Two lines are fitted to the same points, each the best possible under its loss: <span className="c-a">squared error</span> and{' '}
          <span className="c-c">absolute error</span>. Switch on the <strong>glitch</strong> — a GPS fault that logged a long delivery as 5 minutes — and drag it around.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 440px' }}>
          <Plot x={[0, 10]} y={[0, 80]} width={460} height={300} xLabel="distance (km)" yLabel="minutes" ariaLabel="Points with two fitted lines">
            <Clip>
              <FitLine w={l2.w} b={l2.b} color="var(--c-a)" />
              <FitLine w={l1.w} b={l1.b} color="var(--c-c)" dash="7 4" />
            </Clip>
            {pts.map((p, i) => (
              <Handle key={i} x={p[0]} y={p[1]} onDrag={(a, b) => setPts(pts.map((q, k) => (k === i ? [Math.round(a * 10) / 10, Math.round(b)] : q)))} color="var(--c-b)" r={6} label={`point ${i + 1}`} />
            ))}
            {glitch && <Handle x={g[0]} y={g[1]} onDrag={(a, b) => setG([Math.round(a * 10) / 10, Math.round(b)])} color="var(--c-bad)" r={7} label="the glitch" />}
          </Plot>
        </div>
        <div className="widget-col controls">
          <Toggle label="Add the glitch" checked={glitch} onChange={setGlitch} />
          <div className="readout">
            <span className="c-a">squared-error line</span>
            {'\n'}
            <span className="k">  slope {l2.w.toFixed(2)} min/km, MSE {lineMSE(all, l2).toFixed(1)}</span>
            {'\n'}
            <span className="c-c">absolute-error line</span>
            {'\n'}
            <span className="k">  slope {l1.w.toFixed(2)} min/km, MAE {lineMAE(all, l1).toFixed(2)}</span>
          </div>
          <div className="live-note">
            {glitch ? (
              <>
                At 5 km, the glitch has moved the squared-error line by <strong>{shift2.toFixed(1)} min</strong> and the absolute-error line by{' '}
                <strong>{shift1.toFixed(1)} min</strong>. Squaring turns the glitch’s big error into a huge one, and the line bends towards it to shrink it.
              </>
            ) : (
              <>On clean data the two lines nearly agree: when nothing is wildly off, both losses want the same thing. Now add the glitch.</>
            )}
          </div>
          <button className="btn small" onClick={() => setPts(BASE_PTS)}>
            Reset points
          </button>
        </div>
      </div>
    </Widget>
  );
}

// ---------------------------------------------------------------- stuck when wrong

export function StuckWhenWrong() {
  const [z0, setZ0] = useState(-6);
  const [lr, setLr] = useState(1);
  const ce = useMemo(() => escape('ce', z0, lr), [z0, lr]);
  const sq = useMemo(() => escape('mse', z0, lr), [z0, lr]);
  const T = Math.max(Math.round(Math.min(sq.steps, 5000) * 1.15), 10);
  const ceTraj = useMemo(() => trajectory('ce', z0, lr, T), [z0, lr, T]);
  const sqTraj = useMemo(() => trajectory('mse', z0, lr, T), [z0, lr, T]);
  const pAt = (path: number[], t: number) => sigmoid(path[Math.min(path.length - 1, Math.max(0, Math.round(t)))]);
  const gS = sqGrad(z0);
  const gC = bceGrad(z0);
  return (
    <Widget
      title="Stuck when wrong"
      wide
      caption={
        <>
          The true answer is “spam”. The model’s score for spam is <em>z</em>, and its probability is σ(z). Left: how hard each loss pushes on the score (the size of its
          slope) at every possible score. Drag the purple start point far to the left — a model that is confidently wrong — and see how long gradient descent takes to fix
          it under each loss (right).
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <div className="field-label">Push on the score (size of the slope)</div>
          <Plot x={[-9, 4]} y={[0, 1.05]} width={420} height={230} xLabel="score z" ariaLabel="Gradient size for each loss">
            <FnPath f={(z) => Math.abs(bceGrad(z))} color="var(--c-a)" width={2.5} />
            <FnPath f={(z) => Math.abs(sqGrad(z))} color="var(--c-b)" width={2.5} />
            <Dot at={[z0, Math.abs(gC)]} r={5} color="var(--c-a)" />
            <Dot at={[z0, Math.abs(gS)]} r={5} color="var(--c-b)" />
            <Handle x={z0} y={0} onDrag={(a) => setZ0(Math.max(-9, Math.min(-0.5, Math.round(a * 10) / 10)))} color="var(--accent)" label="starting score" />
          </Plot>
          <div className="mono" style={{ fontSize: 12.5 }}>
            at z = {z0.toFixed(1)}: p = {sigmoid(z0).toFixed(4)} · <span className="c-a">cross-entropy push {Math.abs(gC).toFixed(3)}</span> ·{' '}
            <span className="c-b">squared-error push {Math.abs(gS).toFixed(4)}</span>
          </div>
        </div>
        <div className="widget-col" style={{ flex: '1 1 340px' }}>
          <div className="field-label">Probability of the right answer, step by step</div>
          <Plot x={[0, T]} y={[0, 1]} width={360} height={230} xLabel="steps" ariaLabel="Training progress under each loss">
            <FnPath f={(t) => pAt(sqTraj, t)} color="var(--c-b)" width={2.5} samples={Math.min(600, T * 2)} />
            <FnPath f={(t) => pAt(ceTraj, t)} color="var(--c-a)" width={2.5} samples={Math.min(600, T * 2)} />
            <Seg from={[0, 0.5]} to={[T, 0.5]} dash="3 3" />
          </Plot>
          <Slider label="step size" value={lr} min={0.2} max={5} step={0.1} onChange={setLr} format={(v) => v.toFixed(1)} />
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 8 }}>
        Steps until the model gives the right answer more than 50%: <strong className="c-a">cross-entropy {ce.steps}</strong>,{' '}
        <strong className="c-b">
          squared error {sq.escaped ? sq.steps : `more than ${sq.steps}`}
        </strong>
        .{' '}
        {z0 < -3
          ? `The model is confidently wrong, so the sigmoid is nearly flat here. Squared error’s push includes that flatness as a factor and almost vanishes — ${(Math.abs(gC) / Math.abs(gS)).toFixed(0)}× weaker than cross-entropy’s, which cancels it out.`
          : 'Close to 50–50 the two losses push with similar strength. The difference only becomes dramatic when the model is confidently wrong.'}
      </div>
    </Widget>
  );
}

export { sqLoss, bceLoss };
