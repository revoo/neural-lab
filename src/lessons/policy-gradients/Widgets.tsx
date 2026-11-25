// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState } from "react";
import { Widget } from "../../components/Widget";
import { Segmented, Slider } from "../../components/controls";
import {
  Clip,
  Dot,
  FnPath,
  Label,
  Plot,
  Seg,
  usePlot,
} from "../../components/plot";
import {
  CORRIDOR_P_STAR,
  ESTIMATOR_LABELS,
  LAB_ALPHAS,
  LAB_BRIEFS,
  LAB_ESTIMATORS,
  PPO_FAIL,
  PPO_MEANS,
  PUSH_ACTIONS,
  PUSH_MEANS,
  PUSH_SIGMA,
  RACE_SEEDS,
  clipInactive,
  clippedObjective,
  corridorReinforce,
  corridorValues,
  drawBatch,
  estimatorStats,
  evaluateLab,
  expectedPPOReward,
  hallwayOptimalReturn,
  meanOf,
  ppoRobustness,
  raceSummary,
  reuseBatch,
  runHallway,
  sampleIndex,
  seededRng,
  simulateCorridor,
  softmax,
  solveLabExhaustive,
  stdOf,
  wrongWayProbability,
  type Estimator,
  type LabBriefId,
  type LabConfig,
} from "./pg";

const fmt = (v: number, d = 2) =>
  Number.isFinite(v) ? v.toFixed(d) : v > 0 ? "+∞" : "−∞";
const signed = (v: number, d = 2) =>
  (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

/** Moving average for smoothing noisy learning curves (window w, trailing). */
function smooth(xs: number[], w: number): number[] {
  const out: number[] = [];
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    s += xs[i];
    if (i >= w) s -= xs[i - w];
    out.push(s / Math.min(i + 1, w));
  }
  return out;
}

function Polyline({
  pts,
  color,
  width = 2,
  dashed = false,
  opacity = 1,
}: {
  pts: [number, number][];
  color: string;
  width?: number;
  dashed?: boolean;
  opacity?: number;
}) {
  const { x, y } = usePlot();
  return (
    <polyline
      points={pts
        .map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`)
        .join(" ")}
      fill="none"
      stroke={color}
      strokeWidth={width}
      strokeDasharray={dashed ? "5 4" : undefined}
      strokeLinejoin="round"
      opacity={opacity}
    />
  );
}

function Cloud({ pts, colors }: { pts: [number, number][]; colors: string[] }) {
  const { x, y } = usePlot();
  return (
    <g>
      {pts.map(([a, b], i) => (
        <circle
          key={i}
          cx={x(a)}
          cy={y(b)}
          r={3}
          fill={colors[i]}
          opacity={0.55}
        />
      ))}
    </g>
  );
}

// ------------------------------------------------------------------ 1. The aliased corridor

export function AliasedCorridor() {
  const [p, setP] = useState(0.15);
  const [reveal, setReveal] = useState(false);
  const [episode, setEpisode] = useState<{
    squares: number[];
    reached: boolean;
  } | null>(null);
  const [epSeed, setEpSeed] = useState(1);
  const [trace, setTrace] = useState<number[] | null>(null);
  const [frame, setFrame] = useState(0);
  const [learnSeed, setLearnSeed] = useState(1);
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current) window.clearInterval(timer.current);
    },
    [],
  );

  const learn = () => {
    if (timer.current) window.clearInterval(timer.current);
    const tr = corridorReinforce(0.03, 300, learnSeed, 0.15);
    setLearnSeed((s) => s + 1);
    setTrace(tr);
    setFrame(0);
    setP(tr[0]);
    let f = 0;
    timer.current = window.setInterval(() => {
      f = Math.min(tr.length - 1, f + 3);
      setFrame(f);
      setP(Math.round(tr[f] * 1000) / 1000);
      if (f >= tr.length - 1 && timer.current) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
    }, 30);
  };

  const runEpisode = () => {
    const rng = seededRng(epSeed * 31 + 5);
    setEpSeed((s) => s + 1);
    setEpisode(simulateCorridor(p, rng, 60));
  };

  const v = corridorValues(p).v1;
  const vStar = corridorValues(CORRIDOR_P_STAR).v1;
  const learning = trace !== null && frame < trace.length - 1;

  // corridor drawing
  const cell = 100;
  const W = 4 * cell + 40;
  const squares = [1, 2, 3, 4];
  const last = episode ? episode.squares[episode.squares.length - 1] : 1;

  let note: React.ReactNode;
  if (p <= 0)
    note = (
      <>
        Always <strong>Left</strong>: the agent bumps into the left wall
        forever. It never reaches the goal, so its return is −∞.
      </>
    );
  else if (p >= 1)
    note = (
      <>
        Always <strong>Right</strong>: square 1 → square 2, where Right is
        secretly reversed and sends it back to square 1… forever. Return −∞.
      </>
    );
  else
    note = (
      <>
        With P(Right) = <strong>{p.toFixed(2)}</strong> the agent takes{" "}
        <strong>{fmt(-v, 1)}</strong> steps on average (value {signed(v, 2)}).{" "}
        {Math.abs(p - CORRIDOR_P_STAR) < 0.02
          ? "That is as good as any policy can do here: the best mix is P(Right) = 2 − √2 ≈ 0.586."
          : `The best possible mix, P(Right) ≈ 0.586, takes ${fmt(-vStar, 2)} steps.`}
      </>
    );

  return (
    <Widget
      title="The aliased corridor: when the best policy is a coin flip"
      wide
      caption={
        <>
          All three grey squares look identical to the agent, so whatever it
          does, it must do the same thing in each. Drag P(Right) to 0 or 1 (a
          deterministic policy, which is all an argmax can give you) and the
          agent never arrives. Then press{" "}
          <strong>Let REINFORCE learn it</strong> and watch the policy gradient
          climb the curve from P(Right) = 0.15 to near the peak, one episode at
          a time.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} 110`}
        width={W}
        height={110}
        style={{ margin: "0 auto" }}
        role="img"
        aria-label="Corridor of three identical squares and a goal"
      >
        {squares.map((s, i) => {
          const x0 = 20 + i * cell;
          const isGoal = s === 4;
          return (
            <g key={s}>
              <rect
                x={x0 + 3}
                y={18}
                width={cell - 6}
                height={56}
                rx={8}
                fill={
                  isGoal
                    ? "color-mix(in srgb, var(--c-good) 22%, transparent)"
                    : "var(--bg-sunken)"
                }
                stroke={last === s ? "var(--accent)" : "var(--border-strong)"}
                strokeWidth={last === s ? 3 : 1.5}
              />
              <text
                x={x0 + cell / 2}
                y={42}
                textAnchor="middle"
                fontSize={14}
                fill="var(--text-muted)"
              >
                {isGoal ? "goal" : `square ${s}`}
              </text>
              {!isGoal && (
                <text
                  x={x0 + cell / 2}
                  y={62}
                  textAnchor="middle"
                  fontSize={12}
                  fill={
                    reveal && s === 2 ? "var(--c-bad)" : "var(--text-faint)"
                  }
                >
                  {reveal
                    ? s === 2
                      ? "R ← · L →"
                      : s === 1
                        ? "L: wall"
                        : "R → goal"
                    : "looks the same"}
                </text>
              )}
            </g>
          );
        })}
        <text x={20} y={96} fontSize={12} fill="var(--text-muted)">
          −1 reward per step · start in square 1
        </text>
      </svg>

      <div className="btn-row" style={{ margin: "6px 0 10px" }}>
        <button
          type="button"
          className="btn"
          onClick={() => setReveal((r) => !r)}
        >
          {reveal ? "Hide the secret" : "Reveal what the agent can’t see"}
        </button>
        <button
          type="button"
          className="btn"
          onClick={runEpisode}
          disabled={learning}
        >
          Run one episode
        </button>
        <button
          type="button"
          className="btn primary"
          onClick={learn}
          disabled={learning}
        >
          Let REINFORCE learn it
        </button>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(min(260px, 100%), 1fr))",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div>
          <Slider
            label="P(Right) — the whole policy"
            value={p}
            min={0}
            max={1}
            step={0.01}
            onChange={(x) => {
              if (!learning) setP(x);
            }}
            color="accent"
          />
          <Plot
            width={420}
            height={250}
            x={[0, 1]}
            y={[0, 100]}
            xLabel="P(Right)"
            yLabel="average steps to the goal"
            ariaLabel="Expected steps to the goal against P(Right)"
          >
            <Clip>
              <FnPath
                f={(q) => Math.min(140, -corridorValues(q).v1)}
                color="var(--accent)"
                domain={[0.005, 0.995]}
              />
            </Clip>
            {({ x, y }) => (
              <line
                x1={x(CORRIDOR_P_STAR)}
                x2={x(CORRIDOR_P_STAR)}
                y1={y(0)}
                y2={y(100)}
                stroke="var(--c-good)"
                strokeDasharray="4 4"
                strokeWidth={1.5}
              />
            )}
            <Label
              at={[CORRIDOR_P_STAR, 92]}
              dx={5}
              color="var(--c-good)"
              size={12}
            >
              best 0.586
            </Label>
            <Dot
              at={[0.05, -corridorValues(0.05).v1]}
              r={4}
              color="var(--c-b)"
            />
            <Dot
              at={[0.95, -corridorValues(0.95).v1]}
              r={4}
              color="var(--c-b)"
            />
            <Label
              at={[0.95, -corridorValues(0.95).v1]}
              dx={-8}
              dy={4}
              anchor="end"
              color="var(--c-b)"
              size={11.5}
            >
              ε-greedy, ε = 0.1
            </Label>
            {trace && (
              <Polyline
                pts={trace
                  .slice(0, frame + 1)
                  .map(
                    (q) =>
                      [q, Math.min(100, -corridorValues(q).v1)] as [
                        number,
                        number,
                      ],
                  )}
                color="var(--c-grad)"
                width={1.2}
                opacity={0.6}
              />
            )}
            {p > 0 && p < 1 && -v < 100 && (
              <Dot
                at={[p, -v]}
                r={6}
                color="var(--accent)"
                stroke="var(--bg)"
              />
            )}
          </Plot>
        </div>
        <div style={{ display: "grid", gap: 10 }}>
          <div className="live-note">{note}</div>
          {episode && (
            <div className="readout">
              <span className="k">last episode </span>
              {episode.squares.length - 1} steps
              {episode.reached ? "" : " (gave up after 60)"}
              {"\n"}
              <span className="k">path </span>
              {episode.squares
                .slice(0, 24)
                .map((s) => (s === 4 ? "G" : String(s)))
                .join("→")}
              {episode.squares.length > 24 ? "→…" : ""}
            </div>
          )}
          {trace && (
            <div className="readout">
              <span className="k">REINFORCE episode </span>
              {frame} / {trace.length - 1}
              {"\n"}
              <span className="k">P(Right) </span>
              {trace[frame].toFixed(3)}
              {"\n"}
              <span className="k">
                step size α = 0.03, baseline = running average return
              </span>
            </div>
          )}
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. The push explorer

const PRESETS: Record<string, number[]> = {
  uniform: [0, 0, 0],
  leaning: [0, 0.5, 1],
};
const ACTION_COLORS = ["var(--c-b)", "var(--text-muted)", "var(--c-a)"];

export function PushExplorer() {
  const [offset, setOffset] = useState(0);
  const [mode, setMode] = useState<"none" | "baseline">("none");
  const [preset, setPreset] = useState<"uniform" | "leaning">("uniform");
  const [draws, setDraws] = useState<{ a: number; r: number; g: number[] }[]>(
    [],
  );
  const [counter, setCounter] = useState(1);

  const theta = PRESETS[preset];
  const probs = softmax(theta);
  const expectedReward = probs.reduce((s, q, i) => s + q * PUSH_MEANS[i], 0);
  const b = mode === "baseline" ? expectedReward + offset : 0;
  const st = estimatorStats(theta, PUSH_MEANS, PUSH_SIGMA, offset, b);
  const wrong = wrongWayProbability(
    theta,
    PUSH_MEANS,
    PUSH_SIGMA,
    offset,
    b,
    2,
  );

  const draw = (n: number) => {
    const rng = seededRng(counter * 7919 + 13);
    setCounter((c) => c + 1);
    const out: { a: number; r: number; g: number[] }[] = [];
    for (let i = 0; i < n; i++) {
      const a = sampleIndex(probs, rng);
      const r = rng.normal(PUSH_MEANS[a] + offset, PUSH_SIGMA);
      out.push({
        a,
        r,
        g: probs.map((q, k) => (r - b) * ((k === a ? 1 : 0) - q)),
      });
    }
    setDraws((d) => [...d, ...out].slice(-300));
  };

  // reset cloud when the setting changes, so the cloud always matches the readout
  useEffect(() => setDraws([]), [offset, mode, preset]);

  const lastDraw = draws[draws.length - 1];
  const sampleMean = draws.length
    ? [0, 1, 2].map((k) => meanOf(draws.map((d) => d.g[k])))
    : null;
  const lim = 40;

  return (
    <Widget
      title="The push explorer: what one sampled action does to the logits"
      wide
      caption={
        <>
          Three actions with average rewards 1, 2 and 3 (noise ±1). Each draw
          samples an action, pays a reward and pushes all three logits by
          (reward − baseline) × (e<sub>a</sub> − π). Draw 100 samples with no
          offset, then slide the <strong>offset</strong> to +50 and draw again:
          the true gradient (the green dot) doesn’t move, but the cloud of
          single-sample pushes explodes. Now switch on the baseline.
        </>
      }
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 14,
          alignItems: "end",
          marginBottom: 10,
        }}
      >
        <div style={{ minWidth: 220, flex: "1 1 220px" }}>
          <Slider
            label="Reward offset added to every action"
            value={offset}
            min={0}
            max={50}
            step={1}
            onChange={setOffset}
            format={(v) => `+${v}`}
            color="c-b"
          />
        </div>
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: "none", label: "b = 0" },
            { value: "baseline", label: "b = average reward" },
          ]}
        />
        <Segmented
          value={preset}
          onChange={setPreset}
          options={[
            { value: "uniform", label: "π uniform" },
            { value: "leaning", label: "π leaning Great" },
          ]}
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(min(270px, 100%), 1fr))",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div>
          <Plot
            width={360}
            height={360}
            x={[-lim, lim]}
            y={[-lim, lim]}
            xLabel="push on Great"
            yLabel="push on Dull"
            ariaLabel="Cloud of single-sample gradient estimates"
          >
            <Clip>
              <Cloud
                pts={draws.map((d) => [d.g[2], d.g[0]] as [number, number])}
                colors={draws.map((d) => ACTION_COLORS[d.a])}
              />
            </Clip>
            {sampleMean && (
              <Dot
                at={[sampleMean[2], sampleMean[0]]}
                r={5}
                color="none"
                stroke="var(--text)"
              />
            )}
            <Dot
              at={[st.expected[2], st.expected[0]]}
              r={5}
              color="var(--c-good)"
              stroke="var(--bg)"
            />
          </Plot>
          <div
            className="btn-row"
            style={{ justifyContent: "center", marginTop: 4 }}
          >
            <button type="button" className="btn" onClick={() => draw(1)}>
              Draw 1
            </button>
            <button
              type="button"
              className="btn primary"
              onClick={() => draw(100)}
            >
              Draw 100
            </button>
            <button type="button" className="btn" onClick={() => setDraws([])}>
              Clear
            </button>
          </div>
          <div
            style={{
              fontSize: 12.5,
              color: "var(--text-muted)",
              textAlign: "center",
              marginTop: 6,
            }}
          >
            dots coloured by the action sampled:{" "}
            <span style={{ color: "var(--c-b)" }}>Dull</span> ·{" "}
            <span style={{ color: "var(--text-muted)" }}>Decent</span> ·{" "}
            <span style={{ color: "var(--c-a)" }}>Great</span>; green = true
            gradient; ring = average of your draws
          </div>
        </div>

        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ overflowX: "auto" }}>
            <table className="num-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>if sampled…</th>
                  <th>prob.</th>
                  <th>reward − baseline</th>
                  <th>Dull</th>
                  <th>Decent</th>
                  <th>Great</th>
                </tr>
              </thead>
              <tbody>
                {PUSH_ACTIONS.map((name, a) => (
                  <tr key={name}>
                    <td
                      style={{
                        textAlign: "left",
                        color: ACTION_COLORS[a],
                        fontFamily: "var(--font-ui)",
                      }}
                    >
                      {name}
                    </td>
                    <td>{st.probs[a].toFixed(2)}</td>
                    <td>{signed(PUSH_MEANS[a] + offset - b, 1)}</td>
                    {st.pushes[a].map((g, k) => (
                      <td
                        key={k}
                        style={{
                          color:
                            g > 1e-9
                              ? "var(--c-good)"
                              : g < -1e-9
                                ? "var(--c-bad)"
                                : "var(--text-faint)",
                          fontWeight: k === a ? 700 : 400,
                        }}
                      >
                        {signed(g, 2)}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr>
                  <td
                    style={{ textAlign: "left", fontFamily: "var(--font-ui)" }}
                  >
                    expected
                  </td>
                  <td />
                  <td />
                  {st.expected.map((g, k) => (
                    <td
                      key={k}
                      style={{ color: "var(--c-good)", fontWeight: 700 }}
                    >
                      {signed(g, 3)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 12.5, color: "var(--text-muted)" }}>
            Pushes shown at each action’s average reward; real draws add noise.
          </div>
          <div className="readout">
            <span className="k">baseline b </span>
            {b.toFixed(2)}
            {"\n"}
            <span className="k">variance of one sample </span>
            {st.variance.toFixed(2)}
            {"\n"}
            <span className="k">single draws pushing Great the wrong way </span>
            {(wrong * 100).toFixed(0)}%
            {lastDraw && (
              <>
                {"\n"}
                <span className="k">last draw </span>
                {PUSH_ACTIONS[lastDraw.a]}, reward {lastDraw.r.toFixed(1)}
              </>
            )}
          </div>
          <div className="live-note">
            {mode === "none" && offset >= 10 ? (
              <>
                Every reward is now positive, so{" "}
                <strong>every sampled action gets pushed up</strong> — Dull
                included (its own logit moves by {signed(st.pushes[0][0], 1)}).
                The pushes only <em>average</em> out to the right answer:{" "}
                {(wrong * 100).toFixed(0)}% of single draws push Great the wrong
                way, and the variance is{" "}
                {(
                  st.variance /
                  estimatorStats(
                    theta,
                    PUSH_MEANS,
                    PUSH_SIGMA,
                    offset,
                    st.probs.reduce((s, q, i) => s + q * PUSH_MEANS[i], 0) +
                      offset,
                  ).variance
                ).toFixed(0)}
                × what a baseline would give.
              </>
            ) : mode === "baseline" ? (
              <>
                Each reward is now compared with what the policy{" "}
                <strong>usually</strong> gets ({b.toFixed(2)}). Better than
                usual → pushed up; worse → pushed down. The expected gradient is
                identical to the b = 0 case — the baseline costs nothing in bias
                — but the variance is just {st.variance.toFixed(2)}, whatever
                the offset.
              </>
            ) : (
              <>
                With rewards 1, 2 and 3 every action still gets pushed up when
                sampled; Great just gets pushed hardest. On average that’s the
                right direction ({signed(st.expected[2], 3)} for Great), but{" "}
                {(wrong * 100).toFixed(0)}% of single draws push Great’s logit
                down. Now add an offset.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. The hallway race

const RACE_COLORS: Record<Estimator, string> = {
  none: "var(--c-b)",
  average: "var(--text-muted)",
  state: "var(--c-a)",
  td: "var(--c-good)",
};

export function HallwayRace() {
  const [view, setView] = useState<"avg" | "seed">("avg");
  const [seed, setSeed] = useState(1);
  const race = useMemo(() => raceSummary(), []);
  const optimum = hallwayOptimalReturn();
  const seedRuns = useMemo(
    () => race.map((r) => runHallway(r.est, r.alpha, r.beta, 150, seed)),
    [race, seed],
  );

  const curves =
    view === "avg"
      ? race.map((r) => smooth(r.curve, 10))
      : seedRuns.map((r) => smooth(r.returns, 10));

  return (
    <Widget
      title="The long noisy hallway: REINFORCE vs baseline vs actor-critic"
      wide
      caption={
        <>
          Eight squares to an exit worth +10; each step costs 0.5 on average but
          is jittered by noise of ±2. Each method uses its own best step size α
          (and critic rate β), found by searching {`{0.002 … 0.5}`} ×{" "}
          {`{0.05 … 0.5}`} and averaged over {RACE_SEEDS} fixed seeds. Curves
          are smoothed over 10 episodes. Switch to <strong>one seed</strong> and
          step through seeds to see how much luck each method depends on.
        </>
      }
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          alignItems: "end",
          marginBottom: 10,
        }}
      >
        <Segmented
          value={view}
          onChange={setView}
          options={[
            { value: "avg", label: `Average of ${RACE_SEEDS} seeds` },
            { value: "seed", label: "One seed" },
          ]}
        />
        {view === "seed" && (
          <div className="btn-row">
            <button
              type="button"
              className="btn small"
              onClick={() => setSeed((s) => (s <= 1 ? RACE_SEEDS : s - 1))}
            >
              ◀
            </button>
            <span className="mono">seed {seed}</span>
            <button
              type="button"
              className="btn small"
              onClick={() => setSeed((s) => (s >= RACE_SEEDS ? 1 : s + 1))}
            >
              ▶
            </button>
          </div>
        )}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(min(280px, 100%), 1fr))",
          gap: 16,
          alignItems: "start",
        }}
      >
        <Plot
          width={460}
          height={260}
          x={[0, 150]}
          y={[-15, 6]}
          xLabel="episode"
          yLabel="return"
          ariaLabel="Learning curves for three policy-gradient methods"
        >
          {({ x, y }) => (
            <line
              x1={x(0)}
              x2={x(150)}
              y1={y(optimum)}
              y2={y(optimum)}
              stroke="var(--text-faint)"
              strokeDasharray="5 4"
            />
          )}
          <Label
            at={[150, optimum]}
            dy={-6}
            anchor="end"
            color="var(--text-faint)"
            size={11.5}
          >{`walk straight to the exit (${optimum.toFixed(2)})`}</Label>
          <Clip>
            {curves.map((c, i) => (
              <Polyline
                key={race[i].est}
                pts={c.map((v, t) => [t + 1, v] as [number, number])}
                color={RACE_COLORS[race[i].est]}
                width={2.2}
              />
            ))}
          </Clip>
        </Plot>
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ overflowX: "auto" }}>
            <table className="num-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>method</th>
                  <th>step / critic</th>
                  <th>mean return ± seed spread</th>
                  <th>advantage spread</th>
                </tr>
              </thead>
              <tbody>
                {race.map((r) => (
                  <tr key={r.est}>
                    <td
                      style={{
                        textAlign: "left",
                        color: RACE_COLORS[r.est],
                        fontFamily: "var(--font-ui)",
                        fontWeight: 600,
                      }}
                    >
                      {ESTIMATOR_LABELS[r.est]}
                    </td>
                    <td>
                      {r.est === "none" ? r.alpha : `${r.alpha} / ${r.beta}`}
                    </td>
                    <td>
                      {r.score.toFixed(2)} ± {stdOf(r.scores).toFixed(2)}
                    </td>
                    <td>{r.advSpread.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 12.5, color: "var(--text-muted)" }}>
            Mean return over all 150 training episodes; ± is the standard
            deviation across the {RACE_SEEDS} seeds. “Advantage spread” is the
            standard deviation of the number each method multiplies ∇log π by.
          </div>
          <div style={{ display: "grid", gap: 4 }}>
            <div style={{ fontSize: 12.5, color: "var(--text-muted)" }}>
              Learned P(Right) in each square after 150 episodes (seed {seed}):
            </div>
            {seedRuns.map((run, i) => (
              <div
                key={race[i].est}
                style={{ display: "flex", gap: 3, alignItems: "center" }}
              >
                <span
                  style={{
                    width: 110,
                    fontSize: 12,
                    color: RACE_COLORS[race[i].est],
                  }}
                >
                  {ESTIMATOR_LABELS[race[i].est]}
                </span>
                {run.pRight.map((q, k) => (
                  <span
                    key={k}
                    title={`square ${k + 1}: P(Right) = ${q.toFixed(2)}`}
                    style={{
                      flex: "1 1 0",
                      height: 20,
                      borderRadius: 3,
                      background: `color-mix(in srgb, ${RACE_COLORS[race[i].est]} ${Math.round(q * 100)}%, var(--bg-sunken))`,
                      border: "1px solid var(--border)",
                    }}
                  />
                ))}
              </div>
            ))}
          </div>
          <div className="live-note">
            {view === "avg" ? (
              <>
                Actor-critic learns fastest <strong>and</strong> most reliably
                (seed spread {stdOf(race[2].scores).toFixed(2)} vs{" "}
                {stdOf(race[0].scores).toFixed(2)}). The per-state baseline
                doesn’t help here: its difference from no baseline is smaller
                than the seed-to-seed noise. The noise in this hallway lives in
                every step’s reward, and a baseline can’t cancel that — only the
                critic’s one-step TD error can.
              </>
            ) : (
              <>
                Seed {seed}: average return over the run — no baseline{" "}
                {meanOf(seedRuns[0].returns).toFixed(2)}, per-state baseline{" "}
                {meanOf(seedRuns[1].returns).toFixed(2)}, actor-critic{" "}
                {meanOf(seedRuns[2].returns).toFixed(2)}. Step through seeds:
                the Monte-Carlo methods sometimes do brilliantly and sometimes
                stall; actor-critic is boringly consistent.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4a. The clipped objective

export function ClipObjective() {
  const [sign, setSign] = useState<"good" | "bad">("good");
  const [eps, setEps] = useState(0.2);
  const A = sign === "good" ? 1 : -1;
  return (
    <Widget
      title="PPO’s clip: a fence around the old policy"
      caption={
        <>
          The horizontal axis is the ratio r = π<sub>new</sub>(a) / π
          <sub>old</sub>(a): how much more (or less) likely the new policy makes
          an action the batch tried. The dashed line is the unclipped objective
          r·A; the solid line is PPO’s. Where the solid line goes flat, the
          gradient is zero — that sample stops pulling. Flip the advantage’s
          sign and watch which side the fence moves to.
        </>
      }
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 14,
          alignItems: "end",
          marginBottom: 8,
        }}
      >
        <Segmented
          value={sign}
          onChange={setSign}
          options={[
            { value: "good", label: "better than usual (A = +1)" },
            { value: "bad", label: "worse than usual (A = −1)" },
          ]}
        />
        <div style={{ minWidth: 180, flex: "1 1 180px" }}>
          <Slider
            label="clip range ε"
            value={eps}
            min={0.05}
            max={0.5}
            step={0.05}
            onChange={setEps}
            color="accent"
          />
        </div>
      </div>
      <Plot
        width={440}
        height={250}
        x={[0, 2]}
        y={[-2.2, 2.2]}
        xLabel="ratio r"
        yLabel="objective"
        ariaLabel="Clipped and unclipped PPO objective"
      >
        {({ x, y }) => {
          const lo = sign === "good" ? 1 + eps : 0;
          const hi = sign === "good" ? 2 : 1 - eps;
          return (
            <rect
              x={x(lo)}
              y={y(2.2)}
              width={x(hi) - x(lo)}
              height={y(-2.2) - y(2.2)}
              fill="color-mix(in srgb, var(--c-bad) 10%, transparent)"
            />
          );
        }}
        <FnPath f={(r) => r * A} color="var(--text-faint)" dashed width={1.8} />
        <FnPath
          f={(r) => clippedObjective(r, A, eps)}
          color="var(--accent)"
          width={3}
        />
        <Seg
          from={[1, -2.2]}
          to={[1, 2.2]}
          color="var(--text-faint)"
          dash="2 3"
        />
        <Label
          at={[sign === "good" ? 1 + eps : 0.04, sign === "good" ? -1.6 : 1.6]}
          dx={4}
          color="var(--c-bad)"
          size={11.5}
        >
          flat: no more push
        </Label>
      </Plot>
      <div className="live-note" style={{ marginTop: 8 }}>
        {sign === "good" ? (
          <>
            For an action that went <strong>better</strong> than usual, PPO
            happily raises its probability — until it is {(1 + eps).toFixed(2)}×
            what the old policy gave it. Past that the objective is flat, so
            this sample adds nothing more. There is no fence on the left: if an
            update accidentally made a good action <em>rarer</em>, the gradient
            still pulls it back.
          </>
        ) : (
          <>
            For an action that went <strong>worse</strong> than usual, PPO
            lowers its probability — down to {(1 - eps).toFixed(2)}× the old
            value, then stops. On the right there is no fence: if an update made
            a bad action <em>more</em> likely, the full penalty still applies.
            The clip only ever removes the incentive to move further in the
            direction you were already going.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4b. Reusing one batch

export function BatchReuse() {
  const [batchSeed, setBatchSeed] = useState(2);
  const [epochs, setEpochs] = useState(10);
  const lr = 1;
  const theta0 = [0, 0, 0];
  const batch = useMemo(
    () => drawBatch(theta0, seededRng(batchSeed)),
    [batchSeed],
  );
  const plain = useMemo(() => reuseBatch(theta0, batch, 20, lr, null), [batch]);
  const clipped = useMemo(
    () => reuseBatch(theta0, batch, 20, lr, 0.2),
    [batch],
  );
  const sweep = useMemo(
    () => ({ plain: ppoRobustness(10, null), clip: ppoRobustness(10, 0.2) }),
    [],
  );

  const pOld = softmax(theta0);
  const pPlain = softmax(plain[epochs]);
  const pClip = softmax(clipped[epochs]);
  const counts = [0, 1, 2].map(
    (a) => batch.actions.filter((x) => x === a).length,
  );
  const avgAdv = [0, 1, 2].map((a) => {
    const xs = batch.advantages.filter((_, n) => batch.actions[n] === a);
    return xs.length ? meanOf(xs) : NaN;
  });
  const switchedOff = batch.actions.filter((a, n) =>
    clipInactive(pClip[a] / pOld[a], batch.advantages[n], 0.2),
  ).length;
  const looksBest = avgAdv.indexOf(
    Math.max(...avgAdv.map((v) => (Number.isNaN(v) ? -Infinity : v))),
  );

  const bars = (label: string, p: number[], color: string) => (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "92px 1fr",
        gap: 8,
        alignItems: "center",
      }}
    >
      <span style={{ fontSize: 12.5, color }}>{label}</span>
      <div
        style={{
          display: "flex",
          height: 22,
          borderRadius: 4,
          overflow: "hidden",
          border: "1px solid var(--border)",
        }}
      >
        {p.map((q, a) => (
          <div
            key={a}
            title={`${PUSH_ACTIONS[a]}: ${q.toFixed(2)}`}
            style={{
              width: `${q * 100}%`,
              background: ACTION_COLORS[a],
              opacity: 0.85,
              color: "var(--bg)",
              fontSize: 11,
              lineHeight: "22px",
              textAlign: "center",
              overflow: "hidden",
            }}
          >
            {q > 0.12 ? q.toFixed(2) : ""}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <Widget
      title="Reusing one batch: plain policy gradient vs PPO’s clip"
      wide
      caption={
        <>
          The policy starts uniform over Dull, Decent and Great (true average
          rewards 1, 2, 3; noise ±2). It collects a batch of 8 tries, then takes
          several gradient steps (epochs) on that same batch, with step size 1.
          Slide the epochs up. Batch 2 is an unlucky one where Decent happened
          to look best — try others with ◀ ▶.
        </>
      }
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 14,
          alignItems: "end",
          marginBottom: 10,
        }}
      >
        <div className="btn-row">
          <button
            type="button"
            className="btn small"
            onClick={() => setBatchSeed((s) => Math.max(1, s - 1))}
          >
            ◀
          </button>
          <span className="mono">batch {batchSeed}</span>
          <button
            type="button"
            className="btn small"
            onClick={() => setBatchSeed((s) => s + 1)}
          >
            ▶
          </button>
        </div>
        <div style={{ minWidth: 220, flex: "1 1 220px" }}>
          <Slider
            label="epochs on this batch"
            value={epochs}
            min={0}
            max={20}
            step={1}
            onChange={setEpochs}
            format={(v) => String(v)}
            color="accent"
          />
        </div>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(min(270px, 100%), 1fr))",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div style={{ display: "grid", gap: 8 }}>
          <div className="readout">
            {[0, 1, 2].map((a) => (
              <span key={a}>
                <span style={{ color: ACTION_COLORS[a] }}>
                  {PUSH_ACTIONS[a].padEnd(7)}
                </span>
                <span className="k"> tried </span>
                {counts[a]}×<span className="k"> · avg advantage </span>
                {Number.isNaN(avgAdv[a]) ? "—" : signed(avgAdv[a], 2)}
                {a < 2 ? "\n" : ""}
              </span>
            ))}
          </div>
          {bars("old policy", pOld, "var(--text-muted)")}
          {bars("plain", pPlain, "var(--c-b)")}
          {bars("PPO clip 0.2", pClip, "var(--accent)")}
          <div className="readout">
            <span className="k">expected reward </span>old{" "}
            {expectedPPOReward(theta0).toFixed(2)} · plain{" "}
            {expectedPPOReward(plain[epochs]).toFixed(2)} · clip{" "}
            {expectedPPOReward(clipped[epochs]).toFixed(2)}
            {"\n"}
            <span className="k">largest ratio π/π_old </span>plain{" "}
            {Math.max(...pPlain.map((q, a) => q / pOld[a])).toFixed(2)} · clip{" "}
            {Math.max(...pClip.map((q, a) => q / pOld[a])).toFixed(2)}
            {"\n"}
            <span className="k">samples switched off by the clip </span>
            {switchedOff} of 8
          </div>
          <div className="live-note">
            {epochs === 0 ? (
              <>No steps yet. Slide the epochs up.</>
            ) : epochs === 1 ? (
              <>
                After one epoch the two are identical: on a fresh batch the
                ratio is 1 everywhere, so the clip hasn’t engaged and PPO{" "}
                <em>is</em> the plain policy gradient.
              </>
            ) : (
              <>
                After {epochs} epochs, plain reuse has squeezed{" "}
                {(Math.max(...pPlain) * 100).toFixed(0)}% of the probability
                onto{" "}
                <strong>
                  {PUSH_ACTIONS[pPlain.indexOf(Math.max(...pPlain))]}
                </strong>{" "}
                on the strength of 8 noisy tries
                {looksBest !== 2
                  ? " — and in this batch the wrong action looked best"
                  : ""}
                . PPO stopped pulling once each tried action’s ratio left [0.8,
                1.2]: {switchedOff} of the 8 samples are now switched off.
              </>
            )}
          </div>
        </div>
        <div style={{ display: "grid", gap: 6 }}>
          <div style={{ fontSize: 13, fontWeight: 600 }}>
            Over 100 seeds: 20 batches, 10 epochs each
          </div>
          <table className="num-table" style={{ width: "100%" }}>
            <thead>
              <tr>
                <th>step size</th>
                <th>plain: reward / stuck</th>
                <th>clip: reward / stuck</th>
              </tr>
            </thead>
            <tbody>
              {sweep.plain.map((row, i) => (
                <tr key={row.lr}>
                  <td>{row.lr}</td>
                  <td
                    style={{
                      color: row.failures > 10 ? "var(--c-bad)" : undefined,
                    }}
                  >
                    {row.mean.toFixed(2)} / {row.failures}
                  </td>
                  <td
                    style={{
                      color:
                        sweep.clip[i].failures > 10
                          ? "var(--c-bad)"
                          : undefined,
                    }}
                  >
                    {sweep.clip[i].mean.toFixed(2)} / {sweep.clip[i].failures}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 12.5, color: "var(--text-muted)" }}>
            “reward” = expected reward after training (best possible{" "}
            {PPO_MEANS[2]}); “stuck” = seeds that finished below {PPO_FAIL},
            committed to the wrong action. Each method at its own best step size
            is about equally good; the clip’s gift is that it stays good across
            a much wider range of step sizes.
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 5. Lab: stabilise the policy gradient

const LAB_STORAGE_KEY = "nl-pg-lab-v1";
type LabSave = Record<
  LabBriefId,
  { cfg: LabConfig; solved: boolean; best: number | null }
>;

const defaultSave = (): LabSave => ({
  paid: { cfg: { est: "none", alpha: 0.1 }, solved: false, best: null },
  payscales: { cfg: { est: "none", alpha: 0.1 }, solved: false, best: null },
  hallway: { cfg: { est: "none", alpha: 0.1 }, solved: false, best: null },
});

function loadLab(): LabSave {
  try {
    const raw = localStorage.getItem(LAB_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<LabSave>;
      const d = defaultSave();
      for (const k of Object.keys(d) as LabBriefId[]) {
        const v = parsed[k];
        if (
          v &&
          LAB_ESTIMATORS.includes(v.cfg?.est) &&
          (LAB_ALPHAS as readonly number[]).includes(v.cfg?.alpha)
        )
          d[k] = { ...d[k], ...v };
      }
      return d;
    }
  } catch {
    // ignore storage errors
  }
  return defaultSave();
}

const HINTS: Record<LabBriefId, string> = {
  paid: "Every press pays about 50. Without a baseline, whichever button you press first gets a huge push up — so the agent tends to marry its first choice. What would make a 51-coin press feel good and a 49-coin press feel bad?",
  payscales:
    "One running average sits somewhere around +10 (the average of the four rooms). In the +40 room every press looks wonderful; in the −20 room every press looks terrible. The baseline needs to know which room it is in.",
  hallway:
    "Each episode is 8 to 40 noisy steps. A Monte-Carlo return adds up the noise of every remaining step, so the push for an early step is mostly noise. Which estimator judges a step by its own reward plus the critic’s estimate of where it landed?",
};

export function PGLab() {
  const [save, setSave] = useState<LabSave>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [ran, setRan] = useState<LabConfig | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [showOpt, setShowOpt] = useState(false);

  const brief = LAB_BRIEFS[briefIdx];
  const entry = save[brief.id];
  const cfg = entry.cfg;
  const result =
    ran && ran.est === cfg.est && ran.alpha === cfg.alpha
      ? evaluateLab(brief.id, cfg)
      : null;
  const exhaustive = useMemo(
    () => (entry.solved ? solveLabExhaustive(brief.id) : null),
    [brief.id, entry.solved],
  );

  const persist = (next: LabSave) => {
    setSave(next);
    try {
      localStorage.setItem(LAB_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };

  const setCfg = (c: Partial<LabConfig>) => {
    persist({ ...save, [brief.id]: { ...entry, cfg: { ...cfg, ...c } } });
  };

  const run = () => {
    const r = evaluateLab(brief.id, cfg);
    setRan({ ...cfg });
    persist({
      ...save,
      [brief.id]: {
        ...entry,
        solved: entry.solved || r.pass,
        best: r.pass ? Math.max(entry.best ?? -Infinity, r.score) : entry.best,
      },
    });
  };

  const isHall = brief.id === "hallway";
  const yDom: [number, number] = isHall ? [-15, 6] : [0, 1];
  const curve = result
    ? isHall
      ? smooth(result.curve, 10)
      : result.curve
    : null;
  const fmtScore = (s: number) =>
    isHall ? s.toFixed(2) : `${(s * 100).toFixed(1)}%`;

  return (
    <Widget
      title="Lab: Stabilise the policy gradient"
      wide
      caption={
        <>
          Three briefs, one engine: a softmax policy trained by the policy
          gradient. You choose <strong>what to subtract</strong> from the reward
          before pushing (the estimator) and <strong>how hard to push</strong>{" "}
          (the step size α). Critics and baselines all learn at rate β = 0.1.
          Every run is averaged over 20 fixed seeds. Your settings are saved in
          this browser.
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setRan(null);
          setShowHint(false);
          setShowOpt(false);
        }}
        options={LAB_BRIEFS.map((b, i) => ({
          value: i,
          label: `${save[b.id].solved ? "✓ " : ""}${b.title.split(" · ")[0]}`,
        }))}
      />
      <div style={{ margin: "12px 0" }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
        <div
          style={{ fontSize: 13.5, color: "var(--text-muted)", marginTop: 6 }}
        >
          Goal: {brief.metric} ≥{" "}
          <strong>
            {isHall ? brief.threshold.toFixed(1) : `${brief.threshold * 100}%`}
          </strong>{" "}
          (budget: {brief.budget}).
        </div>
      </div>

      <div style={{ display: "grid", gap: 10, marginBottom: 12 }}>
        <Segmented
          value={cfg.est}
          onChange={(est) => setCfg({ est })}
          options={LAB_ESTIMATORS.map((e) => ({
            value: e,
            label: ESTIMATOR_LABELS[e],
          }))}
          label="Subtract from the reward"
        />
        <Segmented
          value={cfg.alpha}
          onChange={(alpha) => setCfg({ alpha })}
          options={LAB_ALPHAS.map((a) => ({ value: a, label: String(a) }))}
          label="Step size"
        />
        <div className="btn-row">
          <button type="button" className="btn primary" onClick={run}>
            Train 20 seeds
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => setShowHint((h) => !h)}
          >
            {showHint ? "Hide hint" : "Hint"}
          </button>
          {entry.solved && (
            <span className="status good">
              Solved
              {entry.best !== null
                ? ` · your best ${fmtScore(entry.best)}`
                : ""}
            </span>
          )}
        </div>
        {showHint && <div className="live-note">{HINTS[brief.id]}</div>}
      </div>

      {result && curve && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(min(280px, 100%), 1fr))",
            gap: 16,
            alignItems: "start",
          }}
        >
          <Plot
            width={440}
            height={230}
            x={[0, curve.length]}
            y={yDom}
            xLabel={isHall ? "episode" : "press"}
            yLabel={isHall ? "return" : "P(best button)"}
            ariaLabel="Average learning curve for the chosen settings"
          >
            {({ x, y }) => (
              <line
                x1={x(0)}
                x2={x(curve.length)}
                y1={y(brief.threshold)}
                y2={y(brief.threshold)}
                stroke="var(--c-good)"
                strokeDasharray="5 4"
              />
            )}
            <Clip>
              <Polyline
                pts={curve.map((v, t) => [t + 1, v] as [number, number])}
                color={result.pass ? "var(--c-good)" : "var(--c-bad)"}
                width={2.2}
              />
            </Clip>
          </Plot>
          <div style={{ display: "grid", gap: 10 }}>
            <div>
              <span className={`status ${result.pass ? "good" : "bad"}`}>
                {result.pass ? "Pass" : "Not yet"} · {fmtScore(result.score)}
              </span>
            </div>
            <div className="readout">
              <span className="k">per seed </span>
              {result.perSeed
                .map((s) => (isHall ? s.toFixed(1) : (s * 100).toFixed(0)))
                .join(" ")}
              {"\n"}
              <span className="k">seeds that individually reach the goal </span>
              {result.perSeed.filter((s) => s >= brief.threshold).length} / 20
            </div>
            <div className="live-note">
              {result.pass
                ? `It works: ${ESTIMATOR_LABELS[cfg.est].toLowerCase()} at α = ${cfg.alpha} reaches ${fmtScore(result.score)}. ${briefIdx < 2 ? "On to the next brief — or look for a better setting." : "That completes the lab."}`
                : result.score < brief.threshold * (isHall ? 0.5 : 0.75)
                  ? `Far off (${fmtScore(result.score)}). Look at the per-seed numbers: are most seeds failing, or do a few lucky seeds succeed while the rest get stuck?`
                  : `Close (${fmtScore(result.score)}), but not there. Is it the estimator or the step size that’s holding it back? Try neighbouring step sizes first.`}
            </div>
          </div>
        </div>
      )}

      {entry.solved && exhaustive && (
        <div style={{ marginTop: 14 }}>
          <button
            type="button"
            className="btn"
            onClick={() => setShowOpt((s) => !s)}
          >
            {showOpt ? "Hide" : "Show"} the exhaustive search
          </button>
          {showOpt && (
            <div style={{ marginTop: 10, display: "grid", gap: 8 }}>
              <div className="live-note">
                All {exhaustive.total} settings were tried:{" "}
                <strong>{exhaustive.passing.length}</strong> pass. The best is{" "}
                {ESTIMATOR_LABELS[exhaustive.best.cfg.est].toLowerCase()} at α ={" "}
                {exhaustive.best.cfg.alpha} ({fmtScore(exhaustive.best.score)}).
                {brief.id !== "hallway"
                  ? " In a one-step problem the per-state baseline and actor-critic are the same algorithm: the TD error r − V(s) is exactly the reward minus the room’s baseline."
                  : ""}
              </div>
              <div style={{ overflowX: "auto" }}>
                <table className="num-table">
                  <thead>
                    <tr>
                      <th />
                      {LAB_ALPHAS.map((a) => (
                        <th key={a}>{a}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {LAB_ESTIMATORS.map((e) => (
                      <tr key={e}>
                        <th style={{ textAlign: "left" }}>
                          {ESTIMATOR_LABELS[e]}
                        </th>
                        {LAB_ALPHAS.map((a) => {
                          const cell = exhaustive.all.find(
                            (x) => x.cfg.est === e && x.cfg.alpha === a,
                          )!;
                          return (
                            <td
                              key={a}
                              style={{
                                color: cell.pass
                                  ? "var(--c-good)"
                                  : "var(--text-faint)",
                                fontWeight: cell.pass ? 700 : 400,
                              }}
                            >
                              {isHall
                                ? cell.score.toFixed(1)
                                : (cell.score * 100).toFixed(0)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </Widget>
  );
}
