// authored-by: Claude Opus 5.5
import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { AND, CLOUD, OR, XOR, update, type Pt, type W } from './perceptron';
import { PerceptronPlot, fmtN, fmtW } from './PerceptronPlot';

type Name = 'or' | 'and' | 'cloud' | 'xor';
const SETS: Record<Name, { pts: Pt[]; label: string }> = {
  or: { pts: OR, label: 'OR' },
  and: { pts: AND, label: 'AND' },
  cloud: { pts: CLOUD, label: 'A separable cloud' },
  xor: { pts: XOR, label: 'XOR' },
};

type State = { w: W; k: number; looks: number; mistakes: number; clean: number; seen: Record<string, number>; cycleAt: number | null };
const fresh = (): State => ({ w: [0, 0, 0], k: 0, looks: 0, mistakes: 0, clean: 0, seen: {}, cycleAt: null });

export function PerceptronStepper() {
  const [name, setName] = useState<Name>('or');
  const [s, setS] = useState<State>(fresh);
  const log = useMachineLog();
  const pts = SETS[name].pts;
  const n = pts.length;
  const converged = s.clean >= n;
  const nextI = s.k % n;

  const stepOnce = (st: State): { st: State; line: string; tone: 'good' | 'bad' | 'info'; repeatOf: number | null } => {
    const i = st.k % n;
    const p = pts[i];
    const r = update(st.w, p, 1);
    const key = `${i}|${st.w.join(',')}`;
    const firstSeen = st.seen[key];
    const looks = st.looks + 1;
    const newCycle = st.cycleAt === null && firstSeen !== undefined && r.mistake;
    const cycleAt = newCycle ? looks : st.cycleAt;
    const next: State = {
      w: r.w,
      k: st.k + 1,
      looks,
      mistakes: st.mistakes + (r.mistake ? 1 : 0),
      clean: r.mistake ? 0 : st.clean + 1,
      seen: { ...st.seen, [key]: firstSeen ?? looks },
      cycleAt,
    };
    const ex = `example ${p.name}`;
    const repeatOf = newCycle ? firstSeen : null;
    if (!r.mistake) return { st: next, line: `look ${looks}: ${ex} should say ${p.y}, said ${r.said}. Right, so no change.`, tone: 'info', repeatOf };
    const sign = p.y === 1 ? '+' : '−';
    return {
      st: next,
      line: `look ${looks}: ${ex} should say ${p.y}, said ${r.said}. Mistake ${next.mistakes}: w ← w ${sign} (${fmtN(p.x[0])}, ${fmtN(p.x[1])}, 1) = ${fmtW(r.w)}`,
      tone: 'bad',
      repeatOf,
    };
  };

  const advance = (count: number) => {
    let st = s;
    for (let c = 0; c < count; c++) {
      if (st.clean >= n) break;
      const r = stepOnce(st);
      st = r.st;
      log.push(r.line, r.tone);
      if (r.repeatOf !== null) log.push(`Look ${st.looks} started from exactly the weights of look ${r.repeatOf}, at the same example. From here every step repeats, for ever.`, 'bad');
      if (st.clean >= n) log.push(`A full pass of ${n} examples with no mistakes: learned, after ${st.mistakes} mistakes in ${st.looks} looks. Final w = ${fmtW(st.w)}.`, 'good');
    }
    setS(st);
  };

  const pick = (v: Name) => {
    setName(v);
    setS(fresh());
    log.clear();
  };

  const p = pts[nextI];
  const note = converged
    ? `Learned. A whole pass through all ${n} examples without a single mistake, so the rule will never change the weights again. It took ${s.mistakes} mistake${s.mistakes === 1 ? '' : 's'}.`
    : s.cycleAt !== null
      ? `Stuck in a loop. After ${s.mistakes} mistakes the weights have come back to a setting they had before, at the same example, so the same sequence of fixes will repeat for ever. Every fix to one corner breaks another.`
      : s.looks === 0
        ? `All weights start at 0. Press Next example: the machine looks at ${p.name}, says 1 if w₁x₁ + w₂x₂ + b ≥ 0, and is told the right answer.`
        : `Next up: ${p.name}, which should say ${p.y}. It is currently on the ${(s.w[0] * p.x[0] + s.w[1] * p.x[1] + s.w[2] >= 0) === (p.y === 1) ? 'right' : 'wrong'} side of the line. Mistakes so far: ${s.mistakes}.`;

  return (
    <Widget
      title="The perceptron, one example at a time"
      caption={
        <>
          Step through <strong>OR</strong> until nothing changes. Then try <strong>XOR</strong> and keep stepping. The shaded region is where the unit fires (says 1); a ringed point is on the wrong side; the large ring marks the example the machine looks at next. The log shows every update.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="data" options={(Object.keys(SETS) as Name[]).map((k) => ({ value: k, label: SETS[k].label }))} value={name} onChange={pick} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <PerceptronPlot pts={pts} w={s.w} focus={converged ? undefined : nextI} label={`${SETS[name].label}: the points and the perceptron's current line`} />
        <div>
          <div className="readout">
            <span className="k">w₁, w₂, b </span>
            {fmtW(s.w)}
            {'\n'}
            <span className="k">looks     </span>
            {s.looks}
            {'\n'}
            <span className="k">mistakes  </span>
            {s.mistakes}
          </div>
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button className="btn primary" onClick={() => advance(1)} disabled={converged}>
              Next example
            </button>
            <button className="btn" onClick={() => advance(n)} disabled={converged}>
              Next {n}
            </button>
            <button
              className="btn"
              onClick={() => {
                setS(fresh());
                log.clear();
              }}
            >
              ↺ Reset
            </button>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
      <div style={{ marginTop: 10 }}>
        <MachineLog lines={log.lines} onClear={log.clear} empty="Each look at an example is logged here, with the update it caused." />
      </div>
    </Widget>
  );
}
