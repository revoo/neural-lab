// authored-by: Codex Meridian 6
import { useState } from 'react';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import {
  average,
  balanced,
  bounds,
  depths,
  distributions,
  extra,
  huffman,
  leaves,
  minima,
  split,
  wrongCost,
  type Tree,
} from './compressorLab';

const animals = ['Cat', 'Dog', 'Otter', 'Rabbit', 'Fox', 'Owl', 'Seal', 'Mouse'];
const storage = 'neural-lab:compressor-settings';
const briefs = [
  { id: 'four', tab: 'Four items', text: 'Build a strategy as short on average as the entropy.' },
  { id: 'eight', tab: 'Eight items', text: 'Can you reach the entropy again with more animals?' },
  { id: 'whole', tab: 'Whole questions', text: 'Reach the shortest possible average for these frequencies.' },
  {
    id: 'belief',
    tab: 'Wrong beliefs',
    text: 'Each animal gets an equally long code. Predict how many extra questions you’ll spend on average when the animals have unequal frequencies.',
  },
];
function Group({
  tree,
  path = '',
  branch = 'Start',
  edit,
}: {
  tree: Tree;
  path?: string;
  branch?: string;
  edit?: (path: string, items: number[]) => void;
}) {
  return (
    <li style={{ marginBlock: 6, overflowWrap: 'anywhere' }}>
      <strong>{branch} </strong>
      {Array.isArray(tree) ? (
        <>
          {tree.map((i) => animals[i]).join(', ')}{' '}
          {tree.length > 1 && edit && (
            <button className="btn small" type="button" onClick={() => edit(path, tree)}>
              Split group
            </button>
          )}
        </>
      ) : (
        <ul style={{ paddingInlineStart: 14 }}>
          <Group tree={tree.yes} path={path + 'y'} branch="Yes" edit={edit} />
          <Group tree={tree.no} path={path + 'n'} branch="No" edit={edit} />
        </ul>
      )}
    </li>
  );
}
export function CompressorLab() {
  const [active, select] = useState(0);
  const [saved, save] = useState<{ trees: Record<number, Tree>; guess: string }>(() => {
    try {
      return JSON.parse(localStorage.getItem(storage) ?? 'null') ?? { trees: {}, guess: '' };
    } catch {
      return { trees: {}, guess: '' };
    }
  });
  const [editing, edit] = useState<{ path: string; items: number[] } | null>(null);
  const [yes, setYes] = useState<number[]>([]);
  const progress = useLabProgress('information-lab'),
    log = useMachineLog();
  const belief = active === 3;
  const index = belief ? 0 : active,
    p = distributions[index];
  const tree = belief ? balanced : (saved.trees[active] ?? leaves(p));
  const cost = average(tree, p);
  const passing = belief ? saved.guess === 'extra' : cost !== null && Math.abs(cost - minima[index]) < 1e-10;
  const update = (next: typeof saved) => {
    save(next);
    try {
      localStorage.setItem(storage, JSON.stringify(next));
    } catch {
      /* Storage is optional. */
    }
  };
  const updateTree = (value: Tree) => update({ ...saved, trees: { ...saved.trees, [active]: value } });
  const formatted = (value: number) => String(Number(value.toFixed(6)));
  const status = belief
    ? saved.guess
      ? 'Measured ' +
        wrongCost +
        ' questions on average. Extra cost ' +
        extra +
        ' questions, equal to the KL divergence in bits.'
      : 'Make a prediction to measure the cost.'
    : cost === null
      ? 'Split every group down to a single animal.'
      : formatted(cost) +
        ' questions on average. ' +
        (passing ? 'This is the shortest possible average.' : 'Try a different grouping.');
  return (
    <LabShell
      title="Lab: Twenty Questions Compressor"
      caption="Split a group by choosing which animals go down the yes branch. The rest go down the no branch."
      briefs={briefs}
      active={active}
      onSelect={(i) => {
        edit(null);
        select(i);
      }}
      progress={progress}
      passing={passing}
      status={status}
      hint={
        belief
          ? 'Compare the unchanged code lengths with the observed frequencies.'
          : 'Look at the depths of the most common animals. How often will you pay for an extra question there?'
      }
      reveal={{
        label: 'the shortest strategy',
        content: (
          <>
            <p>
              Entropy {formatted(bounds[index])} bits · shortest average {formatted(minima[index])} questions. Merge the
              least common groups first to build a Huffman tree.
            </p>
            <ul style={{ paddingInlineStart: 14 }}>
              <Group tree={huffman(p)} />
            </ul>
          </>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear }}
      actions={
        !belief && (
          <button
            className="btn"
            type="button"
            onClick={() => {
              updateTree(leaves(p));
              edit(null);
              log.push('Restored the full group');
            }}
          >
            Start over
          </button>
        )
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(min(245px,100%),1fr))', gap: 12 }}>
        <div>
          <ul style={{ paddingInlineStart: 14 }}>
            <Group
              tree={tree}
              edit={
                belief
                  ? undefined
                  : (path, items) => {
                      edit({ path, items });
                      setYes([]);
                    }
              }
            />
          </ul>
          {editing && (
            <fieldset style={{ minWidth: 0, border: '1px solid var(--border)' }}>
              <legend>Which animals go to yes?</legend>
              <div className="btn-row">
                {editing.items.map((i) => (
                  <label key={i}>
                    <input
                      type="checkbox"
                      checked={yes.includes(i)}
                      onChange={(e) => setYes(e.target.checked ? [...yes, i] : yes.filter((j) => j !== i))}
                    />{' '}
                    {animals[i]}
                  </label>
                ))}
              </div>
              <div className="btn-row">
                <button
                  type="button"
                  className="btn"
                  disabled={!yes.length || yes.length === editing.items.length}
                  onClick={() => {
                    const next = split(tree, editing.path, yes);
                    updateTree(next);
                    edit(null);
                    log.push(
                      'Split ' +
                        editing.items.map((i) => animals[i]).join(', ') +
                        ' · ' +
                        (average(next, p)?.toFixed(4) ?? 'groups remain'),
                    );
                  }}
                >
                  Apply split
                </button>
                <button type="button" className="btn" onClick={() => edit(null)}>
                  Cancel
                </button>
              </div>
            </fieldset>
          )}
        </div>
        <table className="num-table">
          <thead>
            <tr>
              <th>Animal</th>
              <th>Chance</th>
              <th>Questions</th>
            </tr>
          </thead>
          <tbody>
            {depths(tree)
              .sort((a, b) => a.item - b.item)
              .map(({ item, depth }) => (
                <tr key={item}>
                  <td>{animals[item]}</td>
                  <td>{(100 * p[item]).toFixed(2)}%</td>
                  <td>{depth ?? 'Pending'}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      {belief && (
        <div className="btn-row">
          {[
            ['same', 'No extra questions'],
            ['extra', '¼ extra question'],
            ['half', '½ extra question'],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              className="btn"
              aria-pressed={saved.guess === id}
              onClick={() => {
                update({ ...saved, guess: id });
                log.push('Measured average ' + wrongCost + ' questions');
              }}
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </LabShell>
  );
}
