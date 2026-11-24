import { useState, type ReactNode } from 'react';
import { RichText } from './RichText';

/**
 * Ask the learner to commit to a prediction before revealing an experiment or explanation.
 * If `answer` is omitted the prediction is open-ended and simply unlocks the content.
 */
export function Predict({
  question,
  options,
  answer,
  children,
  label = 'Predict first',
}: {
  question: string;
  options: string[];
  answer?: number;
  children?: ReactNode;
  label?: string;
}) {
  const [picked, setPicked] = useState<number | null>(null);
  const done = picked !== null;
  return (
    <div className="predict">
      <div className="predict-label">◆ {label}</div>
      <div className="predict-q"><RichText text={question} /></div>
      <div className="options">
        {options.map((o, i) => {
          let cls = '';
          if (done && answer !== undefined) cls = i === answer ? 'correct' : i === picked ? 'wrong' : '';
          else if (done && i === picked) cls = 'picked';
          return (
            <button key={i} type="button" className={`option ${cls}`} disabled={done} onClick={() => setPicked(i)}>
              <span className="marker">{String.fromCharCode(65 + i)}</span>
              <span><RichText text={o} /></span>
            </button>
          );
        })}
      </div>
      {done && (
        <div className="reveal">
          {answer !== undefined && (
            <div className={`feedback ${picked === answer ? 'good' : 'bad'}`} style={{ marginBottom: 12 }}>
              <div className="feedback-title" style={{ marginBottom: 0 }}>
                {picked === answer ? 'Your intuition was right — now see why.' : 'Interesting guess. Let’s find out what actually happens.'}
              </div>
            </div>
          )}
          {children}
        </div>
      )}
    </div>
  );
}
