import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Label, Plot } from '../../components/plot';

type Email = { text: string; links: number; bangs: number; spam: boolean };
export const EMAILS: Email[] = [
  { text: 'Lunch tomorrow? The new ramen place finally opened!', links: 0, bangs: 1, spam: false },
  { text: 'Here are the slides from today’s design review.', links: 1, bangs: 0, spam: false },
  { text: 'Could you review PR 482 before standup? CI link inside. Thanks!', links: 2, bangs: 1, spam: false },
  { text: 'Happy birthday!! Hope it’s a great one!', links: 0, bangs: 3, spam: false },
  { text: 'Flight confirmation AMS → SFO. Manage booking · check in · seat map', links: 3, bangs: 0, spam: false },
  { text: 'YOU HAVE WON!!! Claim your prize now!!! Click!', links: 3, bangs: 6, spam: true },
  { text: 'Limited offer! Click here! Click here! Click here! Unsubscribe', links: 5, bangs: 3, spam: true },
  { text: 'Your account is suspended! Verify now: link, link, link, link', links: 4, bangs: 2, spam: true },
  { text: 'Earn money from home!!!! Start today!! Info here', links: 2, bangs: 6, spam: true },
  { text: 'Hot deals: 90% off designer watches!! Shop · Shop · Shop · More · Deals · Now', links: 6, bangs: 2, spam: true },
];

export function EmailSpace() {
  const [sel, setSel] = useState(5);
  const e = EMAILS[sel];
  return (
    <Widget
      title="Ten emails, ten points in space"
      caption={
        <>
          Each email has been turned into a 2-number vector: <strong>[number of links, number of exclamation marks]</strong>. Click any point to see
          the email behind it. Notice that the spam (<span className="c-b">orange</span>) gathers in one region and legitimate mail (
          <span className="c-a">blue</span>) in another. Nobody told the plot which emails were spam — the grouping falls out of the numbers.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 320px' }}>
          <Plot x={[-0.5, 7]} y={[-0.5, 7]} width={360} height={340} xLabel="links →" yLabel="↑ exclamation marks" ariaLabel="Emails plotted by links and exclamation marks">
            {({ x, y }) =>
              EMAILS.map((m, i) => (
                <g key={i} onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>
                  <circle
                    cx={x(m.links)}
                    cy={y(m.bangs)}
                    r={i === sel ? 11 : 8}
                    fill={m.spam ? 'var(--c-b)' : 'var(--c-a)'}
                    opacity={i === sel ? 1 : 0.8}
                    stroke={i === sel ? 'var(--text)' : 'none'}
                    strokeWidth={2}
                  />
                </g>
              ))
            }
            <Label at={[e.links, e.bangs]} dx={14} dy={-10} size={12}>{`[${e.links}, ${e.bangs}]`}</Label>
          </Plot>
        </div>
        <div className="widget-col">
          <div className="field-label">Selected email</div>
          <div className="readout" style={{ fontFamily: 'var(--font-ui)', fontSize: 14 }}>
            “{e.text}”
          </div>
          <div className="readout" style={{ marginTop: 10 }}>
            <span className="k">email_vector =</span> [{e.links}, {e.bangs}]
            <br />
            <span className="k">label =</span> <span className={e.spam ? 'c-b' : 'c-a'}>{e.spam ? '"spam"' : '"not spam"'}</span>
          </div>
          <p className="muted" style={{ fontSize: 13.5, marginTop: 10 }}>
            Two numbers is a crude description of an email. Real spam filters use thousands. But the principle is identical, and two dimensions
            have the advantage that we can actually look at them.
          </p>
        </div>
      </div>
    </Widget>
  );
}
