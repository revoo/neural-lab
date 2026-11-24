import { Fragment, type ReactNode } from 'react';
import { Tex } from './Tex';

/**
 * A tiny inline renderer for quiz text and other data-driven strings:
 * $inline math$, $$block math$$, **bold**, `code`, and blank-line paragraphs.
 */
export function RichText({ text, block = false }: { text: string; block?: boolean }) {
  if (!block) return <>{inline(text)}</>;
  return (
    <>
      {text.split(/\n\s*\n/).map((para, i) => {
        const m = para.trim().match(/^\$\$([\s\S]+)\$\$$/);
        return m ? <Tex key={i} block>{m[1]}</Tex> : <p key={i}>{inline(para)}</p>;
      })}
    </>
  );
}

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\$\$([^$]+)\$\$|\$([^$]+)\$|\*\*([^*]+)\*\*|`([^`]+)`|\*([^*\s][^*]*)\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(<Fragment key={k++}>{text.slice(last, m.index)}</Fragment>);
    if (m[1]) out.push(<Tex key={k++} block>{m[1]}</Tex>);
    else if (m[2]) out.push(<Tex key={k++}>{m[2]}</Tex>);
    else if (m[3]) out.push(<strong key={k++}>{m[3]}</strong>);
    else if (m[4]) out.push(<code key={k++}>{m[4]}</code>);
    else if (m[5]) out.push(<em key={k++}>{m[5]}</em>);
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(<Fragment key={k++}>{text.slice(last)}</Fragment>);
  return out;
}
