import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Toggle } from '../../components/controls';
import { FnPath, Plot } from '../../components/plot';
import { ACTIVATIONS, type ActivationName } from '../../lib/activations';

const NOTES: Partial<Record<ActivationName, { personality: string; used: string }>> = {
  sigmoid: {
    personality: 'A smooth dimmer switch from 0 to 1. Lovely for “how likely?”, but it goes almost perfectly flat at both ends — and a flat function gives learning almost nothing to work with.',
    used: 'Output layers that need a probability, and the “gates” inside LSTMs. Rarely used between hidden layers today.',
  },
  tanh: {
    personality: 'Sigmoid’s centred cousin, running from −1 to 1. Same flat ends, same problem in deep stacks.',
    used: 'Classic recurrent networks (RNNs, LSTMs), and some small networks.',
  },
  relu: {
    personality: 'The bouncer: negatives don’t get in, positives pass untouched. Its slope is exactly 1 for every positive input, so it never goes flat on that side.',
    used: 'The default for convolutional networks and deep networks generally since around 2012.',
  },
  leakyRelu: {
    personality: 'ReLU, but negatives are let through at a trickle (×0.1) instead of being blocked — so a neuron can never go completely silent and stop learning.',
    used: 'A common fix when too many ReLU neurons “die” during training.',
  },
  gelu: {
    personality: 'A smooth ReLU. Small negative inputs leak through slightly and the corner at zero is rounded off.',
    used: 'BERT, GPT-2, GPT-3 and many other transformers.',
  },
  silu: {
    personality: 'Another smooth ReLU (input × sigmoid of input), with a small dip below zero.',
    used: 'Inside the SwiGLU blocks of Llama, Mistral and most recent open language models.',
  },
  softplus: {
    personality: 'A perfectly smooth version of ReLU, with no corner at all.',
    used: 'Places where an output must be positive, such as predicted variances.',
  },
};

const CHOICES: ActivationName[] = ['relu', 'leakyRelu', 'gelu', 'silu', 'sigmoid', 'tanh', 'softplus'];

export function ActivationGallery() {
  const [sel, setSel] = useState<ActivationName>('relu');
  const [slope, setSlope] = useState(false);
  const A = ACTIVATIONS[sel];
  const note = NOTES[sel]!;
  return (
    <Widget
      title="A gallery of activation functions"
      wide
      caption={
        <>
          The thick line is the function: what the neuron does to its tally. Switch on <strong>slope</strong> to see how steep it is at each point — which, as you saw in
          the chain-rule lesson, is how much learning signal can pass back through it. Flat regions are where learning stalls.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented value={sel} onChange={setSel} options={CHOICES.map((k) => ({ value: k, label: ACTIVATIONS[k].label }))} />
        <Toggle label="Show slope" checked={slope} onChange={setSlope} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <Plot x={[-4, 4]} y={[-1.5, 3]} width={420} height={260} ariaLabel={`${A.label} activation`}>
            {CHOICES.filter((k) => k !== sel).map((k) => (
              <FnPath key={k} f={ACTIVATIONS[k].f} color="var(--text-faint)" width={1} />
            ))}
            {slope && <FnPath f={A.df} color="var(--c-grad)" width={2} dashed samples={500} />}
            <FnPath f={A.f} color="var(--accent)" width={3.5} samples={500} />
          </Plot>
        </div>
        <div className="widget-col">
          <h4 style={{ margin: '0 0 6px', fontSize: 17 }}>{A.label}</h4>
          <p style={{ margin: '0 0 10px', fontSize: 14.5, lineHeight: 1.55 }}>{note.personality}</p>
          <div className="field-label">Where you’ll meet it</div>
          <p style={{ margin: 0, fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.55 }}>{note.used}</p>
        </div>
      </div>
    </Widget>
  );
}
