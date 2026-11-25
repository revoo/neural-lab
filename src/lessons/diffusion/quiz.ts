// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'diff.one-shot',
    kind: 'choice',
    concepts: ['denoiser', 'diffusion-model'],
    prompt: 'A perfect denoiser is handed pure noise and asked for its best guess of the clean data point. What does it return, and why is that a problem?',
    options: [
      {
        text: 'The average of all the data: with no information, every clean point is equally plausible. An average of many different examples is a blur, not an example.',
        correct: true,
        why: 'For the ring, all 2,000 one-step samples landed on the same point in the empty middle. Diffusion avoids this by moving only a little way towards each guess and guessing again.',
      },
      {
        text: 'A random training example, because it has to pick one.',
        why: 'Squared error rewards the average, not a random pick: guessing one example is very wrong whenever the truth was another, and the average is the least wrong on average.',
      },
      {
        text: 'Nothing useful, but that’s fine: diffusion never uses the denoiser at high noise.',
        why: 'Every sample starts at pure noise, so the very first call is at the highest noise level. What matters is that the sampler only takes a small step towards that guess.',
      },
    ],
  },
  {
    id: 'diff.forward-jump',
    kind: 'numeric',
    concepts: ['diffusion-model'],
    tolerance: 0.005,
    make: (r) => {
      const abar = [0.25, 0.36, 0.64, 0.81][r.int(4)];
      const x0 = [1, 2, -1.5][r.int(3)];
      const eps = [0.5, -1, 1.5][r.int(3)];
      const xt = Math.sqrt(abar) * x0 + Math.sqrt(1 - abar) * eps;
      return {
        prompt: `A clean value x₀ = ${x0} is noised to a step with signal share ᾱ = ${abar}, using noise ε = ${eps}. What is the noisy value? (Three decimal places.)`,
        answer: Number(xt.toFixed(3)),
        explain: `x_t = √ᾱ·x₀ + √(1 − ᾱ)·ε = ${Math.sqrt(abar).toFixed(2)} × ${x0} + ${Math.sqrt(1 - abar).toFixed(2)} × ${eps} = ${xt.toFixed(3)}. One line, however many steps it represents.`,
      };
    },
  },
  {
    id: 'diff.shrink',
    kind: 'choice',
    concepts: ['diffusion-model'],
    prompt: 'Why does the forward process scale the clean point down by √ᾱ as well as adding noise?',
    options: [
      {
        text: 'So that the total spread stays about the same: variances add, and √ᾱ and √(1 − ᾱ) are chosen so the shrunk data plus the noise keep a variance of about 1, ending exactly at the standard bell.',
        correct: true,
        why: 'Then every dataset ends at the same, known distribution, which is where sampling starts.',
      },
      {
        text: 'Because shrinking is what removes the information; the noise just hides the shrinking.',
        why: 'Shrinking alone is reversible (divide by √ᾱ). It’s the noise that destroys information.',
      },
      {
        text: 'To make the numbers smaller so the network trains faster.',
        why: 'The scaling is about where the process ends: a fixed standard bell, independent of the data.',
      },
    ],
  },
  {
    id: 'diff.ddim-ddpm',
    kind: 'choice',
    concepts: ['ddim'],
    prompt: 'DDIM and DDPM use the same trained denoiser. What is the difference between them?',
    options: [
      {
        text: 'At each step DDIM rebuilds the point with the noise its guess implies; DDPM replaces part of that noise with fresh random noise. So DDIM is deterministic given the starting noise, and DDPM is not.',
        correct: true,
        why: 'In our runs DDIM was a little better with 10 or 20 steps; with 50, DDPM won on two of the three shapes. Fresh noise can wash out earlier mistakes but needs steps to clean up after itself.',
      },
      {
        text: 'DDIM needs a different network, trained to take bigger steps.',
        why: 'No retraining: DDIM is purely a different sampling rule for the same denoiser, which is why it caught on so fast.',
      },
      {
        text: 'DDPM adds noise and DDIM removes it.',
        why: 'Both run the reverse process, from noise to data. They differ only in how much fresh randomness each step injects.',
      },
    ],
  },
  {
    id: 'diff.loss-floor',
    kind: 'choice',
    concepts: ['denoiser'],
    prompt: 'A diffusion model’s training loss (squared error in predicting the added noise) stops falling well above zero. Most likely?',
    options: [
      {
        text: 'That may be close to the best possible: from a noisy point, even a perfect denoiser can only predict the noise on average, so the loss has a floor above zero.',
        correct: true,
        why: 'For our ring the best possible loss is about 0.40. Compare against that floor, not against zero.',
      },
      {
        text: 'A bug: a correct model should predict the noise exactly.',
        why: 'The same noisy point can come from many clean points with different noise. No model can know which one happened.',
      },
      {
        text: 'The learning rate is too high.',
        why: 'Possible in general, but a loss above zero is expected even for a perfect denoiser.',
      },
    ],
  },
  {
    id: 'diff.steps',
    kind: 'choice',
    concepts: ['noise-schedule'],
    prompt: 'With a perfect denoiser, 20 evenly spaced steps left 7% of the ring out of place, but some 4-step plans left under 5%. How?',
    options: [
      {
        text: 'Evenly spaced steps waste stops at noise levels where the shape has already gone. Good plans stop once where samples choose a blob and spend the rest near clean data.',
        correct: true,
        why: 'By step 500 of DDPM’s schedule less than 0.1% of the shape remains, so half of an even plan’s stops do nothing useful.',
      },
      {
        text: 'Fewer steps add less numerical error, so fewer is always better.',
        why: 'One step gives pure blur and three can’t meet the brief at all. Few steps work only if they are in the right places.',
      },
      {
        text: 'The 4-step plans used a better denoiser.',
        why: 'The denoiser is the same exact one in both. Only the step plan differs.',
      },
    ],
  },
  {
    id: 'diff.order',
    kind: 'order',
    prompt: 'Put one DDIM step in order.',
    items: [
      'Feed the noisy point and its noise level to the denoiser',
      'Get the best guess of the clean point',
      'Work out the noise that guess implies',
      'Rebuild the point at the next, lower noise level from the guess and the implied noise',
    ],
    explain: 'Guess, infer the noise, re-noise a little less. Repeat until the noise level reaches zero, where the point is simply the final guess.',
  },
];
