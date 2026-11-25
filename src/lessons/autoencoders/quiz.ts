// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'ae.ghost',
    kind: 'choice',
    concepts: ['generative-model'],
    prompt: 'Averaging the pixels of a grinning face and a frowning face gives a ghostly double mouth, not a neutral face. Why?',
    options: [
      {
        text: 'Real faces form a thin, curved sheet inside the space of all pixel patterns. The straight line between two faces in pixel space leaves the sheet.',
        correct: true,
        why: 'Averaging the knobs (smile, eyes) stays on the sheet; averaging pixels cuts across empty space. Generating well means working in the knobs.',
      },
      {
        text: 'Because averaging loses precision: with more bits per pixel the average would be a neutral face.',
        why: 'Precision isn’t the issue. Both mouths are present at half strength because pixel averaging adds the two pictures; no amount of precision turns two curves into one.',
      },
      {
        text: 'Because the two faces weren’t aligned first.',
        why: 'They are perfectly aligned: same head position, same eye positions. The mouths differ in shape, and averaging shapes is not the same as averaging pictures of shapes.',
      },
    ],
  },
  {
    id: 'ae.bottleneck',
    kind: 'choice',
    concepts: ['autoencoder', 'latent-space'],
    prompt: 'An autoencoder is trained only to reproduce its input. How can that teach it anything useful?',
    options: [
      {
        text: 'The input has to pass through a code much smaller than itself, so the network must find the few factors that explain the data in order to rebuild it.',
        correct: true,
        why: 'Our autoencoder rediscovered smile and eye openness from pixels alone. Without the bottleneck, copying would be trivial and it would learn nothing.',
      },
      {
        text: 'It can’t: an autoencoder just learns the identity function.',
        why: 'It would, if the code were as big as the input. The narrow code makes the identity impossible, so it has to compress.',
      },
      {
        text: 'It memorises every training example in the code.',
        why: 'Two numbers can’t store 400 pictures verbatim. It stores a description from which the decoder can redraw each one, which is why it works on faces it has never seen.',
      },
    ],
  },
  {
    id: 'ae.holes',
    kind: 'choice',
    concepts: ['autoencoder', 'vae'],
    prompt: 'A plain autoencoder reconstructs faces beautifully, yet random codes fed to its decoder often give junk. What is missing?',
    options: [
      {
        text: 'Nothing in its training says where codes should live or what codes between training examples should mean, so a random code often lands somewhere the decoder never learned.',
        correct: true,
        why: 'Ours put its codes in a lopsided patch from about −2 to 15; only 31% of the square from −3 to 3 decoded to faces. The VAE’s two rules fix exactly these two gaps.',
      },
      {
        text: 'Its decoder is too small; a bigger decoder would turn any code into a face.',
        why: 'A bigger decoder can draw more detail, but it is still only trained on the codes it was shown. Size doesn’t tell it what to do elsewhere.',
      },
      {
        text: 'It needs more training steps.',
        why: 'More training sharpens reconstructions of training faces; it doesn’t add a rule about where codes go.',
      },
    ],
  },
  {
    id: 'ae.kl',
    kind: 'numeric',
    concepts: ['kl-divergence', 'vae'],
    tolerance: 0.01,
    make: (r) => {
      const mu = [0, 0.5, 1, 1.5, 2][r.int(5)];
      const sigma = [0.5, 1, 2][r.int(3)];
      const kl = 0.5 * (mu * mu + sigma * sigma - 1 - Math.log(sigma * sigma));
      return {
        prompt: `An encoder gives one code number a smudge with centre μ = ${mu} and spread σ = ${sigma}. Using ½(μ² + σ² − 1 − ln σ²), what is its KL pull? (Two decimal places.)`,
        answer: Number(kl.toFixed(2)),
        explain: `½(${mu}² + ${sigma}² − 1 − ln ${sigma}²) = ½(${(mu * mu).toFixed(2)} + ${(sigma * sigma).toFixed(2)} − 1 − ${Math.log(sigma * sigma).toFixed(3)}) = ${kl.toFixed(3)}. ${sigma === 1 ? 'With σ = 1 only the centre is charged.' : 'The σ part is zero only at σ = 1; here it adds ' + (0.5 * (sigma * sigma - 1 - Math.log(sigma * sigma))).toFixed(3) + '.'}`,
      };
    },
  },
  {
    id: 'ae.reparam',
    kind: 'choice',
    concepts: ['reparameterization-trick'],
    prompt: 'Why does a VAE write its sampled code as z = μ + σ·ε instead of “draw z from a bell with centre μ and spread σ”?',
    options: [
      {
        text: 'They give the same distribution of codes, but in the first form the randomness is in ε, which depends on nothing, so the gradient of the loss can flow back into μ and σ.',
        correct: true,
        why: 'Nudge μ and z moves by the same amount; nudge σ and z moves by ε times as much. Ordinary backpropagation handles it.',
      },
      {
        text: 'It makes the codes less random.',
        why: 'The codes are exactly as random either way. The rewrite changes where the randomness sits, not how much there is.',
      },
      {
        text: 'It is faster to compute.',
        why: 'Speed isn’t the point. A random draw has no derivative with respect to its parameters, so without the rewrite the encoder couldn’t be trained by backpropagation.',
      },
    ],
  },
  {
    id: 'ae.collapse',
    kind: 'choice',
    concepts: ['posterior-collapse', 'vae'],
    prompt: 'You train a VAE with β = 10. Every printed face is a real face, and they are all the same face. What happened?',
    options: [
      {
        text: 'The KL pull won: every face is encoded as the standard bell itself, the code carries no information, and the decoder can only draw the average face.',
        correct: true,
        why: 'That is posterior collapse. The KL term is zero exactly when the smudge is the bell, so a large β makes “say nothing” the cheapest code.',
      },
      {
        text: 'The decoder overfitted to one training face.',
        why: 'The decoder is doing the best it can with codes that tell it nothing. The face it draws is the least-wrong average, not any one training face.',
      },
      {
        text: 'The printer’s random numbers were too small.',
        why: 'The printer draws from the same standard bell as before. With collapsed codes, every input to the decoder means the same thing, whatever its size.',
      },
    ],
  },
  {
    id: 'ae.compressor',
    kind: 'choice',
    concepts: ['vae', 'autoencoder'],
    prompt: 'You only need to store faces compactly and rebuild them; you will never generate new ones. Which is the better choice?',
    options: [
      {
        text: 'A plain autoencoder, or a VAE with a small β: the pull towards the bell blurs reconstructions, and you don’t need its promise about where codes live.',
        correct: true,
        why: 'At β = 1 our VAE rebuilt held-out faces about twice as badly as the plain autoencoder. The lab’s compressor brief is won with β = 0.1.',
      },
      {
        text: 'A VAE with β = 1, because it is the more advanced model.',
        why: '“More advanced” at generating, not at compressing. The KL pull deliberately makes codes less precise.',
      },
      {
        text: 'A VAE with a large β, to make the codes as compact as possible.',
        why: 'A large β makes codes carry less information, so reconstructions blur and eventually collapse to the average face.',
      },
    ],
  },
  {
    id: 'ae.order',
    kind: 'order',
    prompt: 'Put one VAE training step in order.',
    items: [
      'The encoder reads a face and outputs a centre μ and a spread σ for each code number',
      'Draw fresh noise ε and form the code z = μ + σ·ε',
      'The decoder draws a face from z',
      'Add the reconstruction error to β times the KL pull',
      'Backpropagate through decoder, the sampling formula and encoder, and update both networks',
    ],
    explain: 'Encode to a smudge, sample from it with the reparameterisation trick, decode, score both rules, then one backward pass trains encoder and decoder together.',
  },
];
