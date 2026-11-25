// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'gans.vanishing',
    kind: 'choice',
    concepts: ['gan', 'minimax-game'],
    prompt: 'With the original minimax generator loss log(1 − D(G(z))), what happens to the generator’s gradient if the discriminator becomes almost perfect (D(G(z)) ≈ 0 on every fake)?',
    options: [
      {
        text: 'The gradient vanishes: d/dz log(1 − σ(z)) = −σ(z) ≈ 0, leaving the generator with almost no learning signal.',
        correct: true,
        why: 'When D(fake) ≈ 0, log(1 − D) is nearly flat, so its derivative shrinks toward zero. Switching to Goodfellow’s non-saturating loss −log D(G(z)) gives derivative σ(z) − 1 ≈ −1, keeping a strong gradient even when fakes are terrible.',
      },
      {
        text: 'The gradient becomes huge and forces the generator to converge in one step.',
        why: 'That is the behaviour of the *non-saturating* loss −log D(G(z)), not the original log(1 − D(G(z))) loss.',
      },
      {
        text: 'The generator ignores D and trains directly on squared error against real data.',
        why: 'A GAN generator never sees real training points directly; every nudge comes through D’s derivative.',
      },
    ],
  },
  {
    id: 'gans.mode-collapse',
    kind: 'choice',
    concepts: ['gan', 'mode-collapse'],
    prompt: 'What is mode collapse in a GAN trained on data with three distinct peaks?',
    options: [
      {
        text: 'The generator produces sharp, realistic samples from only one peak (ignoring the other two), because the discriminator scores each sample individually rather than checking diversity.',
        correct: true,
        why: 'Fooling D on a single real-looking region earns high scores for every individual fake, so an unbalanced generator can dump 100% of its output onto one peak and drop the rest of the distribution.',
      },
      {
        text: 'The generator outputs the blurry average right in the empty gap between all three peaks.',
        why: 'Blurry averaging is the failure mode of squared-error / VAE models; GANs fail in the opposite way by making crisp samples from too few modes.',
      },
      {
        text: 'The discriminator stops outputting probabilities between 0 and 1.',
        why: 'The discriminator’s sigmoid output stays in (0, 1); mode collapse is a loss of variety in the generator’s output distribution.',
      },
    ],
  },
  {
    id: 'gans.equilibrium',
    kind: 'choice',
    concepts: ['gan', 'minimax-game'],
    prompt: 'At the theoretical Nash equilibrium of the GAN minimax game, when the generator perfectly matches the true data distribution, what does the optimal discriminator D(x) output?',
    options: [
      {
        text: '0.5 everywhere: real and fake samples are indistinguishable, so the best detective can only flip a fair coin.',
        correct: true,
        why: 'As Goodfellow et al. (2014) proved, the unique global equilibrium has p_G = p_data and D(x) = 1/2 everywhere.',
      },
      {
        text: '1.0 on real data and 0.0 on fake data.',
        why: 'If D can tell real from fake with 100% accuracy, the generator has not matched the real distribution yet.',
      },
      {
        text: '0.0 everywhere so the minimax value is zero.',
        why: 'Outputting 0 everywhere would misclassify every real example and give infinite discriminator loss.',
      },
    ],
  },
  {
    id: 'gans.balance',
    kind: 'choice',
    concepts: ['gan', 'mode-collapse'],
    prompt: 'In our 30-setting, 5-seed lab search on the 3-mode mixture, what happened at both extremes of the duel (generator much faster than D, and D running 4 steps per tiny G step)?',
    options: [
      {
        text: 'Both extremes broke down: an overly fast generator collapsed or diverged, and an overly dominant discriminator also trapped G in partial mode collapse.',
        correct: true,
        why: 'Keeping the duel balanced — D slightly faster than G without crushing it (e.g. lrRatio = 0.5, 1 D-step, 2D noise) — reliably covered all 3 modes across all 5 seeds.',
      },
      {
        text: 'Cranking D-steps as high as possible always beat 1 D-step per G-step.',
        why: 'On our 5-seed grid, 4 D-steps with lrRatio = 0.25 dropped to 1.4–1.6 covered modes (0/5 seeds covering all 3), while 1 D-step with lrRatio = 0.5 hit 3.0/3 across all 5 seeds in 40 updates.',
      },
      {
        text: 'Setting the generator 4× faster than the discriminator won every seed.',
        why: 'At lrRatio = 4, average mode coverage collapsed to 0.4–2.0 out of 3 with Wasserstein distance above 1.0.',
      },
    ],
  },
];
