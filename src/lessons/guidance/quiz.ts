// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'guide.conditioning',
    kind: 'choice',
    concepts: ['conditioning'],
    prompt: 'What does conditioning change about a diffusion model’s denoiser?',
    options: [
      {
        text: 'It reads the request as an extra input, so its best guess of the clean data averages only over data that matches the request.',
        correct: true,
        why: 'For our blobs, asking for blue means ignoring the orange blobs when averaging. With a perfect denoiser that alone put 99.4% of samples on blue blobs, fairly split.',
      },
      {
        text: 'It filters the finished samples, throwing away the ones that don’t match.',
        why: 'Nothing is thrown away: every sampling step is steered by the label from the very first, pure-noise step.',
      },
      {
        text: 'It changes the starting noise to look like the requested class.',
        why: 'The starting noise is the same standard bell whatever you ask for. The label changes the guesses made along the way.',
      },
    ],
  },
  {
    id: 'guide.cfg',
    kind: 'numeric',
    concepts: ['classifier-free-guidance'],
    tolerance: 0.005,
    make: (r) => {
      const u = [0.2, -0.4, 0.5][r.int(3)];
      const c = [0.8, 0.1, -0.3][r.int(3)];
      const w = [0, 1, 2, 3, 7.5][r.int(5)];
      const g = u + w * (c - u);
      return {
        prompt: `For one pixel, the unconditional noise prediction is ${u} and the prompt-conditioned prediction is ${c}. What is the guided prediction at guidance scale w = ${w}?`,
        answer: Number(g.toFixed(3)),
        explain: `guided = unconditional + w × (conditional − unconditional) = ${u} + ${w} × (${c} − ${u}) = ${g.toFixed(3)}. ${w === 0 ? 'w = 0 ignores the prompt.' : w === 1 ? 'w = 1 is plain conditioning.' : 'w > 1 overshoots past the conditional prediction, exaggerating what the prompt changes.'}`,
      };
    },
  },
  {
    id: 'guide.bayes',
    kind: 'choice',
    concepts: ['classifier-free-guidance'],
    prompt: 'Why is the difference between the labelled and unlabelled predictions called a “classifier’s push”?',
    options: [
      {
        text: 'By Bayes’ rule, the conditional score is the unconditional score plus the gradient of log p(label | x): the direction that makes the noisy point look more like the label.',
        correct: true,
        why: 'So guidance amplifies an implicit classifier. That is why it is strongest where a classifier would be unsure.',
      },
      {
        text: 'Because the model contains a separate classifier network.',
        why: 'That was classifier guidance. Classifier-free guidance gets the same push from one network run with and without the label.',
      },
      {
        text: 'Because it is only used for class labels, not text prompts.',
        why: 'The same algebra works for any condition, including a text prompt.',
      },
    ],
  },
  {
    id: 'guide.cost',
    kind: 'choice',
    concepts: ['classifier-free-guidance'],
    prompt: 'You raise the guidance scale from 1 to 5. With a perfect denoiser and plenty of steps, what happens to samples of “blue”?',
    options: [
      {
        text: 'They become more stereotypically blue and less varied: the blue blob that looks most like orange loses its share.',
        correct: true,
        why: 'In our world the inner blob went from about a third to 0.0%, and the outer blob took 91.7%. With a perfect denoiser, w = 1 was already faithful.',
      },
      {
        text: 'They follow the label more accurately, with no downside.',
        why: 'They were already on blue blobs at w = 1 (99.4%). Guidance can’t add accuracy that isn’t missing; it spends variety.',
      },
      {
        text: 'Nothing changes once samples are on the right class.',
        why: 'The push acts at every noise level, including the high-noise steps where samples choose their blob, so it changes which blob they choose.',
      },
    ],
  },
  {
    id: 'guide.why',
    kind: 'choice',
    concepts: ['classifier-free-guidance'],
    prompt: 'If guidance distorts the distribution, why does nearly every image generator use a scale well above 1?',
    options: [
      {
        text: 'Real denoisers are imperfect and samplers use few steps, so plain conditioning gives muddled, loosely on-prompt images. Guidance makes them crisp and on-prompt, and people accept less variety in exchange.',
        correct: true,
        why: 'In our world, at 3 steps, guidance raised samples landing on a blob from 73.5% to 92.2%, while concentrating them on one blob.',
      },
      {
        text: 'Because w = 1 ignores the prompt.',
        why: 'w = 0 ignores the prompt. w = 1 is plain conditioning.',
      },
      {
        text: 'Because higher guidance makes generation faster.',
        why: 'Each guided step costs two network evaluations, not fewer. Guidance can make few-step sampling acceptable, which is a different thing.',
      },
    ],
  },
  {
    id: 'guide.negative',
    kind: 'choice',
    concepts: ['classifier-free-guidance'],
    prompt: 'How does a negative prompt (“blurry, extra fingers”) fit into classifier-free guidance?',
    options: [
      {
        text: 'Its prediction replaces the unconditional one, so the guided step pushes towards the prompt and away from the negative prompt.',
        correct: true,
        why: 'guided = negative + w × (prompt − negative). The “difference the prompt makes” is now measured against what you don’t want.',
      },
      {
        text: 'The finished image is compared with the negative prompt and regenerated if it matches.',
        why: 'There is no checking afterwards; the negative prompt steers every step.',
      },
      {
        text: 'It is subtracted from the prompt text before encoding.',
        why: 'The two prompts are encoded separately and give two noise predictions; the arithmetic happens on the predictions.',
      },
    ],
  },
];
