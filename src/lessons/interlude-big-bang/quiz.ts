// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'interlude-big-bang.ingredients',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'AlexNet’s core architecture (a convolutional network trained by backpropagation) dated back to 1986–1989. What two new ingredients made its 2012 ImageNet win possible?',
    options: [
      {
        text: 'A massive labelled dataset (ImageNet’s 1.2 million training images) and fast parallel arithmetic on programmable GPUs (via CUDA).',
        correct: true,
        why: 'The ideas had existed for over twenty years; what changed between 1989 and 2012 was having over a million labelled photographs and consumer graphics cards programmable for general-purpose parallel maths.',
      },
      {
        text: 'Hand-engineered SIFT and HoG image descriptors paired with a support vector machine.',
        why: 'Those hand-engineered features were what computer vision used before 2012; AlexNet beat them by learning its features directly from pixels.',
      },
      {
        text: 'Replacing backpropagation with random search over weights.',
        why: 'AlexNet was trained with backpropagation and gradient descent across 60 million parameters over five to six days on two GTX 580 GPUs.',
      },
    ],
  },
  {
    id: 'interlude-big-bang.margin',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'Why did AlexNet’s 2012 result change computer vision almost overnight?',
    options: [
      {
        text: 'It achieved a 15.3% top-5 error rate on ILSVRC 2012 — more than 10.8 percentage points better than the runner-up — showing that learned features could beat years of hand-crafted pipelines.',
        correct: true,
        why: 'Before 2012, learning features directly from pixels was a minority view in computer vision. A >10.8 percentage-point jump on a 1,000-class benchmark overturned that consensus.',
      },
      {
        text: 'It was the first model ever to run on a CPU in under a millisecond.',
        why: 'AlexNet was large (60 million parameters) and needed two 3 GB GPUs for five to six days to train.',
      },
      {
        text: 'It solved ImageNet with 0% error on the first try.',
        why: 'AlexNet reached 15.3% top-5 error — dramatic for 2012, though later deeper networks such as ResNets pushed that much lower (3.57% by 2015).',
      },
    ],
  },
];
