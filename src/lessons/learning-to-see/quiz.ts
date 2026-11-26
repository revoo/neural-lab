// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'learning-to-see.edges-why',
    kind: 'choice',
    concepts: ['convolution', 'cnn'],
    prompt: 'Nobody tells a CNN to detect edges, yet its first-layer filters often turn into oriented edge or bar detectors. Why?',
    options: [
      {
        text: 'Because a 5 × 5 filter can only see a tiny patch, and in small patches of real photos the most useful thing to report is where brightness changes, and in which direction.',
        correct: true,
        why: 'Training pushes every filter towards whatever helps the loss; at this scale, contrast boundaries are the most informative local pattern, so many filters settle there.',
      },
      {
        text: 'Because the network starts with edge detectors and training only sharpens them.',
        why: 'It starts from random numbers; in our run no filter matched an edge template better than chance at step 0.',
      },
      {
        text: 'Because the Fashion-MNIST photos were pre-processed with an edge filter.',
        why: 'The photos are plain grey images. The edges come from training alone.',
      },
    ],
  },
  {
    id: 'learning-to-see.chance-bar',
    kind: 'choice',
    concepts: ['cnn'],
    prompt: 'The lesson counts a filter as an edge detector when it matches an oriented template better than 99% of random filters. Why compare with random filters at all?',
    options: [
      {
        text: 'Because random noise sometimes looks like an edge by luck; the bar says how good a match has to be before luck is an unlikely explanation.',
        correct: true,
        why: 'Without a chance baseline, any threshold would be a guess. With 16 templates to choose from, a random 5 × 5 filter already scores about 0.35 on average.',
      },
      { text: 'Because random filters are what a well-trained network converges to.', why: 'Trained filters end up far more structured than random ones; that is the point of the comparison.' },
      { text: 'To make the network train faster.', why: 'The bar is a measurement, not part of training. It changes nothing the network does.' },
    ],
  },
  {
    id: 'learning-to-see.occlusion',
    kind: 'choice',
    concepts: ['cnn'],
    prompt: 'An occlusion map shows a large drop in the “Sneaker” probability when the toe area is covered. What does it tell you?',
    options: [
      {
        text: 'That this network, on this photo, relies on the toe area: hiding it removes evidence the network was using.',
        correct: true,
        why: 'Occlusion measures the network’s behaviour directly, one forward pass per position. It describes this model and this photo, not sneakers in general.',
      },
      { text: 'That the toe is what makes something a sneaker.', why: 'It reports what the network uses, which can differ from what a person would call defining (or be a quirk of the training photos).' },
      { text: 'That the network has stored a picture of a toe.', why: 'It has learned features that respond there; nothing like a stored image is involved.' },
    ],
  },
  {
    id: 'learning-to-see.separate',
    kind: 'choice',
    concepts: ['embedding', 'pca'],
    prompt: 'In the class map, photos of the same class drift together during training. Which measurement makes that claim checkable?',
    options: [
      {
        text: 'Classifying each test photo by the nearest class average in the inner layer, and comparing that accuracy with the same method on raw pixels.',
        correct: true,
        why: 'If the inner layer groups classes better than the pixels do, the nearest-average rule gets more right there. The 2D picture is only a projection; the accuracy uses all 32 numbers.',
      },
      { text: 'Looking at the 2D picture and judging whether the colours look separated.', why: 'A 2D projection can hide separation (or suggest it); a number computed in the full space can’t be talked into anything.' },
      { text: 'Checking that the training loss went down.', why: 'Loss can fall for many reasons; it doesn’t say how the inner space is arranged.' },
    ],
  },
  {
    id: 'learning-to-see.adversarial',
    kind: 'choice',
    concepts: ['cnn'],
    prompt: 'A few changed pixels flip the network’s answer, though a person would still see the same coat. What is the most accurate conclusion?',
    options: [
      {
        text: 'The network’s decision depends on features that are not the ones a person relies on, so small, targeted changes can move it across a decision boundary.',
        correct: true,
        why: 'The attack searches for the changes that lower the true class’s lead fastest. It works because the network’s evidence is not organised the way human perception is.',
      },
      { text: 'The network is broken and its accuracy on ordinary photos can’t be trusted.', why: 'Its accuracy on ordinary test photos is measured and real; adversarial changes are carefully chosen, not ordinary.' },
      { text: 'Bigger networks are immune to this.', why: 'Large ImageNet networks are fooled too; in 2017 a single pixel was enough on 16.04% of the ImageNet test images tried.' },
    ],
  },
];
