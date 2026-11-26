// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'learning-to-speak.order',
    kind: 'order',
    concepts: ['language-model'],
    prompt: 'Put the measured milestones of the recorded run in the order they happened.',
    items: ['Its characters sort into kinds', 'Its letter mix matches the book', 'Its words have the right length', 'Its first previous-character heads appear', 'Half its words are real'],
    explain: 'Steps 36, 77, 464, 599 and 1,668. Cheap statistics come first; whole words need many more steps.',
  },
  {
    id: 'learning-to-speak.why-baseline',
    kind: 'choice',
    concepts: ['language-model'],
    prompt: 'Why is each milestone a statistic crossing a line set by real passages (or by chance), rather than a judgement made by reading the samples?',
    options: [
      {
        text: 'Because a reader sees what they expect; a stated, checkable criterion makes “it learned X at step N” true or false, and the same for every run.',
        correct: true,
        why: 'Impressions can be talked into anything. A threshold from real text says how close to real is close enough, and a test can pin the step.',
      },
      { text: 'Because the samples are too long to read.', why: 'Four samples of 200 characters are easy to read; the point is to avoid choosing the frame that looks most impressive.' },
      { text: 'Because the model can’t produce real text, so there is nothing to read.', why: 'It does produce real words; the question is when, and the statistics answer that precisely.' },
    ],
  },
  {
    id: 'learning-to-speak.induction',
    kind: 'choice',
    concepts: ['attention'],
    prompt: 'The test text is 24 random letters typed twice. An induction head would make the second copy easier to predict. In our model the second copy is not easier at all. What follows?',
    options: [
      {
        text: 'This model has not learned to copy from earlier in its context: no head finds the earlier occurrence and reads what came next.',
        correct: true,
        why: 'Its induction scores stay at chance (about 0.03) and its loss on the second copy is no lower. The prose reports that plainly rather than assuming a result from larger models.',
      },
      { text: 'Induction heads don’t exist in any real model.', why: 'They have been documented in larger transformer language models trained on far more text; they just didn’t form in ours.' },
      { text: 'The test is broken because the letters are random.', why: 'Random letters are the standard test exactly because nothing but copying can predict them; we also checked a repeated real passage, with the same result.' },
    ],
  },
  {
    id: 'learning-to-speak.embeddings',
    kind: 'choice',
    concepts: ['embedding'],
    prompt: 'Nobody tells the model that vowels are a kind of letter. Why do vowel embeddings end up near each other anyway?',
    options: [
      {
        text: 'Because vowels appear in similar places (between similar consonants), and giving similar vectors to characters that behave alike lowers the loss.',
        correct: true,
        why: 'The gradient moves each embedding towards whatever helps predict its neighbours; characters with similar neighbours get pulled the same way.',
      },
      { text: 'Because the embeddings are initialised in alphabetical order.', why: 'They start as random numbers; at step 0 the kinds are no more separable than shuffled labels.' },
      { text: 'Because the model reads the letters’ shapes.', why: 'It only ever sees a character’s number in the 68-character list; shape plays no part.' },
    ],
  },
  {
    id: 'learning-to-speak.steer',
    kind: 'choice',
    concepts: ['autoregressive'],
    prompt: 'Forcing one “k” after “There was once to ” makes the model write “king” by itself, but “daughter” needs four forced letters. What does that difference show?',
    options: [
      {
        text: 'How strongly each word is part of what it learned: “king” is so familiar that its first letter is enough, while “daughter” needs to be spelled most of the way.',
        correct: true,
        why: 'After a forced letter, greedy decoding follows the model’s own most likely continuation; the more it has learned a word, the sooner that continuation is the word.',
      },
      { text: 'That longer words always need one forcing per letter.', why: '“Daughter” has 8 letters and needed 4; “king” has 4 and needed 1. It depends on what the model knows, not just length.' },
      { text: 'That forcing characters damages the model’s weights.', why: 'Forcing only changes what it is given to read next; its weights are untouched.' },
    ],
  },
];
