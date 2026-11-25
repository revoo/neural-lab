// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'interlude-chatgpt.old-new',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'Inside ChatGPT (released 30 November 2022), which part was the relatively recent addition on top of older ideas?',
    options: [
      {
        text: 'Post-training on human preferences to turn a next-token predictor into a conversational assistant; the transformer came from 2017 and statistical next-symbol prediction of English from Shannon in 1951.',
        correct: true,
        why: 'Language models built on the 2017 transformer and Shannon’s 1951 next-symbol framework already existed (like GPT-3 in 2020); post-training shaped them into an assistant people could converse with.',
      },
      {
        text: 'The transformer architecture was invented specifically in November 2022.',
        why: 'The transformer was introduced five years earlier, in 2017 (“Attention Is All You Need”).',
      },
      {
        text: 'Predicting the next symbol of English text was tried for the first time in 2020.',
        why: 'Claude Shannon analysed prediction and entropy of printed English back in 1951.',
      },
    ],
  },
  {
    id: 'interlude-chatgpt.nobel',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'Which two 2024 Nobel Prizes recognised work rooted in artificial neural networks and computational modelling?',
    options: [
      {
        text: 'Physics (John Hopfield and Geoffrey Hinton, for foundational discoveries enabling machine learning with neural networks) and Chemistry (David Baker for protein design, and Demis Hassabis and John Jumper for protein structure prediction).',
        correct: true,
        why: 'Both 2024 science prizes highlighted how neural-network methods moved from computer science into foundational physics and biology.',
      },
      {
        text: 'Physics (for the invention of CUDA) and Literature (for ChatGPT).',
        why: 'The 2024 Nobel Prize in Physics honoured Hopfield and Hinton for artificial neural networks, and the Chemistry prize honoured Baker, Hassabis and Jumper for protein design and structure prediction.',
      },
      {
        text: 'Economics (for ImageNet) and Medicine (for backpropagation).',
        why: 'The two 2024 AI-linked Nobel Prizes were in Physics and Chemistry.',
      },
    ],
  },
];
