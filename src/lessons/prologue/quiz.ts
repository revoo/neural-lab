// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'prologue.dartmouth',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'In 1955, the proposal for the Dartmouth summer project on artificial intelligence said what about how long progress would take?',
    options: [
      { text: 'That a significant advance could be made if a carefully selected group of scientists worked on it together for a summer.', correct: true, why: 'Those are the proposal’s own words. Learning machines that work as well as the ones in this course took about seventy years, and the people who wrote that sentence were some of the best minds of their time. Being wrong about how hard something is says nothing about how clever you are.' },
      { text: 'That it would take about seventy years.', why: 'That is roughly how long it did take, but nobody predicted it. The proposal hoped for a significant advance in a single summer.' },
      { text: 'That machines could never learn, only follow instructions.', why: 'The proposal assumed the opposite: that every aspect of learning could in principle be described precisely enough for a machine to simulate it.' },
    ],
  },
  {
    id: 'prologue.wait',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'Backpropagation was published in Nature in 1986, convolutional networks read handwritten digits in 1989, and LSTMs appeared in 1997. Why did deep learning only take off after 2012?',
    options: [
      { text: 'The ideas were largely there; what was missing was enough data and fast enough hardware to use them at scale.', correct: true, why: 'AlexNet in 2012 was a convolutional network trained by backpropagation, ideas from the 1980s, but trained on 1.2 million labelled images using two graphics cards. Rich Sutton’s essay “The Bitter Lesson” argues that this pattern, general methods plus more computation winning in the end, has repeated across seventy years of AI.' },
      { text: 'Nobody had thought of using many layers before 2012.', why: 'Multi-layer networks trained by backpropagation were published in 1986, and LeNet (1989) was already a layered convolutional network.' },
      { text: 'The older ideas were wrong and had to be replaced.', why: 'The same ideas, backpropagation and convolution, are inside AlexNet and are still used today. They were early, not wrong.' },
    ],
  },
  {
    id: 'prologue.order',
    kind: 'choice',
    concepts: ['learning'],
    prompt: 'In this course, when you meet an idea with a famous history, which usually comes first?',
    options: [
      { text: 'The experiment: you rediscover the idea in a lab, then find out who found it first, and when.', correct: true, why: 'That is the point of the labs. Working out why something has to be the way it is teaches more than being told, and it puts you in the same position as the people who first faced the problem.' },
      { text: 'The history: who invented it and when, then the idea.', why: 'The history is there, but mostly after the experiment, in the “You are standing where they stood” notes and the history cards below each lab.' },
      { text: 'The equation, then a worked example.', why: 'This course puts the concept and the experiment before the equation, so that by the time the equation appears it looks obvious.' },
    ],
  },
];
