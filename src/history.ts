// authored-by: Claude Opus 5.5
/**
 * History of every lab's idea: when, where, who, the mood of the moment, and
 * before → this → after. Every claim carries its own source URL, and the test
 * in history.test.ts checks that. Dates and quotes were checked against the
 * linked pages in November 2026 (docs/LAB_IMPROVEMENTS.md, "History"); author
 * affiliations were checked on each paper's title page. The timeline page and
 * the per-lab cards both read this file, so they cannot disagree.
 */

export type Claim = { text: string; source: string };

export type HistoryEntry = {
  /** Unique id; also used by <HistoryCard id> in the lesson MDX. */
  id: string;
  lessonId: string;
  /** The lab's name as shown in the lesson. */
  lab: string;
  /** Short title for the idea, used on the timeline. */
  idea: string;
  /** Key years, ascending. The first is where the timeline places the entry. */
  years: number[];
  /** Display form of the years, e.g. "1943 · 1958 · 1969". */
  when: string;
  /** Where it happened and who did it. */
  who: Claim[];
  /** What the field believed at the time. Left out rather than invented when no source was found. */
  mood?: Claim;
  /** Before → this → after, one sourced step each. */
  story: { before: Claim; this: Claim; after: Claim };
  /** A disputed point, named plainly instead of picking a hero. */
  dispute?: Claim;
};

const WP = (page: string) => `https://en.wikipedia.org/wiki/${page}`;

export const HISTORY: HistoryEntry[] = [
  {
    id: 'llm-serving-desk',
    lessonId: 'llm-serving',
    lab: 'The Serving Desk',
    idea: 'Iteration-level batching, chunked prefill and virtual memory for the KV cache',
    years: [2022, 2023],
    when: '2022 · 2023',
    who: [
      { text: 'Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim and Byung-Gon Chun presented Orca at OSDI 2022 (July 2022), introducing iteration-level scheduling and lifting GPT-3 175B throughput up to 36.9× over FasterTransformer.', source: 'https://www.usenix.org/conference/osdi22/presentation/yu' },
      { text: 'Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani and Ramachandran Ramjee proposed chunked prefills and decode-maximal batching in SARATHI in August 2023.', source: 'https://arxiv.org/abs/2308.16369' },
      { text: 'Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang and Ion Stoica introduced PagedAttention and vLLM in September 2023 (SOSP 2023).', source: 'https://arxiv.org/abs/2309.06180' },
    ],
    story: {
      before: { text: 'Early LLM serving engines scheduled whole requests in static batches and pre-allocated contiguous maximum-length KV buffers, wasting 60%–80% of KV memory to fragmentation and over-reservation.', source: 'https://arxiv.org/abs/2309.06180' },
      this: { text: 'PagedAttention borrowed virtual memory from operating systems — splitting KV caches into fixed-size blocks with block tables — cutting KV waste to under 4% and improving throughput 2–4× over FasterTransformer and Orca.', source: 'https://arxiv.org/abs/2309.06180' },
      after: { text: 'In January 2024 DistServe (OSDI 2024) went a step further, disaggregating prefill and decode onto separate GPU pools so long prompts never interfere with decode latency.', source: 'https://arxiv.org/abs/2401.09670' },
    },
    dispute: { text: 'Whether to mix chunked prefills and decodes on one GPU pool (SARATHI) or split them onto separate pools (DistServe) remains a live systems trade-off depending on cluster size and prompt-to-reply ratios.', source: 'https://arxiv.org/abs/2401.09670' },
  },
  {
    id: 'speculative-decoding',
    lessonId: 'speculative-decoding',
    lab: 'Tune the Draft',
    idea: 'Drafting ahead with a small model and verifying losslessly in parallel',
    years: [2022, 2023],
    when: '2022 · 2023',
    who: [
      { text: 'Yaniv Leviathan, Matan Kalman and Yossi Matias (Google Research) posted “Fast Inference from Transformers via Speculative Decoding” on 30 November 2022 — the same day ChatGPT launched — later presented as an ICML 2023 Oral.', source: 'https://arxiv.org/abs/2211.17192' },
      { text: 'Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper (DeepMind) independently published “Accelerating Large Language Model Decoding with Speculative Sampling” on 2 February 2023, demonstrating 2–2.5× speed-ups on 70B Chinchilla.', source: 'https://arxiv.org/abs/2302.01318' },
    ],
    story: {
      before: { text: 'Autoregressive decoding was widely treated as strictly one forward pass per output token unless you compressed the model or approximated its output distribution.', source: 'https://arxiv.org/abs/2211.17192' },
      this: { text: 'Leviathan, Kalman and Matias paired a fast draft model with a rejection-sampling rule that preserves the target model’s distribution identically, achieving 2×–3× speed-ups on T5-XXL without retraining or changing outputs.', source: 'https://arxiv.org/abs/2211.17192' },
      after: { text: 'Later systems such as Medusa (January 2024) replaced the separate draft model with lightweight prediction heads attached directly to the target model’s own hidden states.', source: 'https://arxiv.org/abs/2401.10774' },
    },
  },
  {
    id: 'learning-to-speak',
    lessonId: 'learning-to-speak',
    lab: 'Steer the Sentence',
    idea: 'Watching a character model learn to write, and looking for the heads that copy',
    years: [2015, 2022],
    when: '2015 · 2022',
    who: [
      { text: 'Andrej Karpathy published “The Unreasonable Effectiveness of Recurrent Neural Networks” on 21 May 2015, training character-level networks on texts such as Tolstoy’s War and Peace and sampling from them every 100 iterations of training.', source: 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/' },
      { text: 'Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph and colleagues described “induction heads” in September 2022: attention heads that complete sequences like [A][B] … [A] → [B].', source: 'https://arxiv.org/abs/2209.11895' },
    ],
    story: {
      before: { text: 'At iteration 100 Karpathy’s War and Peace model sampled random jumbles of letters.', source: 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/' },
      this: { text: 'By iteration 500 its words were separated by spaces and it could spell short common words such as “we”, “He” and “and”; by about iteration 2,000 it wrote properly spelled words, quotations and names.', source: 'https://karpathy.github.io/2015/05/21/rnn-effectiveness/' },
      after: { text: 'Olsson and colleagues found that induction heads develop at precisely the same point as a sudden increase in in-context learning, visible as a bump in the training loss.', source: 'https://arxiv.org/abs/2209.11895' },
    },
    dispute: { text: 'How much of in-context learning induction heads explain is still open: the authors call their evidence for the general claim preliminary and indirect, strong and causal for small attention-only models but correlational for larger models with MLPs.', source: 'https://arxiv.org/abs/2209.11895' },
  },
  {
    id: 'fool-classifier',
    lessonId: 'learning-to-see',
    lab: 'Fool the Classifier',
    idea: 'Adversarial examples: tiny changes that flip a network’s answer',
    years: [2013, 2014, 2017],
    when: '2013 · 2014 · 2017',
    who: [
      { text: 'Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow and Rob Fergus reported in December 2013 that a network can be made to misclassify an image by an imperceptible perturbation.', source: 'https://arxiv.org/abs/1312.6199' },
      { text: 'Ian Goodfellow, Jonathon Shlens and Christian Szegedy followed in December 2014 with an explanation and a fast way to generate such examples.', source: 'https://arxiv.org/abs/1412.6572' },
    ],
    story: {
      before: { text: 'Deep networks had recently achieved state-of-the-art performance on speech and visual recognition, but the same expressiveness let them learn uninterpretable solutions with possibly counter-intuitive properties.', source: 'https://arxiv.org/abs/1312.6199' },
      this: { text: 'Szegedy and colleagues found that networks learn input–output mappings that are discontinuous to a significant extent: a perturbation found by maximising the prediction error changes the answer, and the same perturbation can fool a different network trained on a different subset of the data.', source: 'https://arxiv.org/abs/1312.6199' },
      after: { text: 'In 2017 Jiawei Su, Danilo Vasconcellos Vargas and Kouichi Sakurai changed a single pixel and fooled the network on 67.97% of the Kaggle CIFAR-10 test images they tried, and 16.04% of ImageNet test images.', source: 'https://arxiv.org/abs/1710.08864' },
    },
    dispute: { text: 'Why it happens was argued over: early explanations blamed nonlinearity and overfitting, while Goodfellow, Shlens and Szegedy argued the primary cause is the networks’ linear nature.', source: 'https://arxiv.org/abs/1412.6572' },
  },
  {
    id: 'tiny-gpt',
    lessonId: 'tiny-gpt',
    lab: 'Ship the Tiny GPT',
    idea: 'A transformer trained only to predict the next token',
    years: [2017, 2019],
    when: '2017 · 2019',
    who: [
      { text: 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin proposed the Transformer in June 2017, in Attention Is All You Need.', source: 'https://arxiv.org/abs/1706.03762' },
      { text: 'OpenAI announced GPT-2 on 14 February 2019: a large language model pre-trained on a dataset of 8 million web pages.', source: 'https://en.wikipedia.org/wiki/GPT-2' },
    ],
    mood: { text: 'GPT-2 was released in stages because of worries about misuse; a February 2019 headline in The Guardian read “New AI fake text generator may be too dangerous to release, say creators”.', source: 'https://en.wikipedia.org/wiki/GPT-2' },
    story: {
      before: { text: 'The best sequence models were complex recurrent or convolutional networks; the Transformer was based solely on attention, “dispensing with recurrence and convolutions entirely”.', source: 'https://arxiv.org/abs/1706.03762' },
      this: { text: 'GPT-2 was a transformer decoder with no encoder (GPT-1 with modified normalization), trained to predict the next item in a sequence on WebText: pages linked from Reddit posts with at least 3 karma.', source: 'https://en.wikipedia.org/wiki/GPT-2' },
      after: { text: 'By November 2019 OpenAI said it had seen no strong evidence of misuse so far, and released the full 1.5-billion-parameter model on 5 November 2019.', source: 'https://en.wikipedia.org/wiki/GPT-2' },
    },
    dispute: { text: 'Not everyone accepted the caution: Anima Anandkumar said there was no evidence GPT-2 could pose the threats described, and called withholding the model the “opposite of open”.', source: 'https://en.wikipedia.org/wiki/GPT-2' },
  },
  {
    id: 'multimodal-vit',
    lessonId: 'multimodal-vit',
    lab: 'Wire the Multimodal Model',
    idea: 'Images as sequences of patch tokens, and pictures aligned with words',
    years: [2020, 2021],
    when: '2020 · 2021',
    who: [
      { text: 'Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby introduced the Vision Transformer in October 2020.', source: 'https://arxiv.org/abs/2010.11929' },
      { text: 'Alec Radford, Jong Wook Kim and colleagues introduced CLIP in February 2021, trained on 400 million image–text pairs collected from the internet.', source: 'https://arxiv.org/abs/2103.00020' },
    ],
    mood: { text: 'In 2020 the transformer had become the standard architecture for language, but in vision attention was mostly used together with convolutional networks or to replace parts of them.', source: 'https://arxiv.org/abs/2010.11929' },
    story: {
      before: { text: 'Computer vision systems were trained to predict a fixed set of predetermined object categories, so every new visual concept needed more labelled data.', source: 'https://arxiv.org/abs/2103.00020' },
      this: { text: 'The Vision Transformer applied a standard transformer directly to sequences of image patches; trained on mid-sized datasets it fell a few points short of comparable ResNets, but pre-trained on 14 to 300 million images it matched or beat the state of the art: large-scale training trumped inductive bias.', source: 'https://arxiv.org/abs/2010.11929' },
      after: { text: 'CLIP learned by predicting which caption goes with which image, and then classified images using only sentences, matching the original ResNet-50 on ImageNet without using any of its 1.28 million training examples.', source: 'https://arxiv.org/abs/2103.00020' },
    },
  },
  {
    id: 'reasoning',
    lessonId: 'reasoning',
    lab: 'Spend the Test-Time Budget',
    idea: 'Chains of thought, voting, and reinforcement learning from checkable answers',
    years: [2022, 2024, 2025],
    when: '2022 · 2024 · 2025',
    who: [
      { text: 'Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le and Denny Zhou showed in January 2022 that prompting with a few worked chains of thought improves large language models’ reasoning.', source: 'https://arxiv.org/abs/2201.11903' },
      { text: 'Xuezhi Wang, Jason Wei and colleagues proposed self-consistency in March 2022: sample several reasoning paths and keep the most consistent answer.', source: 'https://arxiv.org/abs/2203.11171' },
      { text: 'Zhihong Shao and colleagues introduced Group Relative Policy Optimization (GRPO) with DeepSeekMath in February 2024.', source: 'https://arxiv.org/abs/2402.03300' },
    ],
    mood: { text: 'In 2024, little research had tried to understand how test-time inference methods scale, and much of what existed reported negative results.', source: 'https://arxiv.org/abs/2408.03314' },
    story: {
      before: { text: 'Chain-of-thought prompting worked by showing a model a few demonstrations of intermediate reasoning steps; a 540-billion-parameter model prompted with eight such examples reached state-of-the-art accuracy on the GSM8K maths word problems.', source: 'https://arxiv.org/abs/2201.11903' },
      this: { text: 'Snell, Lee, Xu and Kumar (2024) found that in a FLOPs-matched comparison, test-time compute let a smaller model outperform a 14 times larger one on problems where the smaller model had somewhat non-trivial success rates.', source: 'https://arxiv.org/abs/2408.03314' },
      after: { text: 'DeepSeek-R1 (January 2025) showed that reasoning abilities can be incentivised through pure reinforcement learning, without human-labelled reasoning trajectories, with patterns such as self-reflection and verification emerging.', source: 'https://arxiv.org/abs/2501.12948' },
    },
  },
  {
    id: 'moe',
    lessonId: 'moe',
    lab: 'Balance the Experts',
    idea: 'Sparse mixtures of experts: routing each token to a few specialists',
    years: [2017, 2021],
    when: '2017 · 2021',
    who: [
      { text: 'Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton and Jeff Dean introduced the Sparsely-Gated Mixture-of-Experts layer in January 2017.', source: 'https://arxiv.org/abs/1701.06538' },
      { text: 'William Fedus, Barret Zoph and Noam Shazeer introduced the Switch Transformer in January 2021.', source: 'https://arxiv.org/abs/2101.03961' },
    ],
    mood: { text: 'Conditional computation, where parts of a network are active only for some examples, had been proposed in theory as a way to grow capacity without growing computation, but in practice it faced significant algorithmic and performance challenges.', source: 'https://arxiv.org/abs/1701.06538' },
    story: {
      before: { text: 'In 1991, Robert Jacobs, Michael Jordan, Steven Nowlan and Geoffrey Hinton’s adaptive mixtures of local experts used a linear-softmax gating function to weigh the predictions of every expert.', source: WP('Mixture_of_experts') },
      this: { text: 'Shazeer et al. (2017) let a trainable gating network pick a sparse combination of up to thousands of feed-forward experts, building MoE layers with up to 137 billion parameters between stacked LSTM layers.', source: 'https://arxiv.org/abs/1701.06538' },
      after: { text: 'The Switch Transformer (2021) sent each token to a single expert, with one auxiliary load-balancing loss weighted by α = 10⁻², and trained models with up to 1.6 trillion parameters.', source: 'https://arxiv.org/abs/2101.03961' },
    },
  },
  {
    id: 'gans',
    lessonId: 'gans',
    lab: 'Keep the Duel Balanced',
    idea: 'Generative adversarial networks and the two-player minimax duel',
    years: [2014, 2021],
    when: '2014 · 2021',
    who: [
      { text: 'Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville and Yoshua Bengio at Université de Montréal introduced Generative Adversarial Networks in June 2014.', source: 'https://arxiv.org/abs/1406.2661' },
      { text: 'Prafulla Dhariwal and Alex Nichol (OpenAI) showed in May 2021 that diffusion models surpass GANs on ImageNet synthesis.', source: 'https://arxiv.org/abs/2105.05233' },
    ],
    story: {
      before: { text: 'Deep generative models relied on Markov chains or approximate inference networks (such as variational autoencoders, introduced around the same time in 2013–2014).', source: 'https://arxiv.org/abs/1406.2661' },
      this: { text: 'Goodfellow et al. (2014) trained a generator G and discriminator D together via backpropagation in a minimax two-player game, proving a unique equilibrium with G matching the data and D equal to 1/2 everywhere.', source: 'https://arxiv.org/abs/1406.2661' },
      after: { text: 'In 2021, Dhariwal and Nichol achieved FID 2.97 on ImageNet 128×128 and 4.59 on 256×256 with diffusion models, matching BigGAN-deep while maintaining better coverage of the distribution.', source: 'https://arxiv.org/abs/2105.05233' },
    },
  },
  {
    id: 'svm',
    lessonId: 'svm',
    lab: 'Widen the Street',
    idea: 'Maximum-margin hyperplanes, kernels and the soft margin',
    years: [1992, 1995],
    when: '1992 · 1995',
    who: [
      { text: 'Vladimir Vapnik and Alexey Chervonenkis laid the statistical learning foundations (VC theory) and original linear maximum-margin algorithm.', source: WP('Support_vector_machine') },
      { text: 'Bernhard Boser, Isabelle Guyon and Vladimir Vapnik introduced nonlinear kernel maximum-margin classifiers in 1992.', source: WP('Support_vector_machine') },
      { text: 'Corinna Cortes and Vladimir Vapnik developed the soft-margin support-vector network at AT&T Bell Laboratories, proposed in 1993 and published in Machine Learning in 1995.', source: WP('Support_vector_machine') },
    ],
    mood: { text: 'Throughout 1990–2010, neural networks did not outperform competing machine-learning methods such as support vector machines, kernel regression and boosting.', source: WP('AlexNet') },
    story: {
      before: { text: 'Vapnik’s original maximum-margin algorithm constructed only a linear classifier for cleanly separable data.', source: WP('Support_vector_machine') },
      this: { text: 'Adding the kernel trick (1992) handled curved boundaries via dot products in a lifted space, and the soft margin (1995) handled noisy, overlapping classes.', source: WP('Support_vector_machine') },
      after: { text: 'SVMs dominated 1990s and 2000s benchmarks until large labelled datasets like ImageNet and GPU-trained deep networks overtook them on raw images in 2012.', source: WP('AlexNet') },
    },
    dispute: { text: 'Wikipedia’s own Support Vector Machine article dates Vapnik and Chervonenkis’s original linear algorithm to 1964 in its history section and to 1963 in its nonlinear-kernels section.', source: WP('Support_vector_machine') },
  },
  {
    id: 'mark-i',
    lessonId: 'perceptron',
    lab: 'Train the Mark I',
    idea: 'The perceptron learning rule and its convergence proof',
    years: [1957, 1959, 1962],
    when: '1957 · 1959 · 1962',
    who: [
      { text: 'Frank Rosenblatt, at the Cornell Aeronautical Laboratory, simulates the perceptron on an IBM 704 in 1957.', source: WP('Perceptron') },
      { text: 'His team assembles the Mark I Perceptron at the laboratory in Buffalo, New York, in 1959: a camera of 20 × 20 photocells, with the weights held in potentiometers that electric motors turned during learning.', source: WP('Perceptron') },
      { text: 'Albert Novikoff publishes a proof in 1962 that the rule stops making mistakes after finitely many on any data a line can separate.', source: WP('Perceptron') },
    ],
    mood: { text: 'After a 1958 Navy press conference, The New York Times called the perceptron "the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence."', source: WP('Perceptron') },
    story: {
      before: { text: 'Warren McCulloch and Walter Pitts proposed the threshold unit in 1943; its weights had to be chosen by its designer.', source: WP('Artificial_neuron') },
      this: { text: 'The perceptron learns its own weights from labelled examples, changing them only when it gets an example wrong.', source: WP('Perceptron') },
      after: { text: 'In 1969 Marvin Minsky and Seymour Papert\'s book Perceptrons set out the limits of what such machines could do.', source: WP('AI_winter') },
    },
    dispute: { text: 'Sources disagree on the Mark I\'s date: Wikipedia\'s Mark I article says a first version ran experiments as early as 1958, while its perceptron article dates the machine\'s assembly to 1959 and its first public demonstration to June 1960.', source: WP('Mark_I_Perceptron') },
  },
  {
    id: "twenty-questions",
    lessonId: "information",
    lab: "Twenty Questions Compressor",
    idea: "Optimal binary prefix codes",
    years: [1951, 1952],
    when: "1951 · 1952",
    who: [{ text: "David Huffman developed his prefix code while a graduate student at MIT.", source: WP("Huffman_coding") }],
    story: {
      before: { text: "Robert Fano offered his class a term paper on efficient binary codes as an alternative to a final exam.", source: WP("Huffman_coding") },
      this: { text: "His 1952 paper built the tree from the bottom up, improving on the top-down Shannon–Fano method.", source: WP("Huffman_coding") },
      after: { text: "A related idea from the same period is relative entropy, introduced by Solomon Kullback and Richard Leibler in 1951.", source: WP("Kullback–Leibler_divergence") },
    },
  },
  {
    id: "forecaster",
    lessonId: "probability",
    lab: "Calibrate the Forecaster",
    idea: "A probability distribution with temperature",
    years: [1902, 1989, 1990],
    when: "1902 · 1989 · 1990",
    who: [{ text: "Josiah Willard Gibbs investigated the general form of the Boltzmann distribution in 1902.", source: WP("Boltzmann_distribution") }],
    story: {
      before: { text: "In statistical mechanics, the distribution assigns probabilities to states according to energy and absolute temperature.", source: WP("Boltzmann_distribution") },
      this: { text: "The same normalised exponential form can turn scores into probabilities.", source: WP("Softmax_function") },
      after: { text: "The machine learning name softmax is credited to John S. Bridle in conference papers presented in 1989 and published in 1990.", source: WP("Softmax_function") },
    },
  },
  {
    id: "slipping-gear",
    lessonId: "chain-rule",
    lab: "Find the Slipping Gear",
    idea: "Local slopes and the backward walk",
    years: [1970],
    when: "1970",
    who: [{ text: "Gottfried Wilhelm Leibniz mentioned the chain rule in a 1676 memoir.", source: WP("Chain_rule") }],
    story: {
      before: { text: "Leibniz considered the derivative of the square root of a quadratic expression.", source: WP("Chain_rule") },
      this: { text: "He treated the expression as the composition of the square root and the quadratic.", source: WP("Chain_rule") },
      after: { text: "Seppo Linnainmaa described efficient backward propagation of derivatives in his 1970 master’s thesis.", source: WP("Seppo_Linnainmaa") },
    },
  },
  {
    id: "probe",
    lessonId: "derivatives",
    lab: "The Sensitivity Probe",
    idea: "Two independent approaches to calculus",
    years: [1710, 1716],
    when: "1710–1716",
    who: [{ text: "Isaac Newton and Gottfried Wilhelm Leibniz independently developed calculus in Europe.", source: WP("Leibniz–Newton_calculus_controversy") }],
    story: {
      before: { text: "Newton’s surviving manuscript from 1666 records early work on his method of fluxions.", source: WP("Leibniz–Newton_calculus_controversy") },
      this: { text: "Leibniz published his differential calculus in 1684.", source: WP("Leibniz–Newton_calculus_controversy") },
      after: { text: "The question of priority later prompted a dispute over whether Leibniz had developed calculus independently.", source: WP("Leibniz–Newton_calculus_controversy") },
    },
  },
  {
    id: 'pipeline',
    lessonId: 'matrix-multiplication',
    lab: 'Build the Pipeline',
    idea: 'Faster matrix multiplication',
    years: [1969],
    when: '1969',
    who: [{ text: 'Volker Strassen published a faster matrix multiplication algorithm in 1969.', source: WP('Strassen_algorithm') }],
    story: {
      before: { text: 'Schoolbook multiplication computes each output by taking one row against one column.', source: WP('Strassen_algorithm') },
      this: { text: 'Strassen used sums and differences of blocks to reduce the number of block multiplications.', source: WP('Strassen_algorithm') },
      after: { text: 'The result established that the schoolbook algorithm for general matrix multiplication could be beaten.', source: WP('Strassen_algorithm') },
    },
  },
  {
    id: 'untangle',
    lessonId: 'matrices',
    lab: 'Untangle the Data',
    idea: 'A direction that separates classes',
    years: [1936],
    when: '1936',
    who: [{ text: 'Ronald Fisher developed a discriminant for two classes in 1936.', source: WP('Linear_discriminant_analysis') }],
    story: {
      before: { text: 'Fisher posed a taxonomic problem involving several measurements of each specimen.', source: WP('Linear_discriminant_analysis') },
      this: { text: 'Project the measurements onto a line chosen to spread the class means apart relative to the variation within classes.', source: WP('Linear_discriminant_analysis') },
      after: { text: 'C. R. Rao extended the analysis to more than two classes.', source: WP('Linear_discriminant_analysis') },
    },
  },
  {
    id: 'spam-scorer',
    lessonId: 'dot-product',
    lab: 'Tune the Spam Scorer',
    idea: 'Documents as vectors, compared by cosine',
    years: [1975],
    when: '1975',
    who: [
      { text: 'Gerard Salton, professor of computer science at Cornell University, called "the father of Information Retrieval".', source: WP('Gerard_Salton') },
      { text: 'His group at Cornell developed the SMART Information Retrieval System, which he had started at Harvard.', source: WP('Gerard_Salton') },
    ],
    story: {
      before: { text: 'The SMART Information Retrieval System, where the vector space model was first used.', source: WP('Vector_space_model') },
      this: { text: '"A vector space model for automatic indexing" (1975): documents and queries become vectors of term counts, and their similarity is the cosine between them.', source: WP('Gerard_Salton') },
      after: { text: 'The vector space model went on to be used in information filtering, information retrieval, indexing and relevance rankings.', source: WP('Vector_space_model') },
    },
  },
  {
    id: 'feature-workbench',
    lessonId: 'vectors',
    lab: 'The Feature Workbench',
    idea: 'Things measured as lists of numbers',
    years: [1936],
    when: '1936',
    who: [
      { text: 'The British statistician and biologist Ronald Fisher made the iris data famous in his 1936 paper "The use of multiple measurements in taxonomic problems".', source: WP('Iris_flower_data_set') },
      { text: 'Two of the three species were collected in the Gaspé Peninsula, "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".', source: WP('Iris_flower_data_set') },
    ],
    story: {
      before: { text: 'The botanist Edgar Anderson measured the flowers to quantify how three related iris species vary.', source: WP('Iris_flower_data_set') },
      this: { text: 'Each flower became four numbers, the length and width of its sepals and petals in centimetres: 50 flowers from each of three species.', source: WP('Iris_flower_data_set') },
      after: { text: 'It became a typical test case for many classification techniques in machine learning, and a beginner\'s data set to this day.', source: WP('Iris_flower_data_set') },
    },
  },
  {
    id: 'dice',
    lessonId: 'statistics',
    lab: 'The Dice Workshop',
    idea: 'Sums of random parts form a bell',
    years: [1733, 1812],
    when: '1733 · 1812',
    who: [
      { text: 'Abraham de Moivre approximated the spread of coin-toss counts with the bell curve in a 1733 article.', source: WP('Central_limit_theorem') },
      { text: 'Pierre-Simon Laplace extended it in Théorie analytique des probabilités (1812).', source: WP('Central_limit_theorem') },
    ],
    story: {
      before: { text: 'De Moivre needed the distribution of the number of heads in many coin tosses, an exact sum that grows unwieldy by hand.', source: WP('Central_limit_theorem') },
      this: { text: 'The de Moivre–Laplace theorem, "the earliest version" of the central limit theorem: sums of many random parts come out bell-shaped.', source: WP('Central_limit_theorem') },
      after: { text: 'He initialisation (2015) scales random weights so that sums of hundreds of them keep a steady spread, layer after layer.', source: 'https://arxiv.org/abs/1502.01852' },
    },
  },
  {
    id: 'superposition',
    lessonId: 'high-dimensions',
    lab: 'Superposition',
    idea: 'Models pack more features than neurons',
    years: [2022],
    when: '2022',
    who: [{ text: 'Nelson Elhage and colleagues at Anthropic (with Harvard), Toy Models of Superposition.', source: 'https://arxiv.org/abs/2209.10652' }],
    story: {
      before: { text: 'The hope that each neuron stands for one concept, so a network can be read neuron by neuron.', source: 'https://arxiv.org/abs/2209.10652' },
      this: { text: 'Small models store "additional sparse features in superposition": more features than dimensions, at almost-perpendicular angles.', source: 'https://arxiv.org/abs/2209.10652' },
      after: { text: 'Interpretability research that pulls the packed features apart again, with sparse dictionary learning.', source: 'https://transformer-circuits.pub/2023/monosemantic-features' },
    },
  },
  {
    id: 'logic',
    lessonId: 'logic-lab',
    lab: 'Logic workbench',
    idea: 'Threshold neurons, the perceptron and XOR',
    years: [1943, 1958, 1969],
    when: '1943 · 1958 · 1969',
    who: [
      { text: 'Warren McCulloch (University of Illinois at Chicago) and Walter Pitts (University of Chicago) describe the threshold neuron in 1943.', source: WP('Warren_Sturgis_McCulloch') },
      { text: 'Frank Rosenblatt builds the perceptron at the Cornell Aeronautical Laboratory in Buffalo, New York (1958).', source: WP('Frank_Rosenblatt') },
      { text: 'Marvin Minsky and Seymour Papert publish Perceptrons (1969).', source: WP('Perceptrons_(book)') },
    ],
    mood: { text: 'For neural networks, funding "was difficult to find in the 1970s and early 1980s".', source: WP('AI_winter') },
    story: {
      before: { text: 'A threshold neuron with hand-set weights can compute AND and OR, but no single one computes XOR.', source: WP('Artificial_neuron') },
      this: { text: 'The perceptron learns its own weights from examples.', source: WP('Perceptron') },
      after: { text: 'Perceptrons (1969) proves what single-layer perceptrons cannot do; multilayer networks escape the limit, but nobody yet knew how to train them.', source: WP('AI_winter') },
    },
    dispute: { text: 'The popular story that the book single-handedly killed neural networks is a simplification: Wikipedia\'s article on the book says "there are many mistakes" in that story.', source: WP('Perceptrons_(book)') },
  },
  {
    id: 'prune',
    lessonId: 'network-playground',
    lab: 'Prune it',
    idea: 'Pruning: most of a trained network can go',
    years: [2015, 2018],
    when: '2015 → 2018',
    who: [
      { text: 'Song Han, Huizi Mao and William Dally (Stanford, NVIDIA and Tsinghua), Deep Compression.', source: 'https://arxiv.org/abs/1510.00149' },
      { text: 'Jonathan Frankle and Michael Carbin (MIT CSAIL), The Lottery Ticket Hypothesis.', source: 'https://arxiv.org/abs/1803.03635' },
    ],
    mood: { text: 'After 2012, deep networks were winning benchmarks and "investment and interest in AI boomed", but the winning networks were too big to ship to phones.', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'AlexNet\'s weights take 240 MB.', source: 'https://arxiv.org/abs/1510.00149' },
      this: { text: 'Pruning, quantisation and coding shrink AlexNet 35×, to 6.9 MB, with no loss of accuracy.', source: 'https://arxiv.org/abs/1510.00149' },
      after: { text: '"Winning tickets": subnetworks under 10–20% of the original size that train as well as the whole network on MNIST and CIFAR-10.', source: 'https://arxiv.org/abs/1803.03635' },
    },
  },
  {
    id: 'descent-golf',
    lessonId: 'gradient-descent',
    lab: 'Descent golf',
    idea: 'Gradient descent',
    years: [1847],
    when: '1847',
    who: [{ text: 'Augustin-Louis Cauchy "first suggested it in 1847".', source: WP('Gradient_descent') }],
    story: {
      before: { text: 'Minimising a function of many variables, for which an exact algebraic solution is rarely available.', source: WP('Gradient_descent') },
      this: { text: 'A rule for walking downhill: take repeated steps against the gradient.', source: WP('Gradient_descent') },
      after: { text: 'The same rule, nearly 180 years later, behind every training run in these labs.', source: WP('Stochastic_gradient_descent') },
    },
  },
  {
    id: 'optimizer-golf',
    lessonId: 'optimizers',
    lab: 'Optimizer golf',
    idea: 'Adaptive optimizers and Adam',
    years: [1960, 2014],
    when: '1960 → 2014',
    who: [
      { text: 'Howard Rosenbrock introduces the banana-shaped test function in 1960.', source: WP('Rosenbrock_function') },
      { text: 'Diederik Kingma and Jimmy Ba, Adam (ICLR 2015).', source: 'https://arxiv.org/abs/1412.6980' },
    ],
    mood: { text: 'After AlexNet\'s 2012 win, "investment and interest in AI boomed".', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'A test valley where finding the valley is easy but converging to the minimum is hard.', source: WP('Rosenbrock_function') },
      this: { text: 'Adam: momentum plus a per-parameter step size from a running average of squared gradients.', source: 'https://arxiv.org/abs/1412.6980' },
      after: { text: 'Adam becomes the default baseline: the LoRA paper, for example, trains with it.', source: 'https://arxiv.org/abs/2106.09685' },
    },
  },
  {
    id: 'schedule',
    lessonId: 'lr-schedules',
    lab: 'Design a schedule',
    idea: 'Learning-rate schedules and warm restarts',
    years: [2016],
    when: '2016',
    who: [{ text: 'Ilya Loshchilov and Frank Hutter (University of Freiburg), SGDR.', source: 'https://arxiv.org/abs/1608.03983' }],
    mood: { text: 'The post-2012 boom: "investment and interest in AI boomed".', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'A fixed step size, or one cut by hand at a few chosen moments.', source: 'https://arxiv.org/abs/1608.03983' },
      this: { text: 'Cosine-shaped schedules, with warm restarts, that change the step size smoothly over training.', source: 'https://arxiv.org/abs/1608.03983' },
      after: { text: 'Warm-up then cosine decay becomes a standard recipe, built into training libraries.', source: 'https://huggingface.co/docs/transformers/main_classes/optimizer_schedules' },
    },
  },
  {
    id: 'training-lab',
    lessonId: 'training-lab',
    lab: 'Training-lab challenges',
    idea: 'Universal approximation',
    years: [1988, 1991],
    when: '1988 → 1991',
    who: [{ text: 'George Cybenko gives a rigorous proof for sigmoid networks (March 1988); Kurt Hornik, Maxwell Stinchcombe and Halbert White extend it.', source: WP('Universal_approximation_theorem') }],
    mood: { text: 'In 1987 the LISP-machine market collapsed, and the second AI winter ran to about 2000.', source: WP('AI_winter') },
    story: {
      before: { text: 'No guarantee that a network of simple units could represent an arbitrary function at all.', source: WP('Universal_approximation_theorem') },
      this: { text: 'Proof that one hidden layer with enough neurons can approximate any continuous function as closely as you like.', source: WP('Universal_approximation_theorem') },
      after: { text: 'The open question becomes whether training finds that network, which is exactly the challenges\' lesson: representable is not the same as learnable.', source: WP('Universal_approximation_theorem') },
    },
  },
  {
    id: 'backward-pass',
    lessonId: 'network-from-scratch',
    lab: 'Write the backward pass',
    idea: 'Backpropagation',
    years: [1986],
    when: '1986',
    who: [
      { text: 'David Rumelhart (UC San Diego), Geoffrey Hinton (Carnegie Mellon) and Ronald Williams, "Learning representations by back-propagating errors", Nature.', source: WP('David_Rumelhart') },
      { text: 'Hinton was part of the "Parallel Distributed Processing" group at Carnegie Mellon in the 1980s.', source: WP('Geoffrey_Hinton') },
    ],
    mood: { text: 'The neural-network winter "came to an end in the middle 1980s", when this work revived large-scale interest.', source: WP('AI_winter') },
    story: {
      before: { text: 'No practical way to train the hidden layers of a multilayer network.', source: WP('AI_winter') },
      this: { text: 'Backpropagation: the chain rule run backwards through the network, giving every weight its gradient in one pass.', source: WP('Backpropagation') },
      after: { text: 'Every framework\'s backward(), and every network trained since.', source: WP('Backpropagation') },
    },
    dispute: { text: 'Credit is genuinely shared: Seppo Linnainmaa published reverse-mode automatic differentiation in 1976, Paul Werbos applied it to multilayer networks in the now-standard way in 1982, and others found it independently.', source: WP('Backpropagation') },
  },
  {
    id: 'rescue',
    lessonId: 'vanishing-gradients',
    lab: 'Rescue the deep network',
    idea: 'Vanishing gradients, initialisation and ResNets',
    years: [1991, 2010, 2015],
    when: '1991 → 2010 → 2015',
    who: [
      { text: 'Sepp Hochreiter\'s 1991 diploma thesis at the Technical University of Munich formally identifies the vanishing gradient problem.', source: WP('Vanishing_gradient_problem') },
      { text: 'Xavier Glorot and Yoshua Bengio (Université de Montréal) analyse initialisation (2010).', source: 'https://proceedings.mlr.press/v9/glorot10a.html' },
      { text: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (Microsoft Research), ResNet (2015).', source: 'https://arxiv.org/abs/1512.03385' },
    ],
    mood: { text: 'Through the 1990s, many researchers and engineers "avoided the term artificial intelligence for fear of being viewed as wild-eyed dreamers".', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'Deep networks freeze because gradients shrink layer by layer.', source: WP('Vanishing_gradient_problem') },
      this: { text: 'Variance-preserving initialisation (Glorot 2010, He 2015) and residual connections.', source: 'https://arxiv.org/abs/1502.01852' },
      after: { text: 'ResNet trains 152 layers and wins ILSVRC 2015 at 3.57% error; LayerNorm (2016) follows.', source: 'https://arxiv.org/abs/1512.03385' },
    },
  },
  {
    id: 'clinic',
    lessonId: 'diagnosis-lab',
    lab: 'The model clinic',
    idea: 'A written recipe for debugging training',
    years: [2019],
    when: '2019',
    who: [{ text: 'Andrej Karpathy, "A Recipe for Training Neural Networks".', source: 'http://karpathy.github.io/2019/04/25/recipe/' }],
    mood: { text: 'Training had become an engineering craft, and the post\'s warning was that "neural net training fails silently".', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
    story: {
      before: { text: 'Debugging know-how passed between colleagues as folklore.', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
      this: { text: 'A step-by-step diagnostic recipe: look at the data, start simple, overfit one batch, then regularise.', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
      after: { text: 'The same checklist behind this clinic\'s ten cases.', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
    },
  },
  {
    id: 'attention-puzzles',
    lessonId: 'attention',
    lab: 'Attention puzzles',
    idea: 'Attention and the transformer',
    years: [2014, 2017],
    when: '2014 → 2017',
    who: [
      { text: 'Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio (Université de Montréal).', source: 'https://arxiv.org/abs/1409.0473' },
      { text: 'Vaswani and colleagues, "eight scientists and engineers working at Google", Attention Is All You Need.', source: WP('Attention_Is_All_You_Need') },
    ],
    mood: { text: 'In 2017, researchers at Google Brain introduced the transformer architecture.', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'Squeezing a whole sentence into one fixed-length vector, which Bahdanau et al. conjectured "is a bottleneck".', source: 'https://arxiv.org/abs/1409.0473' },
      this: { text: 'A soft search over the source words: attention.', source: 'https://arxiv.org/abs/1409.0473' },
      after: { text: 'The transformer, "dispensing with recurrence and convolutions entirely" (28.4 BLEU on English→German).', source: 'https://arxiv.org/abs/1706.03762' },
    },
  },
  {
    id: 'previous-word',
    lessonId: 'positions',
    lab: 'Build a previous-word head',
    idea: 'Rotary positions and induction heads',
    years: [2021, 2022],
    when: '2021 → 2022',
    who: [
      { text: 'Jianlin Su and colleagues, RoFormer (rotary position embedding).', source: 'https://arxiv.org/abs/2104.09864' },
      { text: 'Catherine Olsson and colleagues at Anthropic, In-context Learning and Induction Heads.', source: 'https://arxiv.org/abs/2209.11895' },
    ],
    story: {
      before: { text: 'Position added to each word as a fixed vector.', source: 'https://arxiv.org/abs/2104.09864' },
      this: { text: 'Rotary embeddings encode "absolute position with a rotation matrix" with "explicit relative position dependency".', source: 'https://arxiv.org/abs/2104.09864' },
      after: { text: 'Induction heads, built on previous-token heads, "might constitute the mechanism for the majority of all in-context learning" (the authors call their evidence preliminary).', source: 'https://arxiv.org/abs/2209.11895' },
    },
  },
  {
    id: 'compress',
    lessonId: 'tokenization',
    lab: 'Compress the corpus',
    idea: 'Byte-pair encoding',
    years: [1994, 2015],
    when: '1994 → 2015',
    who: [
      { text: 'Philip Gage describes byte-pair encoding as a data-compression method (1994).', source: WP('Byte-pair_encoding') },
      { text: 'Rico Sennrich, Barry Haddow and Alexandra Birch (University of Edinburgh) apply it to translation (ACL 2016).', source: 'https://arxiv.org/abs/1508.07909' },
    ],
    story: {
      before: { text: 'A file-compression trick: repeatedly replace the most common pair of bytes with a new symbol.', source: WP('Byte-pair_encoding') },
      this: { text: 'Subword units for "open-vocabulary" neural translation.', source: 'https://arxiv.org/abs/1508.07909' },
      after: { text: 'BPE tokenizers in GPT-2 and its successors.', source: WP('Byte-pair_encoding') },
    },
  },
  {
    id: 'shannon',
    lessonId: 'next-token-prediction',
    lab: "Shannon's betting game",
    idea: 'Information theory and the entropy of English',
    years: [1948, 1951],
    when: '1948 → 1951',
    who: [{ text: 'Claude Shannon, Bell Labs.', source: WP('Claude_Shannon') }],
    mood: { text: 'Shannon went on to be one of the four organisers of the 1956 Dartmouth workshop that named the field "artificial intelligence".', source: WP('Dartmouth_workshop') },
    story: {
      before: { text: 'No way to measure how much information a message carries.', source: WP('A_Mathematical_Theory_of_Communication') },
      this: { text: 'Information theory (1948), then "Prediction and Entropy of Printed English" (1951): people guess the next letter, and long-range context brings English down to "the order of one bit per letter".', source: 'https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf' },
      after: { text: 'Next-token prediction, scored in bits, is how every large language model is trained.', source: WP('Large_language_model') },
    },
  },
  {
    id: 'sampler',
    lessonId: 'sampling',
    lab: 'Tune the sampler',
    idea: 'Nucleus (top-p) sampling',
    years: [2019],
    when: '2019',
    who: [{ text: 'Ari Holtzman and colleagues (University of Washington and the Allen Institute for AI), The Curious Case of Neural Text Degeneration.', source: 'https://arxiv.org/abs/1904.09751' }],
    story: {
      before: { text: 'Decoding by likelihood "leads to text that is bland and strangely repetitive".', source: 'https://arxiv.org/abs/1904.09751' },
      this: { text: 'Nucleus sampling: sample only from the smallest set of tokens holding most of the probability.', source: 'https://arxiv.org/abs/1904.09751' },
      after: { text: 'top_p becomes a standard parameter of language-model APIs and libraries.', source: 'https://huggingface.co/docs/transformers/main_classes/text_generation' },
    },
  },
  {
    id: 'compute-budget',
    lessonId: 'scaling-laws',
    lab: 'Spend the compute budget',
    idea: 'Scaling laws and Chinchilla',
    years: [2020, 2022, 2024],
    when: '2020 → 2022 → 2024',
    who: [
      { text: 'Jared Kaplan and colleagues (OpenAI and Johns Hopkins), Scaling Laws for Neural Language Models.', source: 'https://arxiv.org/abs/2001.08361' },
      { text: 'Jordan Hoffmann and colleagues at DeepMind, Chinchilla.', source: 'https://arxiv.org/abs/2203.15556' },
      { text: 'Meta AI, Llama 3 (April 2024).', source: WP('Llama_(language_model)') },
    ],
    mood: { text: 'A race to size: Chinchilla\'s comparisons included models of up to 530B parameters.', source: 'https://arxiv.org/abs/2203.15556' },
    story: {
      before: { text: 'Loss follows power laws in model size, data and compute, some spanning "more than seven orders of magnitude".', source: 'https://arxiv.org/abs/2001.08361' },
      this: { text: 'Chinchilla (70B parameters, 1.4T tokens) outperforms Gopher (280B) and GPT-3 (175B) with the same compute.', source: 'https://arxiv.org/abs/2203.15556' },
      after: { text: 'Llama 3 trains small models on about 15 trillion tokens, far past compute-optimal, to make them cheap to serve.', source: WP('Llama_(language_model)') },
    },
  },
  {
    id: 'lora',
    lessonId: 'lora',
    lab: 'LoRA rank budget',
    idea: 'Adapters, LoRA and QLoRA',
    years: [2019, 2021, 2023],
    when: '2019 → 2021 → 2023',
    who: [
      { text: 'Neil Houlsby and colleagues, Parameter-Efficient Transfer Learning for NLP (adapters).', source: 'https://arxiv.org/abs/1902.00751' },
      { text: 'Edward Hu and colleagues (Microsoft), LoRA.', source: 'https://arxiv.org/abs/2106.09685' },
      { text: 'Tim Dettmers and colleagues, QLoRA.', source: 'https://arxiv.org/abs/2305.14314' },
    ],
    mood: { text: 'GPT-3 had 175B parameters, and deploying a separately fine-tuned copy for every task was, in the LoRA authors\' words, "prohibitively expensive".', source: 'https://arxiv.org/abs/2106.09685' },
    story: {
      before: { text: 'Adapters reach within 0.4% of full fine-tuning while adding 3.6% parameters per task.', source: 'https://arxiv.org/abs/1902.00751' },
      this: { text: 'LoRA cuts GPT-3\'s trainable parameters 10,000× and GPU memory 3×.', source: 'https://arxiv.org/abs/2106.09685' },
      after: { text: 'QLoRA fine-tunes a 65B model on a single 48 GB GPU.', source: 'https://arxiv.org/abs/2305.14314' },
    },
  },
  {
    id: 'alignment',
    lessonId: 'rlhf-dpo',
    lab: 'Tune the alignment pipeline',
    idea: 'RLHF and DPO',
    years: [2017, 2022, 2023],
    when: '2017 → 2022 → 2023',
    who: [
      { text: 'Paul Christiano and colleagues, Deep Reinforcement Learning from Human Preferences.', source: 'https://arxiv.org/abs/1706.03741' },
      { text: 'Long Ouyang and colleagues (OpenAI), InstructGPT.', source: 'https://arxiv.org/abs/2203.02155' },
      { text: 'Rafael Rafailov and colleagues (Stanford), Direct Preference Optimization.', source: 'https://arxiv.org/abs/2305.18290' },
    ],
    mood: { text: 'Goodhart\'s law, in Marilyn Strathern\'s 1997 words: "When a measure becomes a target, it ceases to be a good measure."', source: WP("Goodhart's_law") },
    story: {
      before: { text: 'Learning from human preferences between pairs, with feedback on less than 1% of the agent\'s interactions.', source: 'https://arxiv.org/abs/1706.03741' },
      this: { text: 'People prefer the 1.3B-parameter InstructGPT to the 175B GPT-3.', source: 'https://arxiv.org/abs/2203.02155' },
      after: { text: 'DPO replaces "complex and often unstable" RLHF with "a simple classification loss".', source: 'https://arxiv.org/abs/2305.18290' },
    },
  },
  {
    id: 'quant',
    lessonId: 'quantization',
    lab: 'Mixed-precision quantisation',
    idea: 'LLM quantisation: LLM.int8(), GPTQ and k-quants',
    years: [2022, 2023],
    when: '2022 → 2023',
    who: [
      { text: 'Tim Dettmers and colleagues, LLM.int8().', source: 'https://arxiv.org/abs/2208.07339' },
      { text: 'Elias Frantar and colleagues, GPTQ.', source: 'https://arxiv.org/abs/2210.17323' },
      { text: 'Iwan Kawrakow, llama.cpp k-quants.', source: 'https://github.com/ggml-org/llama.cpp/pull/1684' },
    ],
    story: {
      before: { text: '8-bit inference for models up to 175B "without any performance degradation", once emergent outlier features are handled.', source: 'https://arxiv.org/abs/2208.07339' },
      this: { text: 'GPTQ quantises 175B models to 3–4 bits in about four GPU hours.', source: 'https://arxiv.org/abs/2210.17323' },
      after: { text: 'k-quants, including Q4_K_M, are merged into llama.cpp on 5 June 2023, bringing mixed-precision files to laptops.', source: 'https://github.com/ggml-org/llama.cpp/pull/1684' },
    },
  },
  {
    id: 'vram',
    lessonId: 'kv-cache',
    lab: 'VRAM Tetris',
    idea: 'Shared KV heads and PagedAttention',
    years: [2019, 2023],
    when: '2019 → 2023',
    who: [
      { text: 'Noam Shazeer, multi-query attention.', source: 'https://arxiv.org/abs/1911.02150' },
      { text: 'Joshua Ainslie and colleagues, grouped-query attention (GQA).', source: 'https://arxiv.org/abs/2305.13245' },
      { text: 'Woosuk Kwon and colleagues (UC Berkeley, with Stanford and UC San Diego), vLLM and PagedAttention.', source: 'https://arxiv.org/abs/2309.06180' },
    ],
    story: {
      before: { text: 'Sharing keys and values across heads cuts the "memory bandwidth requirements of incremental decoding".', source: 'https://arxiv.org/abs/1911.02150' },
      this: { text: 'GQA keeps quality "close to multi-head" attention at speed comparable to multi-query.', source: 'https://arxiv.org/abs/2305.13245' },
      after: { text: 'PagedAttention gives 2–4× the throughput of FasterTransformer and Orca.', source: 'https://arxiv.org/abs/2309.06180' },
    },
  },
  {
    id: 'wire-policy',
    lessonId: 'mdps',
    lab: 'Wire the Policy',
    idea: 'Dynamic programming and the Bellman equation',
    years: [1953, 1957],
    when: '1953 → 1957',
    who: [{ text: 'Richard Bellman, at the RAND Corporation from 1949.', source: WP('Richard_Bellman') }],
    story: {
      before: { text: 'No general method for planning a sequence of decisions under uncertainty.', source: WP('Bellman_equation') },
      this: { text: 'Dynamic programming (1953), then the book Dynamic Programming (1957) and the Bellman equation.', source: WP('Bellman_equation') },
      after: { text: 'The value backups at the heart of every reinforcement-learning method.', source: WP('Bellman_equation') },
    },
  },
  {
    id: 'q-learner',
    lessonId: 'q-learning',
    lab: 'Tune the Q-Learner',
    idea: 'TD learning, Q-learning and DQN',
    years: [1988, 1989, 2013, 2015],
    when: '1988 → 1989 → 2013/2015',
    who: [
      { text: 'Richard Sutton, temporal-difference learning (1988).', source: WP('Temporal_difference_learning') },
      { text: 'Chris Watkins, Q-learning (1989; convergence proof with Peter Dayan, 1992).', source: WP('Q-learning') },
      { text: 'Volodymyr Mnih and colleagues at DeepMind, London.', source: 'https://arxiv.org/abs/1312.5602' },
    ],
    mood: { text: '1989 sits inside the second AI winter (about 1987–2000).', source: WP('AI_winter') },
    story: {
      before: { text: 'Temporal-difference learning updates a guess from the next guess, without waiting for the final outcome.', source: WP('Temporal_difference_learning') },
      this: { text: 'Q-learning: learn the value of each action from experience, without a model of the world.', source: WP('Q-learning') },
      after: { text: 'Deep Q-networks learn Atari games from raw pixels (2013); by 2015 they reach "a level comparable to that of a professional human games tester across a set of 49 games".', source: 'https://www.nature.com/articles/nature14236' },
    },
  },
  {
    id: 'policy-gradient',
    lessonId: 'policy-gradients',
    lab: 'Stabilise the policy gradient',
    idea: 'REINFORCE and PPO',
    years: [1992, 2017],
    when: '1992 → 2017',
    who: [
      { text: 'Ronald J. Williams, REINFORCE.', source: WP('Policy_gradient_method') },
      { text: 'John Schulman and colleagues (OpenAI), Proximal Policy Optimization.', source: 'https://arxiv.org/abs/1707.06347' },
    ],
    mood: { text: 'Inside the second AI winter (about 1987–2000).', source: WP('AI_winter') },
    story: {
      before: { text: 'REINFORCE, "the first policy gradient method", whose estimates are very noisy.', source: WP('Policy_gradient_method') },
      this: { text: 'Baselines that cut the variance without biasing the gradient.', source: WP('Policy_gradient_method') },
      after: { text: 'PPO\'s clipped objective "enables multiple epochs of minibatch updates", and became the workhorse of RLHF.', source: 'https://arxiv.org/abs/1707.06347' },
    },
  },
  {
    id: 'reward-hacking',
    lessonId: 'reward-hacking',
    lab: 'Reward-Hacking Sandbox',
    idea: 'Reward shaping and specification gaming',
    years: [1975, 1999, 2016],
    when: '1975 · 1999 · 2016',
    who: [
      { text: 'Charles Goodhart, on UK monetary policy (1975).', source: WP("Goodhart's_law") },
      { text: 'Andrew Ng, Daishi Harada and Stuart Russell (UC Berkeley), potential-based reward shaping.', source: 'https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf' },
      { text: 'OpenAI, "Faulty reward functions in the wild" (21 December 2016).', source: 'https://openai.com/index/faulty-reward-functions/' },
    ],
    mood: { text: 'Goodhart\'s law began as a 1975 point about UK monetary policy, long before anyone trained an agent.', source: WP("Goodhart's_law") },
    story: {
      before: { text: 'Hand-added shaping rewards, which cause "well-known bugs".', source: 'https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf' },
      this: { text: 'Potential-based shaping provably keeps the optimal policy unchanged.', source: 'https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf' },
      after: { text: 'The CoastRunners boat circles a lagoon knocking over targets and scores 20% higher than human players without finishing the race.', source: 'https://openai.com/index/faulty-reward-functions/' },
    },
  },
  {
    id: 'linear-regression',
    lessonId: 'linear-regression',
    lab: 'Feature Engineering + Logistic Regression',
    idea: 'Least squares, logistic regression and Ridge',
    years: [1805, 1944, 1958, 1970],
    when: '1805 · 1944 · 1958 · 1970',
    who: [
      { text: 'Adrien-Marie Legendre publishes the method of least squares in Paris (1805); Carl Friedrich Gauss publishes it in 1809 after using it at age 24 to relocate the lost asteroid Ceres in 1801.', source: WP('Least_squares') },
      { text: 'Joseph Berkson coins "logit" in 1944 and champions the logistic model; David Cox refines binary logistic regression in 1958.', source: WP('Logistic_regression') },
      { text: 'Arthur Hoerl and Robert Kennard introduce Ridge regression in Technometrics (1970) to stabilise collinear least squares.', source: WP('Ridge_regression') },
    ],
    mood: { text: 'Berkson\'s logit model "was initially dismissed as inferior to the probit model", only reaching equal footing in statistics journals between 1960 and 1970.', source: WP('Logistic_regression') },
    story: {
      before: { text: 'Least squares fits continuous curves (such as planetary orbits) in one linear solve, but breaks on binary outcomes and collinear features.', source: WP('Least_squares') },
      this: { text: 'Logistic regression models log-odds linearly through a sigmoid, while Ridge shifts the diagonal of XᵀX by λI to tame near-singular matrices.', source: WP('Ridge_regression') },
      after: { text: 'Paired with engineered features, regularised logistic regression became the fast, convex baseline across medicine, credit scoring and large-scale ad prediction.', source: WP('Logistic_regression') },
    },
    dispute: { text: 'Legendre published least squares first in 1805, while Gauss claimed in 1809 to have used it since 1795 — sparking a famous priority dispute (though Gauss went further by tying it to the normal distribution).', source: WP('Least_squares') },
  },
  {
    id: 'knn',
    lessonId: 'knn',
    lab: 'Rescue the kNN Classifier',
    idea: 'Nearest neighbours and the curse of dimensionality',
    years: [1951, 1957, 1967],
    when: '1951 · 1957 · 1967',
    who: [
      { text: 'Evelyn Fix and Joseph Hodges develop non-parametric nearest-neighbour discrimination in a 1951 report for the USAF School of Aviation Medicine in Texas.', source: WP('K-nearest_neighbors_algorithm') },
      { text: 'Richard Bellman coins "the curse of dimensionality" in Dynamic Programming (1957, RAND Corporation).', source: WP('Curse_of_dimensionality') },
      { text: 'Thomas Cover and Peter Hart publish "Nearest neighbor pattern classification" in IEEE Transactions on Information Theory (1967).', source: WP('K-nearest_neighbors_algorithm') },
    ],
    story: {
      before: { text: 'Classical discriminant analysis assumed each class followed a clean parametric distribution such as a Gaussian.', source: WP('K-nearest_neighbors_algorithm') },
      this: { text: 'Fix and Hodges classify by majority vote of the k closest stored examples; Cover and Hart prove that even 1-NN asymptotically has an error rate no worse than twice the optimal Bayes error rate.', source: WP('K-nearest_neighbors_algorithm') },
      after: { text: 'Raw-feature kNN hits Bellman\'s curse of dimensionality when noisy columns accumulate, so modern systems run approximate nearest-neighbour search over learned neural embeddings instead.', source: WP('Curse_of_dimensionality') },
    },
  },
  {
    id: 'decision-trees',
    lessonId: 'decision-trees',
    lab: 'Build the Tabular Ensemble',
    idea: 'CART, Random Forests and Gradient Boosting',
    years: [1984, 1995, 2001, 2016],
    when: '1984 · 1995 · 2001 · 2016',
    who: [
      { text: 'Leo Breiman and colleagues introduce Classification and Regression Trees (CART) and Gini splitting in 1984.', source: WP('Decision_tree_learning') },
      { text: 'Tin Kam Ho introduces random decision forests in 1995; Leo Breiman publishes "Random Forests" (combining bagging and random feature subsets) in 2001.', source: WP('Random_forest') },
      { text: 'Jerome Friedman develops explicit gradient boosting machines (1999/2001); Tianqi Chen and Carlos Guestrin release XGBoost (2016).', source: 'https://arxiv.org/abs/1603.02754' },
    ],
    story: {
      before: { text: 'A single decision tree splits tabular data cleanly without scaling, but deep single trees overfit noisy labels.', source: WP('Decision_tree_learning') },
      this: { text: 'Bagging + random feature splits (Random Forests) slash variance without overfitting as more trees are added, while gradient boosting fits trees sequentially to residuals in function space.', source: WP('Gradient_boosting') },
      after: { text: 'Scalable Newton-leaf tree boosters like XGBoost (2016) scale to billions of rows and remain the state of the art on structured tabular data.', source: 'https://arxiv.org/abs/1603.02754' },
    },
  },
  {
    id: 'clustering-pca',
    lessonId: 'clustering-pca',
    lab: 'Compress & Segment',
    idea: 'PCA, Lloyd\'s k-Means and k-Means++',
    years: [1901, 1957, 1967, 2007],
    when: '1901 · 1957 · 1967 · 2007',
    who: [
      { text: 'Karl Pearson invents principal component analysis in 1901; Harold Hotelling independently develops and names it in 1933.', source: WP('Principal_component_analysis') },
      { text: 'Stuart Lloyd proposes the standard iterative clustering algorithm at Bell Labs in 1957 for pulse-code modulation (published in a journal in 1982); James MacQueen coins the term "k-means" in 1967.', source: WP('K-means_clustering') },
      { text: 'David Arthur and Sergei Vassilvitskii introduce k-means++ seeding in 2007 to avoid bad random initialisations.', source: WP('K-means%2B%2B') },
    ],
    story: {
      before: { text: 'High-dimensional unlabelled data had no labels to steer feature selection, and naive random k-means starts frequently got stuck in poor local minima.', source: WP('K-means_clustering') },
      this: { text: 'PCA finds the orthogonal axes of maximum variance (equivalently minimum squared reconstruction error), while k-means++ spreads initial centroids proportionally to squared distance.', source: WP('K-means%2B%2B') },
      after: { text: 'Together they anchor modern unsupervised pipelines — from denoising genomics tables to building Voronoi IVF cells in billion-scale vector databases.', source: WP('Principal_component_analysis') },
    },
  },
  {
    id: 'images-as-tensors',
    lessonId: 'images-as-tensors',
    lab: 'Build the Vision Preprocessor',
    idea: 'Digital pixels, sampling and colour spaces',
    years: [1949, 1957, 1978, 1993],
    when: '1949 · 1957 · 1978 · 1993',
    who: [
      { text: 'Claude Shannon publishes the sampling theorem in Proceedings of the IRE (1949), formalising the Nyquist rate that prevents aliasing.', source: WP('Nyquist%E2%80%93Shannon_sampling_theorem') },
      { text: 'Russell Kirsch and his team at the National Bureau of Standards scan the first digital photograph in 1957 — a 176×176, 1-bit image of his infant son on the SEAC computer.', source: WP('Russell_Kirsch') },
      { text: 'Alvy Ray Smith formally describes the HSV colour model at NYIT in 1978; ITU-R publishes the Rec. 709 HDTV standard (1993).', source: WP('HSL_and_HSV') },
    ],
    story: {
      before: { text: 'Computers had no way to ingest photographs at all; signals were continuous film or analogue voltage.', source: WP('Russell_Kirsch') },
      this: { text: 'Kirsch turned a photograph into an array of numbers (pixels), Shannon\'s sampling theorem explained why downsampling without filtering creates aliasing, and HSV separated hue from brightness.', source: WP('Nyquist%E2%80%93Shannon_sampling_theorem') },
      after: { text: 'Modern vision frameworks pack RGB frames into standardised FP16/FP32 NCHW tensors — still obeying the same sampling and colour-space rules.', source: WP('Rec._709') },
    },
  },
  {
    id: 'convolution',
    lessonId: 'convolution',
    lab: 'Design the Kernel',
    idea: 'Edge-detecting kernels and the Neocognitron',
    years: [1959, 1968, 1979],
    when: '1959 · 1968 · 1979',
    who: [
      { text: 'David Hubel and Torsten Wiesel discover orientation-selective simple and complex cells in mammalian primary visual cortex (1959).', source: WP('Neocognitron') },
      { text: 'Irwin Sobel and Gary Feldman present the 3×3 isotropic edge-gradient kernel at the Stanford Artificial Intelligence Laboratory (SAIL) in 1968.', source: WP('Sobel_operator') },
      { text: 'Kunihiko Fukushima proposes the Neocognitron (1979/1980), a hierarchical multilayered network that learns shift-invariant convolutional filters.', source: WP('Neocognitron') },
    ],
    story: {
      before: { text: 'Early computer vision relied on hand-crafted integer kernels like Sobel\'s 3×3 filters to extract vertical and horizontal edges.', source: WP('Sobel_operator') },
      this: { text: 'Fukushima stacked local receptive fields with weight sharing inspired by Hubel and Wiesel, letting a network learn spatial filters automatically.', source: WP('Neocognitron') },
      after: { text: 'Combining learned convolutional stacks with backpropagation produced modern CNNs — whose first layers still spontaneously rediscover oriented edge filters.', source: WP('Neocognitron') },
    },
  },
  {
    id: 'cnns',
    lessonId: 'cnns',
    lab: 'Architect the CNN',
    idea: 'LeNet, AlexNet, stacked 3×3s, and Global Average Pooling',
    years: [1989, 2012, 2013, 2014],
    when: '1989 · 2012 · 2013 · 2014',
    who: [
      { text: 'Yann LeCun and colleagues at AT&T Bell Labs train LeNet-1 with backpropagation (1989) and deploy LeNet-5 in cheque-reading systems (1998).', source: WP('LeNet') },
      { text: 'Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton at the University of Toronto win ILSVRC 2012 with AlexNet (15.3% top-5 error, 60M parameters across two 3 GB GTX 580 GPUs).', source: WP('AlexNet') },
      { text: 'Min Lin, Qiang Chen, and Shuicheng Yan introduce global average pooling in Network in Network (2013) to replace fully connected heads.', source: 'https://arxiv.org/abs/1312.4400' },
      { text: 'Oxford\'s Visual Geometry Group (VGGNet, 2014) shows that stacking 3×3 convolutions matches 5×5 and 7×7 receptive fields with far fewer weights.', source: WP('VGGNet') },
    ],
    story: {
      before: { text: 'Early CNNs like AlexNet used large 11×11 first-layer kernels and massive dense classification heads that barely fit across two GPUs.', source: WP('AlexNet') },
      this: { text: 'Stacking small 3×3 filters with 2×2 pooling grew receptive fields cheaply, while global average pooling collapsed spatial maps directly into channel averages before the final linear classifier.', source: WP('VGGNet') },
      after: { text: 'ResNet (2015) combined small 3×3 residual blocks with global average pooling, setting the template for a decade of modern vision backbones.', source: WP('Residual_neural_network') },
    },
  },
  {
    id: 'object-detection',
    lessonId: 'object-detection',
    lab: 'Tune the Object Detector',
    idea: 'From sliding cascades to R-CNN, YOLO, and Soft-NMS',
    years: [2001, 2013, 2015, 2017],
    when: '2001 · 2013 · 2015 · 2017',
    who: [
      { text: 'Paul Viola and Michael Jones introduce boosted cascades of Haar-like features (2001), detecting faces at 15 fps on a 700 MHz Pentium III.', source: WP('Viola%E2%80%93Jones_object_detection_framework') },
      { text: 'Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik introduce R-CNN (2013), pairing selective-search region proposals with CNN features.', source: WP('Region-based_convolutional_neural_network') },
      { text: 'Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi introduce YOLO (2015), predicting boxes and class scores in a single forward pass.', source: WP('You_Only_Look_Once') },
      { text: 'Navaneeth Bodla, Bharat Singh, Rama Chellappa, and Larry S. Davis propose Soft-NMS (2017), decaying overlapping box scores continuously instead of zeroing them.', source: 'https://arxiv.org/abs/1704.04503' },
    ],
    story: {
      before: { text: 'Early detectors slid hand-crafted classifiers across thousands of windows and scales, then merged overlapping hits with hard thresholds.', source: WP('Viola%E2%80%93Jones_object_detection_framework') },
      this: { text: 'Two-stage region networks (R-CNN / Faster R-CNN) and single-shot grid detectors (YOLO) brought deep CNNs to localisation — while Soft-NMS fixed greedy suppression deleting crowded neighbours.', source: 'https://arxiv.org/abs/1704.04503' },
      after: { text: 'Modern real-time detectors pair multi-scale feature pyramids with tuned NMS — or eliminate NMS altogether with end-to-end set matching.', source: WP('You_Only_Look_Once') },
    },
  },
  {
    id: 'one-hot-vs-dense',
    lessonId: 'one-hot-vs-dense',
    lab: 'Choose & Size the Representation',
    idea: 'Distributional semantics, LSA, and neural embedding tables',
    years: [1954, 1957, 1990, 2005],
    when: '1954 · 1957 · 1990 · 2005',
    who: [
      { text: 'Linguist Zellig Harris formulates the distributional hypothesis (1954): words occurring in similar contexts tend to share meaning.', source: WP('Distributional_semantics') },
      { text: 'J. R. Firth popularises the idea in 1957 with the dictum that "a word is characterized by the company it keeps".', source: WP('Distributional_semantics') },
      { text: 'Scott Deerwester, Susan Dumais, George Furnas, Thomas Landauer, and Richard Harshman introduce Latent Semantic Analysis (1990), compressing sparse term-document counts into dense SVD vectors.', source: WP('Distributional_semantics') },
      { text: 'Yoshua Bengio and colleagues (including Morin & Bengio, 2005) establish neural language models that learn dense distributed word vectors jointly via lookup tables.', source: WP('Word_embedding') },
    ],
    story: {
      before: { text: 'Early statistical NLP represented words as orthogonal atomic indices or sparse bag-of-words counts, so seeing "cat" taught nothing about "kitten".', source: WP('Distributional_semantics') },
      this: { text: 'Projecting discrete symbol IDs into a shared dense vector table let linear and neural models share statistical strength across related symbols.', source: WP('Word_embedding') },
      after: { text: 'Embedding tables became the universal front door of deep learning — from Word2Vec and recommender SKUs to every modern Transformer tokeniser.', source: WP('Word_embedding') },
    },
  },
  {
    id: 'word-embeddings',
    lessonId: 'word-embeddings',
    lab: 'Train & Debug Word2Vec',
    idea: 'Word2Vec, negative sampling, GloVe, and subword vectors',
    years: [2013, 2014, 2017],
    when: '2013 · 2014 · 2017',
    who: [
      { text: 'Tomas Mikolov and colleagues at Google release Word2Vec (2013), pairing Skip-Gram and CBOW with negative sampling to train dense word vectors on billions of tokens.', source: WP('Word_embedding') },
      { text: 'Yoav Goldberg and Omer Levy derive the negative-sampling objective (2014), while Jeffrey Pennington, Richard Socher, and Christopher Manning introduce GloVe at Stanford on global co-occurrence counts.', source: WP('GloVe') },
      { text: 'Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov publish fastText (2017), composing word vectors from character n-grams so rare and unseen words get embeddings too.', source: WP('Word_embedding') },
    ],
    story: {
      before: { text: 'Earlier neural language models paid a full softmax over the entire vocabulary on every word pair, bottlenecking training on large corpora.', source: WP('Word2Vec') },
      this: { text: 'Replacing full softmax with a handful of random negative samples turned training into tiny binary dot-product updates — and linear vector analogies emerged from co-occurrence statistics.', source: 'https://arxiv.org/abs/1402.3722' },
      after: { text: 'Static word vectors eventually gave way to contextual Transformer representations for polysemy, while negative sampling lived on in item recommenders and contrastive learning.', source: WP('Word_embedding') },
    },
  },
  {
    id: 'similarity-search',
    lessonId: 'similarity-search',
    lab: 'Tune the Vector Index',
    idea: 'LSH, Product Quantization, HNSW, and GPU vector search (FAISS)',
    years: [1998, 2011, 2016, 2017],
    when: '1998 · 2011 · 2016 · 2017',
    who: [
      { text: 'Piotr Indyk and Rajeev Motwani introduce Locality-Sensitive Hashing (1998) to break the high-dimensional curse of exact spatial trees.', source: WP('Locality-sensitive_hashing') },
      { text: 'Hervé Jégou, Matthijs Douze, and Cordelia Schmid publish Product Quantization for Nearest Neighbor Search (2011), compressing vectors into short codebook IDs.', source: WP('Vector_quantization') },
      { text: 'Yury Malkov and Dmitry Yashunin propose Hierarchical Navigable Small World (HNSW) graphs (2016), layering proximity graphs for fast greedy vector navigation.', source: WP('Hierarchical_navigable_small_world') },
      { text: 'Jeff Johnson, Matthijs Douze, and Hervé Jégou open-source FAISS (2017), indexing a billion vectors across four GPUs with IVF and product quantization.', source: 'https://arxiv.org/abs/1702.08734' },
    ],
    story: {
      before: { text: 'Exact spatial trees like k-d trees degraded to brute-force scans in high dimensions, making million-scale similarity queries impractical.', source: WP('Hierarchical_navigable_small_world') },
      this: { text: 'Trading a tiny fraction of exact recall via hashing, Voronoi partitioning, graph navigation, and vector quantization brought billion-scale search into milliseconds and commodity RAM.', source: 'https://arxiv.org/abs/1702.08734' },
      after: { text: 'Vector databases built on IVF, HNSW, and scalar/product quantization now sit behind semantic search, recommendation engines, and RAG pipelines.', source: WP('Hierarchical_navigable_small_world') },
    },
  },
  {
    id: 'contrastive-learning',
    lessonId: 'contrastive-learning',
    lab: 'Tune Contrastive Training',
    idea: 'InfoNCE, Momentum Contrast (MoCo), SimCLR, and CLIP',
    years: [2018, 2019, 2020, 2021],
    when: '2018 · 2019 · 2020 · 2021',
    who: [
      { text: 'Aaron van den Oord, Yazhe Li, and Oriol Vinyals introduce Contrastive Predictive Coding (2018), formalising the InfoNCE loss across speech, images, text, and RL.', source: 'https://arxiv.org/abs/1807.03748' },
      { text: 'Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick introduce MoCo (2019), decoupling dictionary size from mini-batch VRAM with a FIFO queue and momentum encoder.', source: 'https://arxiv.org/abs/1911.05722' },
      { text: 'Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton publish SimCLR (2020), matching supervised ResNet-50 on ImageNet with large-batch contrastive learning.', source: 'https://arxiv.org/abs/2002.05709' },
      { text: 'Alec Radford, Jong Wook Kim, Ilya Sutskever, and colleagues at OpenAI release CLIP (2021), training symmetric image–text InfoNCE on 400M web pairs for zero-shot vision.', source: 'https://arxiv.org/abs/2103.00020' },
    ],
    story: {
      before: { text: 'Vision models relied heavily on expensive human-labelled datasets like ImageNet and could only predict a fixed list of pre-chosen categories.', source: 'https://arxiv.org/abs/2103.00020' },
      this: { text: 'Pulling matched views or image–caption pairs together while pushing thousands of negatives apart on the unit hypersphere learned transferable features with zero manual class labels.', source: 'https://arxiv.org/abs/2002.05709' },
      after: { text: 'Contrastive encoders like CLIP became the shared vision–language backbone behind modern multimodal LLMs, diffusion models, and zero-shot semantic search.', source: 'https://arxiv.org/abs/2103.00020' },
    },
  },
  {
    id: 'rnns',
    lessonId: 'rnns',
    lab: 'Tame BPTT',
    idea: 'Simple RNNs, Backpropagation Through Time, Vanishing Gradients & Gradient Clipping',
    years: [1987, 1990, 1991, 1994, 2013],
    when: '1987 · 1990 · 1991 · 1994 · 2013',
    who: [
      { text: 'Tony Robinson and Frank Fallside (Cambridge, 1987) and Paul Werbos (1988) independently derive backpropagation through time by unfolding recurrent cells across steps.', source: WP('Backpropagation_through_time') },
      { text: 'Jeffrey Elman at UC San Diego introduces the Simple Recurrent Network in Finding Structure in Time (1990), showing a shared hidden state can learn sequential phrase structure.', source: WP('Jeffrey_Elman') },
      { text: 'Sepp Hochreiter (1991 diploma thesis, TU Munich) and Yoshua Bengio, Patrice Simard, and Paolo Frasconi (1994) prove that error gradients vanish or explode exponentially across long lags.', source: WP('Vanishing_gradient_problem') },
      { text: 'Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio (arXiv 2012 / ICML 2013) analyse exploding gradient cliffs geometrically and popularise gradient-norm clipping.', source: 'https://arxiv.org/abs/1211.5063' },
    ],
    story: {
      before: { text: 'Feed-forward networks had to bake a fixed window size into their input layer, leaving no shared machinery for variable-length sequences like speech or text.', source: WP('Recurrent_neural_network') },
      this: { text: 'Unfolding one recurrent cell through time let standard backpropagation train sequence memory, while spectral analysis and gradient clipping exposed and contained its exponential cliff dynamics.', source: 'https://arxiv.org/abs/1211.5063' },
      after: { text: 'Those exact mathematical limits of vanilla BPTT motivated gated cells (LSTMs and GRUs), sequence-to-sequence attention, and eventually modern state-space models.', source: WP('Vanishing_gradient_problem') },
    },
  },
  {
    id: 'lstms-grus',
    lessonId: 'lstms-grus',
    lab: 'Wire the Gated Cell',
    idea: 'LSTM Constant Error Carousel, Forget Gate & Gated Recurrent Unit (GRU)',
    years: [1997, 1999, 2014, 2015],
    when: '1997 · 1999 · 2014 · 2015',
    who: [
      { text: 'Sepp Hochreiter and Jürgen Schmidhuber publish Long Short-Term Memory in Neural Computation (1997), introducing the Constant Error Carousel with input and output gates to carry gradients across hundreds of steps.', source: WP('Long_short-term_memory') },
      { text: 'Felix Gers, Jürgen Schmidhuber, and Fred Cummins add the forget gate (ICANN 1999 / Neural Computation 2000), allowing an LSTM cell to reset its own state on continual streams.', source: WP('Long_short-term_memory') },
      { text: 'Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, and Yoshua Bengio introduce the Gated Recurrent Unit (GRU) at EMNLP 2014, merging cell and hidden state with 25% fewer parameters.', source: 'https://arxiv.org/abs/1406.1078' },
      { text: 'Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio (2014) and Klaus Greff et al. (2015) systematically compare gated variants, confirming the forget gate as critical and showing GRUs rival LSTMs.', source: 'https://arxiv.org/abs/1412.3555' },
    ],
    story: {
      before: { text: 'Plain tanh RNNs overwrote their hidden state every step, causing gradients to decay exponentially over lags of more than a handful of tokens.', source: WP('Long_short-term_memory') },
      this: { text: 'Splitting off an additive cell highway guarded by learned sigmoid gates let recurrent networks remember cues across long horizons without exploding or vanishing.', source: WP('Long_short-term_memory') },
      after: { text: 'LSTMs and GRUs powered Google Translate, speech recognition, and game-playing agents throughout the 2010s until attention removed the final sequential bottleneck.', source: WP('Long_short-term_memory') },
    },
  },
  {
    id: 'seq2seq-attention',
    lessonId: 'seq2seq-attention',
    lab: 'Break the Bottleneck',
    idea: 'Seq2Seq Encoder–Decoder, Bahdanau & Luong Cross-Attention, and the Jump to Transformers',
    years: [2014, 2015, 2017],
    when: '2014 · 2015 · 2017',
    who: [
      { text: 'Kyunghyun Cho et al. (June 2014) and Ilya Sutskever, Oriol Vinyals, and Quoc Le (September 2014) introduce RNN/LSTM Encoder–Decoder models for machine translation, compressing a whole sentence into one fixed vector.', source: 'https://arxiv.org/abs/1409.3215' },
      { text: 'Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (September 2014) diagnose the fixed-length vector as a bottleneck and introduce additive soft-alignment attention over all encoder states.', source: 'https://arxiv.org/abs/1409.0473' },
      { text: 'Minh-Thang Luong, Hieu Pham, and Christopher Manning (2015) compare global and local attention and introduce fast dot-product scoring, gaining 5.0 BLEU over non-attentional NMT.', source: 'https://arxiv.org/abs/1508.04025' },
      { text: 'Ashish Vaswani and colleagues at Google (2017) remove the remaining O(L) recurrent loop altogether in Attention Is All You Need, training all positions in parallel with multi-head attention.', source: 'https://arxiv.org/abs/1706.03762' },
    ],
    story: {
      before: { text: 'Early neural translation models compressed an entire source sentence into one final hidden vector, causing accuracy to crumble on long sentences.', source: 'https://arxiv.org/abs/1409.0473' },
      this: { text: 'Letting the decoder attend directly to every encoder hidden state broke the fixed-vector bottleneck and learned crisp cross-lingual word alignments end to end.', source: 'https://arxiv.org/abs/1508.04025' },
      after: { text: 'Once attention proved it could route information in O(1) hops, dropping recurrence completely produced the Transformer architecture behind modern LLMs.', source: 'https://arxiv.org/abs/1706.03762' },
    },
  },
  {
    id: 'autoencoders',
    lessonId: 'autoencoders',
    lab: 'The Face Printer',
    idea: 'Autoencoders, Variational Autoencoders (VAE) & β-VAE',
    years: [1986, 2006, 2013, 2014, 2018],
    when: '1986 · 2006 · 2013 · 2014 · 2018',
    who: [
      { text: 'David Rumelhart, James McClelland, and the PDP group (1986) propose training neural networks in auto-association mode, followed by early speech and image bottlenecks in 1987–1988.', source: WP('Autoencoder') },
      { text: 'Geoffrey Hinton and Ruslan Salakhutdinov publish Reducing the Dimensionality of Data with Neural Networks in Science (2006), showing deep autoencoders beat PCA for nonlinear compression.', source: WP('Autoencoder') },
      { text: 'Diederik Kingma and Max Welling (December 2013) introduce Auto-Encoding Variational Bayes (VAE) and the reparameterisation trick, enabling gradient-based training of continuous latent generative models.', source: 'https://arxiv.org/abs/1312.6114' },
      { text: 'Danilo Rezende, Shakir Mohamed, and Daan Wierstra (January 2014) independently publish stochastic backpropagation for deep latent Gaussian models.', source: 'https://arxiv.org/abs/1401.4082' },
      { text: 'Irina Higgins, Christopher Burgess, and colleagues at DeepMind analyse β-VAE (ICLR 2017 / arXiv 2018), tuning the KL weight β to trade reconstruction fidelity against smooth, disentangled latents.', source: 'https://arxiv.org/abs/1804.03599' },
    ],
    story: {
      before: { text: 'Plain autoencoders could compress images into low-dimensional codes, but left arbitrary gaps between codes so sampling random points produced nonsense.', source: WP('Autoencoder') },
      this: { text: 'Encoding each input as a Gaussian smudge and penalising its KL divergence from a standard normal made the latent space smooth enough to sample brand-new examples directly.', source: 'https://arxiv.org/abs/1312.6114' },
      after: { text: 'VAEs became the first-stage image compressors inside Latent Diffusion and Stable Diffusion, while VQ-VAEs and sparse autoencoders now power multimodal tokenisation and mechanistic interpretability.', source: WP('Autoencoder') },
    },
  },
  {
    id: 'diffusion',
    lessonId: 'diffusion',
    lab: 'Spend the Step Budget',
    idea: 'Diffusion Probabilistic Models (DDPM), DDIM & EDM Step Schedules',
    years: [2015, 2020, 2022],
    when: '2015 · 2020 · 2022',
    who: [
      { text: 'Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli (2015) introduce diffusion generative models inspired by nonequilibrium thermodynamics, destroying structure with Gaussian noise and learning to reverse it.', source: 'https://arxiv.org/abs/1503.03585' },
      { text: 'Jonathan Ho, Ajay Jain, and Pieter Abbeel publish Denoising Diffusion Probabilistic Models (DDPM, June 2020), showing that predicting the added noise with squared error matches GAN image quality.', source: 'https://arxiv.org/abs/2006.11239' },
      { text: 'Jiaming Song, Chenlin Meng, and Stefano Ermon introduce DDIM (October 2020), replacing stochastic reverse steps with deterministic non-Markovian jumps that cut sampling steps 10–50×.', source: 'https://arxiv.org/abs/2010.02502' },
      { text: 'Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine at NVIDIA (EDM, 2022) dissect the diffusion design space and show concentrating sampling steps at lower noise dramatically cuts denoiser calls.', source: 'https://arxiv.org/abs/2206.00364' },
    ],
    story: {
      before: { text: 'One-shot generators forced a single pass to map noise straight to crisp pixels (yielding blurry VAEs or unstable adversarial GANs).', source: 'https://arxiv.org/abs/2006.11239' },
      this: { text: 'Iterative denoising turned generation into simple supervised noise prediction, while deterministic samplers (DDIM) and non-uniform step schedules shrank 1,000-step sampling down to a handful of calls.', source: 'https://arxiv.org/abs/2206.00364' },
      after: { text: 'Diffusion and its straight-line cousin flow matching now drive image, video, audio, and protein-structure generators including Stable Diffusion, DALL·E, Sora, and AlphaFold 3.', source: 'https://arxiv.org/abs/2010.02502' },
    },
  },
  {
    id: 'guidance',
    lessonId: 'guidance',
    lab: 'Tune the Guidance',
    idea: 'Classifier Guidance, Classifier-Free Guidance (CFG) & CFG Rescaling',
    years: [2021, 2022, 2023],
    when: '2021 · 2022 · 2023',
    who: [
      { text: 'Prafulla Dhariwal and Alex Nichol (May 2021) introduce classifier guidance in Diffusion Models Beat GANs on Image Synthesis, steering reverse diffusion with gradients of a noise-aware classifier.', source: 'https://arxiv.org/abs/2105.05233' },
      { text: 'Jonathan Ho and Tim Salimans (NeurIPS 2021 Workshop / arXiv 2022) propose Classifier-Free Diffusion Guidance (CFG), dropping labels during training so one model provides both conditional and unconditional estimates.', source: 'https://arxiv.org/abs/2207.12598' },
      { text: 'Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, and colleagues at OpenAI release GLIDE (December 2021), showing human evaluators strongly prefer CFG over CLIP-guided text-to-image diffusion.', source: 'https://arxiv.org/abs/2112.10741' },
      { text: 'Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang (2023) diagnose exposure and oversaturation artifacts at high CFG scales and introduce CFG rescaling.', source: 'https://arxiv.org/abs/2305.08891' },
    ],
    story: {
      before: { text: 'Plain conditional diffusion struggled to match GAN sample sharpness without training an expensive separate noise-aware classifier at every noise level.', source: 'https://arxiv.org/abs/2105.05233' },
      this: { text: 'Extrapolating from the unconditional prediction past the conditional one gave a single knob — guidance scale w — trading sample diversity for crisp prompt alignment.', source: 'https://arxiv.org/abs/2207.12598' },
      after: { text: 'CFG became the universal steering dial inside Stable Diffusion, Imagen, DALL·E 2, and video generators, with modern distillation baking guided trajectories directly into few-step students.', source: 'https://arxiv.org/abs/2112.10741' },
    },
  },
  {
    id: 'rag',
    lessonId: 'rag',
    lab: 'Tune the Retriever',
    idea: 'Okapi BM25, Dense Passage Retrieval (DPR) & Retrieval-Augmented Generation (RAG)',
    years: [1994, 2020],
    when: '1994 · 2020',
    who: [
      { text: 'Stephen Robertson, Karen Spärck Jones, and the Okapi team at City University London formulate Okapi BM25 (TREC-3, 1994), becoming the gold-standard lexical ranking function.', source: WP('Okapi_BM25') },
      { text: 'Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, and colleagues at Facebook AI Research publish Dense Passage Retrieval (DPR, April 2020), beating BM25 on open-domain QA with dual-encoder embeddings.', source: 'https://arxiv.org/abs/2004.04906' },
      { text: 'Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, and colleagues introduce Retrieval-Augmented Generation (RAG, May 2020), coupling a parametric language model with a non-parametric retrieved document index at inference time.', source: 'https://arxiv.org/abs/2005.11401' },
    ],
    story: {
      before: { text: 'Language models froze their world knowledge inside static weights at training time and hallucinated confidently when asked about private, recent, or rare facts.', source: WP('Retrieval-augmented_generation') },
      this: { text: 'Retrieving relevant source passages at query time and pasting them into the prompt grounded generation in verifiable, instantly updatable documents without retraining.', source: 'https://arxiv.org/abs/2005.11401' },
      after: { text: 'Hybrid lexical-plus-vector RAG pipelines with rerankers now underpin enterprise search, customer support assistants, and codebase Q&A across production AI.', source: WP('Retrieval-augmented_generation') },
    },
  },
  {
    id: 'tool-calling',
    lessonId: 'tool-calling',
    lab: 'Harden the Agent Pipeline',
    idea: 'ReAct, Toolformer, Gorilla & Constrained JSON Decoding',
    years: [2022, 2023],
    when: '2022 · 2023',
    who: [
      { text: 'Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, and colleagues (October 2022) publish ReAct, interleaving language-model reasoning steps with external tool actions and observations.', source: 'https://arxiv.org/abs/2210.03629' },
      { text: 'Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, and colleagues at Meta AI introduce Toolformer (February 2023), showing LLMs can learn in a self-supervised loop when and how to invoke calculators, search, and APIs.', source: 'https://arxiv.org/abs/2302.04761' },
      { text: 'Shishir Patil, Tianjun Zhang, Xin Wang, and Joseph Gonzalez at UC Berkeley publish Gorilla (May 2023), showing retrieval-aware training dramatically reduces hallucinated API calls.', source: 'https://arxiv.org/abs/2305.15334' },
      { text: 'Brandon Willard and Rémi Louf (July 2023) introduce finite-state-machine guided generation (Outlines), guaranteeing syntactically valid JSON schemas with negligible decoding overhead.', source: 'https://arxiv.org/abs/2307.09702' },
    ],
    story: {
      before: { text: 'Early chat models could only emit free-form prose and failed at arithmetic, live data lookup, and structured API execution.', source: 'https://arxiv.org/abs/2210.03629' },
      this: { text: 'Turning tool calls into a host-verified loop with structured JSON outputs, deterministic guards, and selective retries turned brittle text generators into reliable multi-step workflows.', source: 'https://arxiv.org/abs/2307.09702' },
      after: { text: 'Model-agnostic tool protocols (such as MCP), sandboxed code execution, and compiler/test-backed retry loops now power modern coding and enterprise agents.', source: 'https://arxiv.org/abs/2305.15334' },
    },
  },
  {
    id: 'evaluation',
    lessonId: 'evaluation',
    lab: 'Size Your Eval Suite',
    idea: 'McNemar’s Paired Test, MMLU & LLM-as-a-Judge',
    years: [1947, 2020, 2023],
    when: '1947 · 2020 · 2023',
    who: [
      {
        text: 'Quinn McNemar (1947, Psychometrika) introduces the paired 2×2 test for correlated proportions, comparing two tests on the same subjects purely through their discordant outcomes.',
        source: WP('McNemar%27s_test'),
      },
      {
        text: 'Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (September 2020) publish MMLU (Measuring Massive Multitask Language Understanding) as a broad multitask benchmark.',
        source: 'https://arxiv.org/abs/2009.03300',
      },
      {
        text: 'Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Ion Stoica, and colleagues (June 2023, NeurIPS) analyse LLM-as-a-judge biases and release MT-Bench and Chatbot Arena.',
        source: 'https://arxiv.org/abs/2306.05685',
      },
    ],
    story: {
      before: {
        text: 'Frequentist power analysis quantifies the chance of detecting a real gain as a function of effect size, sample size, test statistic, and significance threshold — yet small or unpaired evaluations routinely miss true improvements or chase noise.',
        source: WP('Power_(statistics)'),
      },
      this: {
        text: 'McNemar’s test drops shared agreements and runs an exact binomial check on the discordant pairs alone; decades later, Zheng et al. documented position, verbosity, and self-enhancement biases in pairwise LLM judges and showed GPT-4 can exceed 80% agreement with human preferences.',
        source: 'https://arxiv.org/abs/2306.05685',
      },
      after: {
        text: 'Production AI teams combine deterministic unit-test benchmarks such as SWE-bench (2023) with position-swapped LLM judges and locked held-out suites calibrated against human labels.',
        source: 'https://arxiv.org/abs/2310.06770',
      },
    },
    dispute: {
      text: 'Automated LLM judges scale far more cheaply than human annotators, yet Zheng et al. showed measurable position, verbosity, and self-enhancement biases and weaker reasoning on hard maths or coding questions — making calibration against exact or human checks essential.',
      source: 'https://arxiv.org/abs/2306.05685',
    },
  },
];

/** Era bands for the timeline: the mood of each period, sourced. */
export type Era = { from: number; to: number; label: string; mood: Claim; kind: 'winter' | 'boom' | 'founding' };

export const ERAS: Era[] = [
  {
    from: 1956,
    to: 1974,
    label: 'The founding years',
    kind: 'founding',
    mood: { text: 'The 1956 Dartmouth workshop names the field "artificial intelligence", and early optimism runs high: Rosenblatt predicted the perceptron "may eventually be able to learn, make decisions, and translate languages".', source: WP('AI_winter') },
  },
  {
    from: 1974,
    to: 1980,
    label: 'First AI winter',
    kind: 'winter',
    mood: { text: 'Funding for neural-network work "was difficult to find in the 1970s and early 1980s".', source: WP('AI_winter') },
  },
  {
    from: 1987,
    to: 2000,
    label: 'Second AI winter',
    kind: 'winter',
    mood: { text: 'After the LISP-machine market collapsed in 1987, many engineers "avoided the term artificial intelligence for fear of being viewed as wild-eyed dreamers".', source: WP('History_of_artificial_intelligence') },
  },
  {
    from: 2012,
    to: 2026,
    label: 'The deep-learning boom',
    kind: 'boom',
    mood: { text: 'After AlexNet\'s 2012 win, "investment and interest in AI boomed".', source: WP('History_of_artificial_intelligence') },
  },
];

/** Field-wide milestones that aren't a lab's own idea but anchor the timeline. */
export const MILESTONES: { year: number; label: string; source: string }[] = [
  { year: 1950, label: 'Turing’s "Computing Machinery and Intelligence", ending with learning machines', source: WP('Computing_Machinery_and_Intelligence') },
  { year: 1956, label: 'Dartmouth workshop names "artificial intelligence"', source: WP('Dartmouth_workshop') },
  { year: 1958, label: 'Press reports hail the perceptron', source: WP('Perceptron') },
  { year: 1969, label: 'Minsky and Papert’s book Perceptrons', source: WP('Perceptrons_(book)') },
  { year: 1973, label: 'The Lighthill report on AI research', source: WP('Lighthill_report') },
  { year: 1986, label: 'Backpropagation paper in Nature', source: WP('Backpropagation') },
  { year: 1997, label: 'Long short-term memory (LSTM)', source: WP('Long_short-term_memory') },
  { year: 2009, label: 'ImageNet first presented', source: WP('ImageNet') },
  { year: 2012, label: 'AlexNet wins ImageNet', source: WP('History_of_artificial_intelligence') },
  { year: 2017, label: 'The transformer: "Attention Is All You Need"', source: WP('Attention_Is_All_You_Need') },
  { year: 2022, label: 'ChatGPT released (30 November)', source: WP('ChatGPT') },
  { year: 2024, label: 'Nobel Prize in Physics to John Hopfield and Geoffrey Hinton', source: 'https://www.nobelprize.org/prizes/physics/2024/summary/' },
];

export function historyById(id: string): HistoryEntry | undefined {
  return HISTORY.find((h) => h.id === id);
}

/** Every claim in an entry, in reading order (who, mood, story, dispute). */
export function claimsOf(h: HistoryEntry): Claim[] {
  return [...h.who, ...(h.mood ? [h.mood] : []), h.story.before, h.story.this, h.story.after, ...(h.dispute ? [h.dispute] : [])];
}

// ------------------------------------------------------------------ Timeline layout (pure, tested)

/** The overview axis: a compressed "earlier" stub for pre-1940 ideas, then 1940 → 2026 linearly. */
export const TIMELINE = { stubWidth: 70, start: 1940, end: 2026, width: 900 } as const;

export function timelineX(year: number): number {
  const { stubWidth, start, end, width } = TIMELINE;
  if (year < start) {
    // pre-1940 ideas (1733 → 1847) share the stub, in order
    const t = (year - 1700) / (start - 1700);
    return 8 + Math.max(0, Math.min(1, t)) * (stubWidth - 20);
  }
  return stubWidth + ((year - start) / (end - start)) * (width - stubWidth - 12);
}

/**
 * Stack labelled dots into lanes so no two in a lane are closer than `gap`
 * pixels: the first lane with room wins (greedy, in x order).
 */
export function assignLanes(xs: number[], gap: number): number[] {
  const order = xs.map((x, i) => ({ x, i })).sort((a, b) => a.x - b.x);
  const laneEnd: number[] = [];
  const lanes = new Array<number>(xs.length);
  for (const { x, i } of order) {
    let lane = laneEnd.findIndex((e) => x - e >= gap);
    if (lane === -1) {
      lane = laneEnd.length;
      laneEnd.push(x);
    } else laneEnd[lane] = x;
    lanes[i] = lane;
  }
  return lanes;
}
