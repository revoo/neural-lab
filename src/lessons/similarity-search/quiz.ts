// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'ann.unit-norm-equiv',
    kind: 'choice',
    concepts: ['ann-search', 'cosine-similarity'],
    prompt: 'If every embedding vector in your database and every query vector is L2-normalised to length 1, how do the top-k rankings of Cosine similarity, Dot Product, and Euclidean (L2) distance compare?',
    options: [
      {
        text: 'They produce the exact same ranking, because ‖a − b‖² = 2(1 − a·b) = 2(1 − cos(a, b)) on the unit sphere.',
        correct: true,
        why: 'Normalising vectors once at index time lets vector databases use super-fast raw dot products (MIPS) while getting exact Cosine and Euclidean ordering.',
      },
      {
        text: 'Dot Product still favours longer documents.',
        why: 'Once L2-normalised, every vector has exact length 1.0, so length differences disappear.',
      },
      {
        text: 'Euclidean distance reverses the ranking of Cosine similarity.',
        why: 'Larger cosine gives *smaller* L2 distance — so maximising cosine and minimising L2 pick the exact same nearest neighbours.',
      },
    ],
  },
  {
    id: 'ann.ivf-boundary',
    kind: 'choice',
    concepts: ['ivf-index'],
    prompt: 'In an IVF (Inverted File) index with nlist = 12 Voronoi buckets, why did setting nprobe = 1 miss 4 of the 5 true nearest neighbours for a query near a cell border?',
    options: [
      {
        text: 'Because nprobe = 1 searches only the single nearest centroid’s bucket, missing true neighbours sitting just across the Voronoi fence in adjacent buckets.',
        correct: true,
        why: 'Probing even 2–6 nearest centroids (`nprobe > 1`) checks neighbouring cells and recovers >90% recall while still skipping ~75% of the database.',
      },
      {
        text: 'Because k-Means deleted those 4 vectors during indexing.',
        why: 'Every vector is stored intact in one bucket; `nprobe = 1` simply chose not to open the neighbouring bucket.',
      },
      {
        text: 'Because IVF uses random hyperplanes instead of centroids.',
        why: 'Random hyperplanes are LSH (SimHash); IVF uses k-Means Voronoi centroids.',
      },
    ],
  },
  {
    id: 'ann.lsh-vs-ivf',
    kind: 'choice',
    concepts: ['ivf-index', 'lsh'],
    prompt: 'On our 600 × 16D clustered benchmark, IVF reached 91.3% Recall@5 scanning 24.5% of vectors, while random-hyperplane LSH needed 29.3% scan for 90.7% Recall@5. Why does IVF usually beat LSH on real embeddings?',
    options: [
      {
        text: 'IVF adapts its Voronoi partitions to the actual cluster geometry via k-Means, whereas LSH draws data-oblivious random hyperplanes that frequently slice straight through dense clusters.',
        correct: true,
        why: 'In production vector databases (FAISS, pgvector, Qdrant), data-aware indexes like IVF and HNSW dominate pure LSH for this exact reason.',
      },
      {
        text: 'Because LSH cannot use more than 1 hash table.',
        why: 'Multi-table LSH uses L tables (our best used L = 16), which boosts recall at the cost of checking more unioned buckets.',
      },
      {
        text: 'Because LSH requires O(N²) memory.',
        why: 'LSH memory is O(N·L), linear in N.',
      },
    ],
  },
  {
    id: 'ann.pq-ram',
    kind: 'numeric',
    concepts: ['product-quantization'],
    tolerance: 0.1,
    make: (rng) => {
      const millions = [5, 10, 20][rng.int(3)];
      const bytesPerVec = [64, 1536][rng.int(2)];
      const label = bytesPerVec === 64 ? 'PQ-64 (64 bytes/vector)' : 'Scalar INT8 (1,536 bytes/vector)';
      const gib = Number(((millions * 1_000_000 * bytesPerVec) / 1024 ** 3).toFixed(2));
      return {
        prompt: `How many GiB (1 GiB = 1,073,741,824 bytes) does a database of ${millions} million 1,536D embeddings take when stored as ${label}?`,
        answer: gib,
        explain: `${millions}×10⁶ × ${bytesPerVec} bytes ÷ 1024³ = ${gib.toFixed(2)} GiB.`,
      };
    },
  },
];
