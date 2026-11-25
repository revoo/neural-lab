// authored-by: Gemini Antigravity

export function ragLabRealCode(size: number, lambda: number, k: number): string {
  const mode =
    lambda === 0
      ? 'BM25 / TF-IDF lexical only'
      : lambda === 1
        ? 'dense vector search only (no lexical)'
        : `hybrid fusion: ${(1 - lambda).toFixed(2)} * lexical + ${lambda.toFixed(2)} * dense`;
  return [
    `# ${mode} · ${size}-sentence chunks · top_k = ${k}`,
    `chunks = split_by_sentences(docs, sentences_per_chunk=${size})`,
    lambda === 0
      ? `hits = bm25_index.search(query, top_k=${k})`
      : lambda === 1
        ? `hits = vector_db.query(embed(query), top_k=${k})`
        : `hits = hybrid_search(query, alpha=${lambda.toFixed(2)}, top_k=${k})  # Weaviate / Qdrant / pgvector`,
  ].join('\n');
}
