// authored-by: Claude Opus 5.5

/** Size of the lab's toy vocabulary: top-k = 16 keeps every token, i.e. top-k is off. */
export const TOY_VOCAB = 16;

/**
 * The Hugging Face `generate` call for the sampler lab's settings. Transformers
 * applies temperature, then top-k, then top-p, the same order as the lab; it
 * treats top_k=0 and top_p=1.0 as "off".
 */
export function hfGenerateLine(temperature: number, topK: number, topP: number): string {
  const k = topK >= TOY_VOCAB ? '0' : String(topK);
  return `model.generate(**inputs, do_sample=True, temperature=${temperature.toFixed(1)}, top_k=${k}, top_p=${topP.toFixed(2)})`;
}
