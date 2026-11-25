// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { sklearnUnsupLines } from './realCode';

describe('clustering-pca realCode', () => {
  const base = { b1Scaled: true, b1M: 2, b2Scaled: true, b2K: 4, b2Init: 'kmeans++' as const, b3M: 2, b3K: 4 };
  it('emits StandardScaler + PCA, StandardScaler + KMeans, and PCA -> KMeans pipelines', () => {
    expect(sklearnUnsupLines('pca-scale', base)).toContain('make_pipeline(StandardScaler(), PCA(n_components=2))');
    expect(sklearnUnsupLines('segment', base)).toContain('KMeans(n_clusters=4, init="kmeans++", n_init=1)');
    expect(sklearnUnsupLines('denoise', base)).toContain('make_pipeline(PCA(n_components=2), KMeans(n_clusters=4');
    expect(sklearnUnsupLines('denoise', { ...base, b3M: 8 })).toContain('no PCA');
  });
});
