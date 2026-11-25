// authored-by: Gemini Antigravity
import type { InitMode, UnsupBrief } from './unsupervised';

export type UnsupCodeState = {
  b1Scaled: boolean;
  b1M: number;
  b2Scaled: boolean;
  b2K: number;
  b2Init: InitMode;
  b3M: number;
  b3K: number;
};

/** Equivalent scikit-learn PCA / KMeans pipeline for the active Compress & Segment brief. */
export function sklearnUnsupLines(briefId: UnsupBrief['id'], st: UnsupCodeState): string {
  if (briefId === 'pca-scale') {
    const pca = `PCA(n_components=${st.b1M})`;
    return st.b1Scaled
      ? `pipe = make_pipeline(StandardScaler(), ${pca})\nZ = pipe.fit_transform(X_sensors)`
      : `Z = ${pca}.fit_transform(X_sensors)  # unscaled raw units`;
  }
  if (briefId === 'segment') {
    const km = `KMeans(n_clusters=${st.b2K}, init="${st.b2Init}", n_init=1)`;
    return st.b2Scaled
      ? `pipe = make_pipeline(StandardScaler(), ${km})\nlabels = pipe.fit_predict(X_customers)`
      : `labels = ${km}.fit_predict(X_customers)  # unscaled € vs visits`;
  }
  const km = `KMeans(n_clusters=${st.b3K}, init="k-means++")`;
  return st.b3M === 8
    ? `labels = ${km}.fit_predict(X_logs)  # all 8 columns, no PCA`
    : `pipe = make_pipeline(PCA(n_components=${st.b3M}), ${km})\nlabels = pipe.fit_predict(X_logs)`;
}
