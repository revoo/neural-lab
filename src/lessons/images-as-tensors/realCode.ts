// authored-by: Gemini Antigravity
import type { AliasMethod, ColorRep, TensorDtype, TensorLayout, TensorNorm, VisionBriefId } from './vision';

/** Live Python / PyTorch preprocessing snippet matching the active Vision Preprocessor settings. */
export function visionCodeLines(
  briefId: VisionBriefId,
  st: { rep: ColorRep; alias: AliasMethod; layout: TensorLayout; dtype: TensorDtype; norm: TensorNorm },
): string {
  if (briefId === 'lighting') {
    if (st.rep === 'raw-rgb') return 'feat = img.astype(np.float32)  # raw RGB (sensitive to brightness)';
    if (st.rep === 'gray-avg') return 'feat = img.mean(axis=-1)  # naive greyscale';
    if (st.rep === 'gray-709') return 'feat = 0.2126*R + 0.7152*G + 0.0722*B  # Rec. 709 luminance';
    if (st.rep === 'rgb-l1') return 'feat = img / np.maximum(img.sum(axis=-1, keepdims=True), 1.0)  # chromaticity';
    return 'hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)\nfeat = hsv[..., 0]  # hue channel (brightness-invariant)';
  }
  if (briefId === 'aliasing') {
    if (st.alias === 'stride-0') return 'thumb = img[::4, ::4]  # un-filtered strided subsampling (aliases!)';
    if (st.alias === 'stride-1') return 'thumb = img[1::4, 1::4]  # shifted strided slice (still aliases!)';
    if (st.alias === 'max-pool') return 'thumb = F.max_pool2d(img, kernel_size=4, stride=4)';
    return 'thumb = F.avg_pool2d(img, kernel_size=4, stride=4)  # or cv2.resize(..., cv2.INTER_AREA)';
  }
  const dt = st.dtype === 'uint8' ? 'torch.uint8' : st.dtype === 'fp16' ? 'torch.float16' : 'torch.float32';
  const perm = st.layout === 'NCHW' ? 'x = batch.permute(0, 3, 1, 2)  # NHWC -> NCHW' : 'x = batch  # stays NHWC (channels last)';
  const cast = `x = x.to(${dt})`;
  const norm =
    st.norm === 'none'
      ? '# raw 0..255 pixel range'
      : st.norm === 'div255'
        ? 'x = x / 255.0'
        : 'x = (x - mean[:, None, None]) / std[:, None, None]  # per-channel z-score';
  return `${perm}\n${cast}\n${norm}`;
}
