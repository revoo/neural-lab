// authored-by: Gemini Antigravity
import type { ConvBriefId } from './conv';

/** Live PyTorch / NumPy snippet matching the active Design the Kernel settings. */
export function convCodeLines(
  briefId: ConvBriefId,
  st: { vertW: number[]; diagW: number[]; k: 3 | 5; pad: 0 | 1 | 2; stride: 1 | 2 },
): string {
  if (briefId === 'stack-shape') {
    return [
      'stack = nn.Sequential(',
      `    nn.Conv2d(C, C, kernel_size=${st.k}, stride=${st.stride}, padding=${st.pad}),`,
      `    nn.Conv2d(C, C, kernel_size=${st.k}, stride=${st.stride}, padding=${st.pad}),`,
      ')',
    ].join('\n');
  }
  const w = briefId === 'vertical' ? st.vertW : st.diagW;
  const rows = [0, 3, 6].map((i) => `[${w[i]}, ${w[i + 1]}, ${w[i + 2]}]`).join(', ');
  return `kernel = torch.tensor([[[${rows}]]], dtype=torch.float32)\nout = F.conv2d(img, kernel, padding=1)`;
}
