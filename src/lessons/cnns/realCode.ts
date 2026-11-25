// authored-by: Gemini Antigravity
import { buildBrief1Stack, type CnnBriefId, type HeadChoice, type RfConfig, type ShiftChoice } from './cnn';

export function cnnRealCode(
  briefId: CnnBriefId,
  st: RfConfig & { shiftChoice: ShiftChoice; headChoice: HeadChoice },
): string {
  if (briefId === 'rf-budget') {
    const stack = buildBrief1Stack({ k: st.k, nConv: st.nConv, nPools: st.nPools });
    const lines = stack.map((l) =>
      l.kind === 'conv'
        ? `    nn.Conv2d(16, 16, kernel_size=${l.k}, padding=${l.p}), nn.ReLU(),`
        : `    nn.MaxPool2d(kernel_size=${l.k}, stride=${l.s}),`,
    );
    return `backbone = nn.Sequential(\n${lines.join('\n')}\n)`;
  }

  if (briefId === 'shift-gen') {
    if (st.shiftChoice.startsWith('dense-')) {
      const h = Number(st.shiftChoice.split('-')[1]);
      return [
        `model = nn.Sequential(`,
        `    nn.Flatten(), nn.Linear(64, ${h}), nn.ReLU(), nn.Linear(${h}, 3),`,
        `)`,
      ].join('\n');
    }
    const f = Number(st.shiftChoice.split('-')[1]);
    return [
      `model = nn.Sequential(`,
      `    nn.Conv2d(1, ${f}, kernel_size=3), nn.ReLU(),`,
      `    nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(${f}, 3),`,
      `)`,
    ].join('\n');
  }

  if (st.headChoice === 'vgg-fc') {
    return [
      `head = nn.Sequential(`,
      `    nn.Flatten(), nn.Linear(7 * 7 * 512, 4096), nn.ReLU(), nn.Linear(4096, 1000),`,
      `)`,
    ].join('\n');
  }
  if (st.headChoice === 'flatten-linear') {
    return `head = nn.Sequential(nn.Flatten(), nn.Linear(7 * 7 * 512, 1000))`;
  }
  if (st.headChoice === 'bottleneck-gap') {
    return [
      `head = nn.Sequential(`,
      `    nn.Conv2d(512, 128, kernel_size=1), nn.ReLU(),`,
      `    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1000),`,
      `)`,
    ].join('\n');
  }
  return [
    `head = nn.Sequential(`,
    `    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1000),`,
    `)`,
  ].join('\n');
}
