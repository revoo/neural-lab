// authored-by: Gemini Antigravity
/**
 * Maps the learner's active settings in Lab: The Credit-Assignment Workbench
 * to clean, standard PyTorch code (`autograd`, `nn.init`, `requires_grad`,
 * `register_hook`). Verified against standard PyTorch 2.x APIs.
 */
import type { Brief2Config, Brief3Config } from './creditLab';

export function brief1PyTorchCode(): string {
  return [
    'import torch',
    'import torch.nn.functional as F',
    '',
    'x = torch.tensor([2.0, -1.0])',
    'W1 = torch.tensor([[0.5, -0.6], [-0.5, 0.4]], requires_grad=True)',
    'b1 = torch.tensor([0.0, -0.2], requires_grad=True)',
    'W2 = torch.tensor([[-0.8], [0.7]], requires_grad=True)',
    'b2 = torch.tensor([0.2], requires_grad=True)',
    '',
    'h = F.relu(x @ W1 + b1)          # [1.5, 0.0] — h2 is dead',
    'y_hat = (h @ W2 + b2).squeeze()  # -1.0 vs target 1.0',
    'loss = 0.5 * (y_hat - 1.0) ** 2',
    'loss.backward()                  # fills .grad on every leaf tensor',
    '# W2.grad[0] = -3.0, W1.grad = [[+3.2, 0.0], [-1.6, 0.0]]',
  ].join('\n');
}

export function brief2PyTorchCode(cfg: Brief2Config): string {
  const actLine =
    cfg.act === 'tanh'
      ? 'act = nn.Tanh()'
      : cfg.act === 'sigmoid'
        ? 'act = nn.Sigmoid()'
        : 'act = nn.ReLU()';

  let initBlock: string;
  if (cfg.initMode === 'zeros') {
    initBlock = [
      'nn.init.zeros_(fc1.weight); nn.init.zeros_(fc1.bias)',
      'nn.init.zeros_(fc2.weight); nn.init.zeros_(fc2.bias)  # dh = dz2 @ 0 = 0',
    ].join('\n');
  } else if (cfg.initMode === 'broken') {
    initBlock = [
      'nn.init.constant_(fc1.weight, 3.5); nn.init.constant_(fc1.bias, 0.2)',
      'nn.init.constant_(fc2.weight, 1.0); nn.init.zeros_(fc2.bias)  # identical twins',
    ].join('\n');
  } else if (cfg.initMode === 'asym-large') {
    initBlock = [
      'nn.init.normal_(fc1.weight, std=6.0)  # asymmetric, but saturates tanh/sigmoid',
      'nn.init.normal_(fc2.weight, std=2.5)',
    ].join('\n');
  } else {
    initBlock = [
      'nn.init.normal_(fc1.weight, std=0.8)  # Xavier-scale: breaks symmetry without saturation',
      'nn.init.normal_(fc2.weight, std=0.8)',
    ].join('\n');
  }

  return [
    'import torch, torch.nn as nn',
    '',
    'fc1, fc2 = nn.Linear(2, 4), nn.Linear(4, 1)',
    actLine,
    initBlock,
    `opt = torch.optim.SGD(list(fc1.parameters()) + list(fc2.parameters()), lr=${cfg.lr})`,
    'for _ in range(200):',
    '    opt.zero_grad()',
    '    loss = nn.functional.binary_cross_entropy_with_logits(fc2(act(fc1(X))), y)',
    '    loss.backward()',
    '    opt.step()',
  ].join('\n');
}

export function brief3PyTorchCode(cfg: Brief3Config): string {
  const lines = [
    'import torch, torch.nn as nn',
    '',
    'fc1, fc2 = nn.Linear(2, 2), nn.Linear(2, 1)',
    `for p in fc2.parameters(): p.requires_grad = ${cfg.trainOut ? 'True' : 'False'}`,
  ];

  if (!cfg.trainH1 && !cfg.trainH2) {
    lines.push('for p in fc1.parameters(): p.requires_grad = False  # frozen random features');
  } else if (cfg.trainH1 && cfg.trainH2) {
    lines.push('for p in fc1.parameters(): p.requires_grad = True   # both hidden units train');
  } else {
    const mask = `[${cfg.trainH1 ? '1.0' : '0.0'}, ${cfg.trainH2 ? '1.0' : '0.0'}]`;
    lines.push(`mask = torch.tensor(${mask})[:, None]  # selectively freeze one hidden unit`);
    lines.push('fc1.weight.register_hook(lambda g: g * mask)');
    lines.push('fc1.bias.register_hook(lambda g: g * mask.squeeze(1))');
  }

  lines.push(`opt = torch.optim.SGD([p for p in (*fc1.parameters(), *fc2.parameters()) if p.requires_grad], lr=${cfg.lr})`);
  return lines.join('\n');
}
