// authored-by: Gemini Antigravity
/**
 * The Kestrel K2 Shop environment: real state, real JSON Schema validation, real business-rule
 * guards, and a transparent deterministic policy standing in for a language model.
 *
 * No LLM runs in the browser. Every tool call, schema check, database mutation and prompt
 * transcript is real; the "model" deciding the next action is a small set of inspectable rules.
 */

export type OrderStatus = 'delivered' | 'refunded' | 'repair_open';

export type Order = {
  order_id: string;
  customer: string;
  item: string;
  price_eur: number;
  days_since_delivery: number;
  used: boolean;
  status: OrderStatus;
};

export type StockItem = {
  sku: string;
  name: string;
  in_stock: number;
};

export type RefundRecord = {
  refund_id: string;
  order_id: string;
  amount_eur: number;
  reason: string;
};

export type RepairTicket = {
  ticket_id: string;
  order_id: string;
  part_sku: string;
  covered_by_warranty: boolean;
};

export type ShopState = {
  orders: Order[];
  inventory: StockItem[];
  refunds: RefundRecord[];
  tickets: RepairTicket[];
};

export function initialShopState(): ShopState {
  return {
    orders: [
      { order_id: 'K-1042', customer: 'M. Lind', item: 'Kestrel K2', price_eur: 599, days_since_delivery: 9, used: false, status: 'delivered' },
      { order_id: 'K-1088', customer: 'R. Patel', item: 'Kestrel K2', price_eur: 599, days_since_delivery: 10, used: true, status: 'delivered' },
      { order_id: 'K-1103', customer: 'A. Novak', item: 'Kestrel K2', price_eur: 599, days_since_delivery: 182, used: true, status: 'delivered' },
      { order_id: 'K-1015', customer: 'C. Osei', item: 'Kestrel K2', price_eur: 599, days_since_delivery: 6, used: false, status: 'refunded' },
    ],
    inventory: [
      { sku: 'K2-PUMP', name: 'Vibration pump assembly', in_stock: 4 },
      { sku: 'K2-BOILER', name: 'Thermoblock boiler', in_stock: 0 },
      { sku: 'K2-GASKET', name: 'Group head gasket', in_stock: 19 },
    ],
    refunds: [
      { refund_id: 'RF-001', order_id: 'K-1015', amount_eur: 599, reason: 'unused_return' },
    ],
    tickets: [],
  };
}

export function cloneState(s: ShopState): ShopState {
  return {
    orders: s.orders.map((o) => ({ ...o })),
    inventory: s.inventory.map((i) => ({ ...i })),
    refunds: s.refunds.map((r) => ({ ...r })),
    tickets: s.tickets.map((t) => ({ ...t })),
  };
}

// ------------------------------------------------------------------ tool schemas

export type FieldSpec = {
  type: 'string' | 'number';
  required: boolean;
  pattern?: RegExp;
  patternHint?: string;
  enum?: readonly string[];
  min?: number;
  max?: number;
};

export type ToolSpec = {
  name: 'lookup_order' | 'check_stock' | 'issue_refund' | 'create_repair_ticket';
  description: string;
  sideEffect: boolean;
  fields: Record<string, FieldSpec>;
};

export const TOOL_SPECS: ToolSpec[] = [
  {
    name: 'lookup_order',
    description: 'Fetch an order record by its Kestrel order ID.',
    sideEffect: false,
    fields: {
      order_id: { type: 'string', required: true, pattern: /^K-\d{4}$/, patternHint: 'format "K-1234"' },
    },
  },
  {
    name: 'check_stock',
    description: 'Check warehouse stock for a spare-part SKU.',
    sideEffect: false,
    fields: {
      sku: { type: 'string', required: true, enum: ['K2-PUMP', 'K2-BOILER', 'K2-GASKET'] },
    },
  },
  {
    name: 'issue_refund',
    description: 'Refund money to the customer’s original card (irreversible).',
    sideEffect: true,
    fields: {
      order_id: { type: 'string', required: true, pattern: /^K-\d{4}$/, patternHint: 'format "K-1234"' },
      amount_eur: { type: 'number', required: true, min: 1, max: 1000 },
      reason: { type: 'string', required: true, enum: ['unused_return', 'used_return_minus_fee', 'warranty_fault'] },
    },
  },
  {
    name: 'create_repair_ticket',
    description: 'Open a free warranty repair ticket and reserve a courier collection.',
    sideEffect: true,
    fields: {
      order_id: { type: 'string', required: true, pattern: /^K-\d{4}$/, patternHint: 'format "K-1234"' },
      part_sku: { type: 'string', required: true, enum: ['K2-PUMP', 'K2-BOILER', 'K2-GASKET'] },
    },
  },
];

export const TOOL_BY_NAME = Object.fromEntries(TOOL_SPECS.map((t) => [t.name, t])) as Record<ToolSpec['name'], ToolSpec>;

export const SYSTEM_PROMPT = [
  'You are a support agent for Kestrel Espresso.',
  'Policy: unused K2 returns within 30 days get a full refund; used K2 returns within 14 days get a refund minus a 25 EUR cleaning fee; faults within 730 days (2 years) are covered by a free warranty repair.',
  'Always inspect the order before acting. To call a tool, emit one JSON object: {"tool": "<name>", "args": { ... }}.',
  'Tools:',
  ...TOOL_SPECS.map((t) => {
    const args = Object.entries(t.fields)
      .map(([k, f]) => `${k}: ${f.enum ? f.enum.map((v) => `"${v}"`).join(' | ') : f.type}`)
      .join(', ');
    return `  - ${t.name}(${args}): ${t.description}`;
  }),
].join('\n');

// ------------------------------------------------------------------ 3-stage validation

export type StageResult = { ok: boolean; detail: string };

export type ParsedCall = {
  tool: ToolSpec['name'];
  args: Record<string, unknown>;
};

export type ValidationReport = {
  syntax: StageResult;
  schema: StageResult;
  business: StageResult;
  parsed?: ParsedCall;
  overallOk: boolean;
};

/**
 * Three-stage gatekeeper:
 *  1. JSON syntax (what constrained decoding can guarantee)
 *  2. JSON Schema: known tool, required keys, types, enums, numeric bounds (structured outputs)
 *  3. Business & state invariants: order exists, not already refunded, amount within order price
 *     and return policy (only your application code can enforce this).
 */
export function validateToolCall(raw: string, state: ShopState): ValidationReport {
  let obj: unknown;
  try {
    obj = JSON.parse(raw);
  } catch (err) {
    return {
      syntax: { ok: false, detail: `Invalid JSON: ${err instanceof Error ? err.message : 'parse error'}` },
      schema: { ok: false, detail: 'Skipped — could not parse JSON.' },
      business: { ok: false, detail: 'Skipped — could not parse JSON.' },
      overallOk: false,
    };
  }
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) {
    return {
      syntax: { ok: true, detail: 'Valid JSON value.' },
      schema: { ok: false, detail: 'Top-level value must be a JSON object with "tool" and "args".' },
      business: { ok: false, detail: 'Skipped — failed schema validation.' },
      overallOk: false,
    };
  }
  const rec = obj as Record<string, unknown>;
  const toolName = rec.tool;
  if (typeof toolName !== 'string' || !(toolName in TOOL_BY_NAME)) {
    return {
      syntax: { ok: true, detail: 'Valid JSON object.' },
      schema: { ok: false, detail: `Unknown tool ${JSON.stringify(toolName)}. Allowed: ${TOOL_SPECS.map((t) => t.name).join(', ')}.` },
      business: { ok: false, detail: 'Skipped — failed schema validation.' },
      overallOk: false,
    };
  }
  const spec = TOOL_BY_NAME[toolName as ToolSpec['name']];
  if (typeof rec.args !== 'object' || rec.args === null || Array.isArray(rec.args)) {
    return {
      syntax: { ok: true, detail: 'Valid JSON object.' },
      schema: { ok: false, detail: '"args" must be a JSON object.' },
      business: { ok: false, detail: 'Skipped — failed schema validation.' },
      overallOk: false,
    };
  }
  const args = rec.args as Record<string, unknown>;
  const schemaErrors: string[] = [];
  for (const key of Object.keys(args)) {
    if (!(key in spec.fields)) schemaErrors.push(`Unexpected argument "${key}" for ${spec.name}.`);
  }
  for (const [key, field] of Object.entries(spec.fields)) {
    const v = args[key];
    if (v === undefined || v === null) {
      if (field.required) schemaErrors.push(`Missing required argument "${key}".`);
      continue;
    }
    if (field.type === 'string') {
      if (typeof v !== 'string') {
        schemaErrors.push(`"${key}" must be a string, got ${typeof v}.`);
        continue;
      }
      if (field.pattern && !field.pattern.test(v)) {
        schemaErrors.push(`"${key}" value "${v}" does not match ${field.patternHint ?? field.pattern.source}.`);
      }
      if (field.enum && !field.enum.includes(v)) {
        schemaErrors.push(`"${key}" must be one of [${field.enum.join(', ')}], got "${v}".`);
      }
    } else if (field.type === 'number') {
      if (typeof v !== 'number' || !Number.isFinite(v)) {
        schemaErrors.push(`"${key}" must be a number, got ${typeof v}.`);
        continue;
      }
      if (field.min !== undefined && v < field.min) schemaErrors.push(`"${key}" (${v}) is below minimum ${field.min}.`);
      if (field.max !== undefined && v > field.max) schemaErrors.push(`"${key}" (${v}) exceeds maximum ${field.max}.`);
    }
  }
  if (schemaErrors.length > 0) {
    return {
      syntax: { ok: true, detail: 'Valid JSON object.' },
      schema: { ok: false, detail: schemaErrors.join(' ') },
      business: { ok: false, detail: 'Skipped — failed schema validation.' },
      overallOk: false,
    };
  }

  const parsed: ParsedCall = { tool: spec.name, args };

  // Stage 3: business rules & live database invariants
  if (spec.name === 'issue_refund') {
    const oid = args.order_id as string;
    const amt = args.amount_eur as number;
    const order = state.orders.find((o) => o.order_id === oid);
    if (!order) {
      return {
        syntax: { ok: true, detail: 'Valid JSON object.' },
        schema: { ok: true, detail: `Matches ${spec.name} schema.` },
        business: { ok: false, detail: `Order ${oid} does not exist — cannot refund an unknown order.` },
        parsed,
        overallOk: false,
      };
    }
    if (order.status === 'refunded') {
      return {
        syntax: { ok: true, detail: 'Valid JSON object.' },
        schema: { ok: true, detail: `Matches ${spec.name} schema.` },
        business: { ok: false, detail: `Order ${oid} was already refunded (idempotency guard blocked a double refund).` },
        parsed,
        overallOk: false,
      };
    }
    if (amt > order.price_eur) {
      return {
        syntax: { ok: true, detail: 'Valid JSON object.' },
        schema: { ok: true, detail: `Matches ${spec.name} schema.` },
        business: { ok: false, detail: `Refund €${amt} exceeds order ${oid} total of €${order.price_eur}.` },
        parsed,
        overallOk: false,
      };
    }
    const maxAllowed = order.used ? order.price_eur - 25 : order.price_eur;
    const windowDays = order.used ? 14 : 30;
    if (order.days_since_delivery > windowDays) {
      return {
        syntax: { ok: true, detail: 'Valid JSON object.' },
        schema: { ok: true, detail: `Matches ${spec.name} schema.` },
        business: { ok: false, detail: `Order ${oid} is ${order.days_since_delivery} days old (past the ${windowDays}-day return window).` },
        parsed,
        overallOk: false,
      };
    }
    if (amt > maxAllowed) {
      return {
        syntax: { ok: true, detail: 'Valid JSON object.' },
        schema: { ok: true, detail: `Matches ${spec.name} schema.` },
        business: { ok: false, detail: `Used machine requires a €25 cleaning fee; max refund is €${maxAllowed}, not €${amt}.` },
        parsed,
        overallOk: false,
      };
    }
  } else if (spec.name === 'create_repair_ticket') {
    const oid = args.order_id as string;
    const order = state.orders.find((o) => o.order_id === oid);
    if (!order) {
      return {
        syntax: { ok: true, detail: 'Valid JSON object.' },
        schema: { ok: true, detail: `Matches ${spec.name} schema.` },
        business: { ok: false, detail: `Order ${oid} does not exist.` },
        parsed,
        overallOk: false,
      };
    }
    if (order.days_since_delivery > 730) {
      return {
        syntax: { ok: true, detail: 'Valid JSON object.' },
        schema: { ok: true, detail: `Matches ${spec.name} schema.` },
        business: { ok: false, detail: `Order ${oid} is past the 2-year (730-day) warranty.` },
        parsed,
        overallOk: false,
      };
    }
  }

  return {
    syntax: { ok: true, detail: 'Valid JSON object.' },
    schema: { ok: true, detail: `Matches ${spec.name} schema.` },
    business: { ok: true, detail: spec.sideEffect ? 'Passed all shop invariants and idempotency checks.' : 'Read-only lookup — safe to execute.' },
    parsed,
    overallOk: true,
  };
}

// ------------------------------------------------------------------ tool execution

export type ToolExecResult = {
  ok: boolean;
  output: Record<string, unknown>;
  nextState: ShopState;
  touchedOrder?: string;
  touchedSku?: string;
  mutated: boolean;
};

export function executeTool(call: ParsedCall, state: ShopState): ToolExecResult {
  const next = cloneState(state);
  if (call.tool === 'lookup_order') {
    const oid = call.args.order_id as string;
    const order = next.orders.find((o) => o.order_id === oid);
    if (!order) return { ok: false, output: { error: 'ORDER_NOT_FOUND', order_id: oid }, nextState: next, mutated: false };
    return { ok: true, output: { ...order }, nextState: next, touchedOrder: oid, mutated: false };
  }
  if (call.tool === 'check_stock') {
    const sku = call.args.sku as string;
    const item = next.inventory.find((i) => i.sku === sku);
    if (!item) return { ok: false, output: { error: 'SKU_NOT_FOUND', sku }, nextState: next, mutated: false };
    return { ok: true, output: { ...item }, nextState: next, touchedSku: sku, mutated: false };
  }
  if (call.tool === 'issue_refund') {
    const oid = call.args.order_id as string;
    const amt = call.args.amount_eur as number;
    const reason = call.args.reason as string;
    const order = next.orders.find((o) => o.order_id === oid)!;
    order.status = 'refunded';
    const refund_id = `RF-00${next.refunds.length + 1}`;
    next.refunds.push({ refund_id, order_id: oid, amount_eur: amt, reason });
    return {
      ok: true,
      output: { refund_id, order_id: oid, amount_eur: amt, status: 'paid_to_original_card', eta_working_days: 5 },
      nextState: next,
      touchedOrder: oid,
      mutated: true,
    };
  }
  // create_repair_ticket
  const oid = call.args.order_id as string;
  const part_sku = call.args.part_sku as string;
  const order = next.orders.find((o) => o.order_id === oid)!;
  order.status = 'repair_open';
  const item = next.inventory.find((i) => i.sku === part_sku);
  if (item && item.in_stock > 0) item.in_stock -= 1;
  const ticket_id = `RP-00${next.tickets.length + 1}`;
  next.tickets.push({ ticket_id, order_id: oid, part_sku, covered_by_warranty: true });
  return {
    ok: true,
    output: { ticket_id, order_id: oid, part_sku, courier_collection: 'booked', cost_eur: 0 },
    nextState: next,
    touchedOrder: oid,
    touchedSku: part_sku,
    mutated: true,
  };
}

// ------------------------------------------------------------------ deterministic inspectable agent loop

export type CustomerTicket = {
  id: 'unused' | 'used' | 'warranty' | 'missing';
  label: string;
  message: string;
};

export const CUSTOMER_TICKETS: CustomerTicket[] = [
  {
    id: 'unused',
    label: 'Unused return (K-1042)',
    message: 'Hi, order K-1042 arrived last week and we haven’t opened the box. Can I get a refund?',
  },
  {
    id: 'used',
    label: 'Used return (K-1088)',
    message: 'I tried my Kestrel K2 from order K-1088 for a few days and decided it’s not for me. Please refund it.',
  },
  {
    id: 'warranty',
    label: 'Broken pump (K-1103)',
    message: 'My machine from order K-1103 stopped building pressure — the pump is dead after six months. Can you refund or fix it?',
  },
  {
    id: 'missing',
    label: 'Unknown order (K-9999)',
    message: 'Can you check on a refund for order K-9999?',
  },
];

export type TranscriptTurn = {
  role: 'user' | 'assistant_call' | 'tool_result' | 'assistant_final';
  text: string;
};

export type TraceStep = {
  step: number;
  rule: string;
  kind: 'tool_call' | 'final_answer';
  call?: ParsedCall;
  callJson?: string;
  validation?: ValidationReport;
  toolOutput?: Record<string, unknown>;
  finalAnswer?: string;
  stateAfter: ShopState;
  touchedOrder?: string;
  touchedSku?: string;
  mutated: boolean;
  transcriptSoFar: TranscriptTurn[];
  note: string;
};

/**
 * Deterministic policy standing in for an LLM so every decision is transparent and inspectable:
 *   Rule 1: If no order has been looked up yet, extract the K-#### ID and call lookup_order.
 *   Rule 2: If lookup_order returned ORDER_NOT_FOUND, apologise and ask for the order number.
 *   Rule 3: If unused & <= 30 days & delivered -> issue_refund(price_eur, "unused_return").
 *   Rule 4: If used & <= 14 days & delivered -> issue_refund(price_eur - 25, "used_return_minus_fee").
 *   Rule 5: If fault mentioned & <= 730 days & delivered -> check_stock("K2-PUMP"), then create_repair_ticket.
 *   Rule 6: Once the action succeeded, reply to the customer with the exact IDs and amounts.
 */
export const POLICY_RULES = [
  '1. Never guess order details: if no lookup_order observation exists yet, extract K-#### and call lookup_order.',
  '2. If lookup_order returns ORDER_NOT_FOUND, stop and ask the customer to double-check their order ID.',
  '3. If the order is delivered, unused, and ≤ 30 days old: call issue_refund for the full price.',
  '4. If the order is delivered, used, and ≤ 14 days old: call issue_refund for price − €25 cleaning fee.',
  '5. If the return window has passed but a fault is reported within 730 days: call check_stock for the part, then create_repair_ticket.',
  '6. Once the tool result confirms the refund or repair ticket, emit a final answer citing the real IDs and numbers.',
];

export function runTicketTrace(ticketId: CustomerTicket['id']): TraceStep[] {
  const ticket = CUSTOMER_TICKETS.find((t) => t.id === ticketId)!;
  let state = initialShopState();
  const transcript: TranscriptTurn[] = [{ role: 'user', text: ticket.message }];
  const steps: TraceStep[] = [];

  let orderObs: Order | null = null;
  let orderMissing = false;
  let stockObs: StockItem | null = null;
  let actionDone: { kind: 'refund' | 'repair'; output: Record<string, unknown> } | null = null;

  for (let s = 1; s <= 6; s++) {
    let call: ParsedCall | null = null;
    let rule = '';
    let finalAnswer: string | null = null;

    if (!orderObs && !orderMissing) {
      const m = ticket.message.match(/K-\d{4}/);
      const oid = m ? m[0] : 'K-0000';
      rule = POLICY_RULES[0];
      call = { tool: 'lookup_order', args: { order_id: oid } };
    } else if (orderMissing) {
      rule = POLICY_RULES[1];
      finalAnswer = 'I couldn’t find order K-9999 in our system. Could you double-check the order number on your confirmation email?';
    } else if (orderObs && !actionDone) {
      if (!orderObs.used && orderObs.days_since_delivery <= 30 && orderObs.status === 'delivered') {
        rule = POLICY_RULES[2];
        call = { tool: 'issue_refund', args: { order_id: orderObs.order_id, amount_eur: orderObs.price_eur, reason: 'unused_return' } };
      } else if (orderObs.used && orderObs.days_since_delivery <= 14 && orderObs.status === 'delivered') {
        rule = POLICY_RULES[3];
        call = { tool: 'issue_refund', args: { order_id: orderObs.order_id, amount_eur: orderObs.price_eur - 25, reason: 'used_return_minus_fee' } };
      } else if (orderObs.days_since_delivery <= 730 && orderObs.status === 'delivered') {
        if (!stockObs) {
          rule = POLICY_RULES[4];
          call = { tool: 'check_stock', args: { sku: 'K2-PUMP' } };
        } else {
          rule = POLICY_RULES[4];
          call = { tool: 'create_repair_ticket', args: { order_id: orderObs.order_id, part_sku: 'K2-PUMP' } };
        }
      }
    } else if (actionDone) {
      rule = POLICY_RULES[5];
      if (actionDone.kind === 'refund') {
        const o = actionDone.output;
        finalAnswer =
          orderObs && orderObs.used
            ? `I’ve issued refund ${o.refund_id} for €${o.amount_eur} on order ${o.order_id} (€599 minus the €25 cleaning fee for a used machine returned within 14 days). It will reach your original card within 5 working days.`
            : `I’ve issued full refund ${o.refund_id} for €${o.amount_eur} on order ${o.order_id} (unused return within 30 days). It will reach your original card within 5 working days.`;
      } else {
        const o = actionDone.output;
        finalAnswer = `Order ${o.order_id} is past the 14-day return window (182 days), so it isn’t eligible for a refund, but the pump is covered by your two-year warranty. We have 4 pumps in stock and I’ve opened free repair ticket ${o.ticket_id} with courier collection booked.`;
      }
    }

    if (call) {
      const callJson = JSON.stringify({ tool: call.tool, args: call.args });
      const validation = validateToolCall(callJson, state);
      const exec = executeTool(call, state);
      state = exec.nextState;
      transcript.push({ role: 'assistant_call', text: callJson });
      transcript.push({ role: 'tool_result', text: JSON.stringify(exec.output) });

      if (call.tool === 'lookup_order') {
        if (exec.ok) orderObs = exec.output as unknown as Order;
        else orderMissing = true;
      } else if (call.tool === 'check_stock' && exec.ok) {
        stockObs = exec.output as unknown as StockItem;
      } else if (call.tool === 'issue_refund' && exec.ok) {
        actionDone = { kind: 'refund', output: exec.output };
      } else if (call.tool === 'create_repair_ticket' && exec.ok) {
        actionDone = { kind: 'repair', output: exec.output };
      }

      const note = exec.mutated
        ? `Turn ${s}: the host validated ${call.tool} and mutated the database (${exec.touchedOrder ? `order ${exec.touchedOrder}` : ''}${exec.touchedSku ? `, reserved 1 × ${exec.touchedSku}` : ''}). The result is pasted back into the prompt so the model can see it succeeded.`
        : exec.ok
          ? `Turn ${s}: read-only tool ${call.tool} ran in your host code and returned JSON. The database did not change; the model now reads that observation on the next turn.`
          : `Turn ${s}: ${call.tool} returned an error observation (${String(exec.output.error)}). Good tools return structured errors instead of crashing, so the loop can recover cleanly.`;

      steps.push({
        step: s,
        rule,
        kind: 'tool_call',
        call,
        callJson,
        validation,
        toolOutput: exec.output,
        stateAfter: cloneState(state),
        touchedOrder: exec.touchedOrder,
        touchedSku: exec.touchedSku,
        mutated: exec.mutated,
        transcriptSoFar: transcript.map((t) => ({ ...t })),
        note,
      });
    } else if (finalAnswer) {
      transcript.push({ role: 'assistant_final', text: finalAnswer });
      steps.push({
        step: s,
        rule,
        kind: 'final_answer',
        finalAnswer,
        stateAfter: cloneState(state),
        mutated: false,
        transcriptSoFar: transcript.map((t) => ({ ...t })),
        note: `Turn ${s}: no tool call emitted — the host sees plain text and ends the loop, sending the final answer to the customer.`,
      });
      break;
    }
  }
  return steps;
}

// ------------------------------------------------------------------ Schema gatekeeper presets

export type GatePreset = {
  id: string;
  label: string;
  raw: string;
  why: string;
};

export const GATE_PRESETS: GatePreset[] = [
  {
    id: 'valid',
    label: 'Valid refund (K-1088)',
    raw: JSON.stringify({ tool: 'issue_refund', args: { order_id: 'K-1088', amount_eur: 574, reason: 'used_return_minus_fee' } }, null, 2),
    why: 'Passes JSON syntax, schema types/enums, and Kestrel’s €25 used-machine fee rule.',
  },
  {
    id: 'syntax',
    label: 'Broken JSON syntax',
    raw: '```json\n{"tool": "issue_refund", "args": {"order_id": "K-1042", "amount_eur": 599,}}\n```',
    why: 'Markdown fences plus a trailing comma — fails JSON.parse before any tool runs. Constrained decoding prevents this class of bug completely.',
  },
  {
    id: 'schema',
    label: 'Wrong type & unknown enum',
    raw: JSON.stringify({ tool: 'issue_refund', args: { order_id: 'K-1042', amount_eur: '599', reason: 'customer_unhappy' } }, null, 2),
    why: 'Valid JSON, but amount_eur is a string and "customer_unhappy" is not in the allowed enum. Strict structured outputs prevent this too.',
  },
  {
    id: 'overrefund',
    label: 'Valid schema, wrong amount',
    raw: JSON.stringify({ tool: 'issue_refund', args: { order_id: 'K-1088', amount_eur: 599, reason: 'used_return_minus_fee' } }, null, 2),
    why: 'Syntax and JSON Schema both pass! Yet K-1088 is a used machine (€574 max). Only your application’s business guard catches it.',
  },
  {
    id: 'duplicate',
    label: 'Valid schema, double refund',
    raw: JSON.stringify({ tool: 'issue_refund', args: { order_id: 'K-1015', amount_eur: 599, reason: 'unused_return' } }, null, 2),
    why: 'Syntax and JSON Schema both pass, but order K-1015 is already refunded! Without an idempotency check in your backend, a retry pays the customer twice.',
  },
];
