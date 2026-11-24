/**
 * A tiny scalar automatic-differentiation engine, in the spirit of micrograd.
 *
 * Each Value remembers which Values produced it and the *local derivative* of itself
 * with respect to each of them. Backpropagation is then just: walk the graph in reverse
 * topological order and push `grad × local` into each parent.
 *
 * Storing the local derivative on the edge (rather than a closure) makes the graph easy
 * to visualise: the backprop lesson draws exactly these numbers.
 */
export type Edge = { node: Value; local: number };

let nextId = 0;

export class Value {
  readonly id = nextId++;
  data: number;
  grad = 0;
  readonly op: string;
  label: string;
  readonly parents: Edge[];

  constructor(data: number, label = '', op = '', parents: Edge[] = []) {
    this.data = data;
    this.label = label;
    this.op = op;
    this.parents = parents;
  }

  static of(x: Value | number): Value {
    return x instanceof Value ? x : new Value(x, String(x));
  }

  add(o: Value | number): Value {
    const b = Value.of(o);
    return new Value(this.data + b.data, '', '+', [
      { node: this, local: 1 },
      { node: b, local: 1 },
    ]);
  }
  sub(o: Value | number): Value {
    const b = Value.of(o);
    return new Value(this.data - b.data, '', '−', [
      { node: this, local: 1 },
      { node: b, local: -1 },
    ]);
  }
  mul(o: Value | number): Value {
    const b = Value.of(o);
    return new Value(this.data * b.data, '', '×', [
      { node: this, local: b.data },
      { node: b, local: this.data },
    ]);
  }
  pow(k: number): Value {
    return new Value(this.data ** k, '', `^${k}`, [{ node: this, local: k * this.data ** (k - 1) }]);
  }
  neg(): Value {
    return this.mul(-1);
  }
  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, '', 'exp', [{ node: this, local: e }]);
  }
  log(): Value {
    return new Value(Math.log(this.data), '', 'ln', [{ node: this, local: 1 / this.data }]);
  }
  tanh(): Value {
    const t = Math.tanh(this.data);
    return new Value(t, '', 'tanh', [{ node: this, local: 1 - t * t }]);
  }
  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    return new Value(s, '', 'σ', [{ node: this, local: s * (1 - s) }]);
  }
  relu(): Value {
    return new Value(Math.max(0, this.data), '', 'ReLU', [{ node: this, local: this.data > 0 ? 1 : 0 }]);
  }

  named(label: string): this {
    this.label = label;
    return this;
  }

  /** Compute d(this)/d(node) for every node in the graph. */
  backward(): Value[] {
    const order = topoSort(this);
    for (const v of order) v.grad = 0;
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) {
      const v = order[i];
      for (const e of v.parents) e.node.grad += v.grad * e.local;
    }
    return order;
  }
}

/** Nodes ordered so every node comes after all of its inputs. */
export function topoSort(root: Value): Value[] {
  const seen = new Set<Value>();
  const out: Value[] = [];
  const visit = (v: Value) => {
    if (seen.has(v)) return;
    seen.add(v);
    for (const e of v.parents) visit(e.node);
    out.push(v);
  };
  visit(root);
  return out;
}
