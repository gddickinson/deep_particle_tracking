// NumPy legacy RandomState (MT19937) in JavaScript, so a seed gives the same draws as np.random.seed(seed) in Python.
// Covers random_sample / rand, randint (masked rejection), uniform, the polar Box-Muller gauss behind randn / normal
// (with its cached second value) and poisson (multiplication method below 10, PTRS from 10), as in
// numpy/random/src/legacy + distributions.c + mt19937. Copied from the lightsheet_simulator web app (+ poisson).

const N = 624, M = 397;

export class RandomState {
  constructor(seed = 0) { this.mt = new Uint32Array(N); this.seed(seed); }

  // np.random.seed(int): mt19937_seed (Knuth's initialiser, the same as init_genrand), clears the cached gauss.
  seed(s) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, p) + i) >>> 0;
    }
    this.pos = N; this.hasGauss = false; this.gaussVal = 0;
  }

  _gen() {
    const mt = this.mt;
    let y, i = 0;
    for (; i < N - M; i++) { y = (mt[i] & 0x80000000) | (mt[i + 1] & 0x7fffffff); mt[i] = mt[i + M] ^ (y >>> 1) ^ (-(y & 1) & 0x9908b0df); }
    for (; i < N - 1; i++) { y = (mt[i] & 0x80000000) | (mt[i + 1] & 0x7fffffff); mt[i] = mt[i + M - N] ^ (y >>> 1) ^ (-(y & 1) & 0x9908b0df); }
    y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff);
    mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ (-(y & 1) & 0x9908b0df);
    this.pos = 0;
  }

  uint32() {
    if (this.pos >= N) this._gen();
    let y = this.mt[this.pos++];
    y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
    return y >>> 0;
  }

  // random_sample(): 53-bit double in [0, 1)
  random() { const a = this.uint32() >>> 5, b = this.uint32() >>> 6; return (a * 67108864 + b) / 9007199254740992; }

  // np.random.rand(n) / random_sample(n) in C order
  rand(n) { const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = this.random(); return out; }

  // np.random.randint(low, high): integers in [low, high) by masked rejection sampling (32-bit path)
  randint(low, high) {
    if (high <= low) throw new Error(`randint: low >= high (${low}, ${high})`);
    const rng = high - 1 - low;
    if (rng === 0) return low;
    let mask = rng; mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    mask >>>= 0;
    let v;
    while ((v = (this.uint32() & mask) >>> 0) > rng);
    return low + v;
  }

  uniform(low, high) { return low + (high - low) * this.random(); }

  // legacy_gauss: polar Box-Muller, returns f*x2 and caches f*x1 for the next call
  gauss() {
    if (this.hasGauss) { this.hasGauss = false; const g = this.gaussVal; this.gaussVal = 0; return g; }
    let x1, x2, r2;
    do { x1 = 2 * this.random() - 1; x2 = 2 * this.random() - 1; r2 = x1 * x1 + x2 * x2; } while (r2 >= 1 || r2 === 0);
    const f = Math.sqrt(-2 * Math.log(r2) / r2);
    this.gaussVal = f * x1; this.hasGauss = true;
    return f * x2;
  }

  randn(n) { const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = this.gauss(); return out; }
  normal(loc, scale) { return loc + scale * this.gauss(); }

  // random_poisson(lam): 0 for lam == 0, multiplication method for lam < 10, PTRS (Hoermann 1993) otherwise
  poisson(lam) {
    if (lam >= 10) return this._ptrs(lam);
    if (lam === 0) return 0;
    const enlam = Math.exp(-lam);
    let X = 0, prod = 1.0;
    for (;;) { prod *= this.random(); if (prod > enlam) X += 1; else return X; }
  }

  _ptrs(lam) {
    const slam = Math.sqrt(lam), loglam = Math.log(lam), b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
    const invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
      const U = this.random() - 0.5, V = this.random(), us = 0.5 - Math.abs(U);
      const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
      if (us >= 0.07 && V <= vr) return k;
      if (k < 0 || (us < 0.013 && V > us)) continue;
      if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - loggam(k + 1)) return k;
    }
  }
}

const LOGGAM_A = [8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04, -5.952380952380952e-04,
  8.417508417508418e-04, -1.917526917526918e-03, 6.410256410256410e-03, -2.955065359477124e-02,
  1.796443723688307e-01, -1.39243221690590e+00];

// random_loggam(x): log Gamma(x) as numpy computes it for PTRS
export function loggam(x) {
  if (x === 1.0 || x === 2.0) return 0.0;
  const n = x < 7.0 ? Math.trunc(7 - x) : 0;
  let x0 = x + n;
  const x2 = (1.0 / x0) * (1.0 / x0);
  let gl0 = LOGGAM_A[9];
  for (let k = 8; k >= 0; k--) { gl0 *= x2; gl0 += LOGGAM_A[k]; }
  let gl = gl0 / x0 + 0.5 * 1.8378770664093453e+00 + (x0 - 0.5) * Math.log(x0) - x0;
  if (x < 7.0) for (let k = 1; k <= n; k++) { gl -= Math.log(x0 - 1.0); x0 -= 1.0; }
  return gl;
}
