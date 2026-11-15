// polymarket-faa-attribution-01 entry (Node 22, standard library only).
//
// Model: flat median at each market's last price, with Student-t(4) quantile
// bands whose width is set per market from its own recent realised volatility
// and widens sub-diffusively with horizon:
//
//   q_tau(h) = clamp(x + z_tau * sigma * h^0.40, 0, 1), rounded to 6 decimals
//
//   sigma = clamp(0.7 * std(1-min diffs over the last 120 prices), 0.0006, 0.01)
//
//   z = standardised Student-t(4) quantiles at (0.05, 0.25, 0.5, 0.75, 0.95)
//     = (-1.963243, -0.523742, 0, 0.523742, 1.963243)
//
// Why this shape (calibrated on the kit's practice instance, NVIDIA CEO):
//   - The median is flat: on the practice, every tried drift and every
//     mean-reversion strength made the median error strictly worse
//     (e.g. MR strength 0.25: loss 0.49 vs 0.32 flat). Prices here behave
//     like a martingale at this horizon; the last price is the best centre.
//   - Bands from the last 120 minutes only: longer windows (720/1440/4320
//     min) drag in stale, jumpier regimes and score far worse (0.58-1.28
//     vs 0.31). A 120-minute window adapts to the current regime.
//   - h^0.40, not sqrt(h): realised 4-hour moves grow slower than a random
//     walk (long flat stretches between ticks); alpha 0.40 beats 0.50
//     (0.306 vs 0.340) on the practice.
//   - Student-t(4) tails: wider 5/95 and narrower shoulders than Gaussian,
//     matching the step-and-jump character of these prices; beats Gaussian
//     at the same sigma (0.305 vs 0.321).
//   - sigma floor 0.0006: markets that have not moved in 2 hours still get
//     a minimal band, so a single tick does not fall outside every quantile.
//     Cap 0.01 keeps one wild 2-hour window from exploding the bands.
//
// Practice (p00..p04, truth from the next case's input; p05 unscored, it has
// no successor): loss 0.305, median_error 0.880, coverage_90 0.887, vs the
// house persistence baseline's 0.393 / 0.879 / 0.938 on the same five cases.
// Per case: p00 0.476, p01 0.311, p02 0.343, p03 0.234, p04 0.203 — better
// than the baseline on every case. (The kit quotes the baseline at 0.363589
// over all six practice cases including p05, which has no locally available
// truth; the comparison above is on the five cases both can be scored on.)
//
// Deterministic: pure arithmetic on the case file, no clock, no randomness,
// no network. Runs in well under a second per case.
//
// What an entry must do (from the kit's starter, kept here):
//   - read every <id>.in in the input directory (the grader runs one case at
//     a time, with only that case's file there), canonical JSON
//     {"schema":"makefun/polymarket-case/v1","case","at","horizon_minutes",
//     "quantiles","event","markets":[{"id","name","slug","yes_token","start",
//     "prices"}]}, where `prices` holds one Yes price per minute from `start`
//     to `at` inclusive, so the last one is the price at the checkpoint;
//   - write <id>.out to the output directory: {"forecasts":[{"id","q"}]},
//     every market of the case exactly once, `q` exactly horizon_minutes
//     rows, row h-1 the forecast for minute at+h, each row one finite number
//     in [0,1] per quantile level, non-decreasing; at most 1 MiB.
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/**
 * Standardised Student-t(4) quantiles at each level: t4 ppf
 * (-2.776445, -0.740697, 0, 0.740697, 2.776445) divided by sqrt(4/(4-2)),
 * so the scale is comparable to a unit-variance normal. Wider tails and
 * narrower shoulders than the normal's (-1.644854, -0.674490, 0, ...).
 */
const Z = [-1.963243, -0.523742, 0, 0.523742, 1.963243];
/** Minutes of history the realised volatility is estimated from. */
const VOL_WINDOW = 120;
/** Shrinkage applied to the realised per-minute volatility. */
const VOL_MULT = 0.7;
/** Horizon exponent: band width grows as h^ALPHA, sub-diffusive. */
const ALPHA = 0.4;
/** Minimum / maximum per-minute volatility (price units). */
const SIGMA_FLOOR = 0.0006;
const SIGMA_CAP = 0.01;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Population std of the 1-minute diffs over the last VOL_WINDOW prices. */
function realisedSigma(prices: number[]): number {
  const n = prices.length;
  if (n < 2) return 0;
  const start = Math.max(0, n - VOL_WINDOW);
  const m = n - start - 1; // number of diffs
  if (m < 1) return 0;
  let sum = 0;
  for (let i = start; i < n - 1; i++) sum += (prices[i + 1] as number) - (prices[i] as number);
  const mean = sum / m;
  let sq = 0;
  for (let i = start; i < n - 1; i++) {
    const d = ((prices[i + 1] as number) - (prices[i] as number)) - mean;
    sq += d * d;
  }
  return Math.sqrt(sq / m);
}

/** The forecast of one market: horizon rows of the five quantiles around its last price. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = Math.min(Math.max(realisedSigma(m.prices) * VOL_MULT, SIGMA_FLOOR), SIGMA_CAP);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const w = sigma * Math.pow(h, ALPHA);
    rows.push(Z.map((z) => price6(x + z * w)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
