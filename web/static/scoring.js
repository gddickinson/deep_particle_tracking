// Score a seeded movie with the classical baseline in the browser: the same result shape as POST /api/score.
import { simulateMovie, gtTracks, measuredSnr } from './sim_engine.js';
import { runBaseline, allMetrics } from './track_engine.js';

export const MATCH_PX = 5.0;

export function scoreRun(movie, baseline = {}) {
  const b = runBaseline(movie.frames, movie.H, movie.W, baseline);
  const m = allMetrics(b.positions, movie.gt.map(g => g.map(([, y, x]) => [y, x])), b.tracks, gtTracks(movie), MATCH_PX);
  const d = m.detection, t = m.tracking;
  return { result: b, metrics: m, summary: { precision: d.precision, recall: d.recall, f1: d.f1, mota: t.mota, motp: t.motp,
    id_switches: t.id_switches, tracks: b.tracks.length, measured_snr: measuredSnr(movie).snr } };
}

export function scoreMovie(seed, sim = {}, baseline = {}) {
  return scoreRun(simulateMovie(seed, sim), baseline).summary;
}
