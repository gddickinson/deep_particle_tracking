// Static-site data layer: GETs come from the JSON exported by scripts/build_static.py (computed by the Python code at
// build time). POST /score is answered in the browser by the JS ports (same numbers as Python, see engine.test.mjs).
// The trained-model routes need the local app.
const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const p = path.split('?')[0].replace(/^\/+/, '');
  if (p === 'models') return { static: true, torch: { available: false }, checkpoints: [] };
  if (p === 'score' && body) return (await import('./scoring.js')).scoreMovie(body.seed, body.sim, body.baseline);
  if (body) throw new Error('Running a trained model needs the local app (Python server with PyTorch).');
  return file(p);
}
