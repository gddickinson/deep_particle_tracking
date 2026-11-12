// Draw one movie frame on a canvas with optional overlays: ground truth (teal rings), baseline detections (orange
// crosses) and tracks (periwinkle lines up to the current frame). Row 0 is at the top (as saved TIFFs and arrays);
// a pixel centre (y, x) is drawn at ((x + 0.5) s, (y + 0.5) s) for a display scale s.

const COL = { gt: '#5cf2c8', det: '#ff8a3d', track: '#8fa2ff', miss: '#ff5c7a' };
export const COLOURS = COL;

// Grey levels: per-frame min-max, like matplotlib's imshow default in the desktop viewer.
function toImage(frame, H, W) {
  let lo = Infinity, hi = -Infinity;
  for (const v of frame) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const k = hi > lo ? 255 / (hi - lo) : 0, img = new ImageData(W, H);
  for (let i = 0; i < frame.length; i++) {
    const g = (frame[i] - lo) * k, j = 4 * i;
    img.data[j] = img.data[j + 1] = img.data[j + 2] = g; img.data[j + 3] = 255;
  }
  return img;
}

export function drawFrame(canvas, movie, t, opts = {}) {
  const { H, W } = movie, s = Math.max(1, Math.floor((opts.size || 640) / Math.max(H, W)));
  if (canvas.width !== W * s || canvas.height !== H * s) { canvas.width = W * s; canvas.height = H * s; }
  const off = (canvas._off ||= document.createElement('canvas'));
  off.width = W; off.height = H;
  off.getContext('2d').putImageData(toImage(movie.frames[t], H, W), 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, W * s, H * s);
  const P = (y, x) => [(x + 0.5) * s, (y + 0.5) * s];
  ctx.lineWidth = Math.max(1, s / 3);
  if (opts.tracks && opts.result) {
    ctx.strokeStyle = COL.track; ctx.globalAlpha = 0.85;
    for (const tr of opts.result.tracks) {
      ctx.beginPath(); let on = false;
      for (let k = 0; k <= t && k < tr.length; k++) {
        const [y, x] = tr[k];
        if (Number.isNaN(y) || y === null) { on = false; continue; }
        const [px, py] = P(y, x);
        if (on) ctx.lineTo(px, py); else ctx.moveTo(px, py);
        on = true;
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  if (opts.gt) {
    ctx.strokeStyle = COL.gt;
    for (const [, y, x] of movie.gt[t]) { const [px, py] = P(y, x); ctx.beginPath(); ctx.arc(px, py, 3.2 * s, 0, 2 * Math.PI); ctx.stroke(); }
  }
  if (opts.det && opts.result) {
    ctx.strokeStyle = COL.det; const r = 1.8 * s;
    for (const [y, x] of opts.result.positions[t]) {
      const [px, py] = P(y, x); ctx.beginPath();
      ctx.moveTo(px - r, py); ctx.lineTo(px + r, py); ctx.moveTo(px, py - r); ctx.lineTo(px, py + r); ctx.stroke();
    }
  }
  if (opts.bar !== false) {  // 1 µm scale bar (0.1 µm pixels)
    const L = 10 * s;
    ctx.fillStyle = '#fff'; ctx.fillRect(canvas.width - L - 8, canvas.height - 12, L, Math.max(2, s / 2));
    ctx.font = `${Math.max(10, 3 * s)}px ui-monospace, monospace`; ctx.textAlign = 'right';
    ctx.fillText('1 µm', canvas.width - 8, canvas.height - 16);
  }
}
