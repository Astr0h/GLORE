// Canvas waveform with optional beat grid, phrase regions, playhead and click-to-seek.

const peakCache = new WeakMap();

function computePeaks(buffer, width) {
  let byWidth = peakCache.get(buffer);
  if (!byWidth) { byWidth = new Map(); peakCache.set(buffer, byWidth); }
  if (byWidth.has(width)) return byWidth.get(width);
  const chans = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) chans.push(buffer.getChannelData(c));
  const len = chans[0].length;
  const step = Math.max(1, Math.floor(len / width));
  const peaks = new Float32Array(width * 2);
  for (let x = 0; x < width; x++) {
    let min = 1, max = -1;
    const start = x * step, end = Math.min(len, start + step);
    for (let i = start; i < end; i += 4) { // stride 4 is plenty for display
      for (const ch of chans) { const v = ch[i]; if (v < min) min = v; if (v > max) max = v; }
    }
    peaks[x * 2] = min; peaks[x * 2 + 1] = max;
  }
  byWidth.set(width, peaks);
  return peaks;
}

class Waveform {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} opts { color, onSeek(seconds), duration() (shared timeline length) }
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = opts;
    this.buffer = null;
    this.beats = []; this.downbeats = []; this.phrases = [];
    this.selection = null; // { start, end } seconds
    this.playhead = 0;
    canvas.addEventListener('click', (e) => {
      if (!this.opts.onSeek || !this._timeline()) return;
      const r = canvas.getBoundingClientRect();
      this.opts.onSeek(((e.clientX - r.left) / r.width) * this._timeline());
    });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  _timeline() { return (this.opts.duration && this.opts.duration()) || (this.buffer ? this.buffer.duration : 0); }

  set(data) { Object.assign(this, data); this.draw(); }

  draw() {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(c.clientWidth * dpr));
    const h = Math.max(1, Math.floor(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const g = c.getContext('2d');
    g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const T = this._timeline();
    if (!this.buffer || !T) return;
    const px = (t) => (t / T) * w;

    // phrase regions (vocals)
    g.fillStyle = css.getPropertyValue('--phrase').trim() || 'rgba(255,255,255,0.06)';
    for (const p of this.phrases) g.fillRect(px(p.start), 0, Math.max(1, px(p.end) - px(p.start)), h);

    // selection
    if (this.selection) {
      g.fillStyle = css.getPropertyValue('--selection').trim();
      g.fillRect(px(this.selection.start), 0, px(this.selection.end) - px(this.selection.start), h);
    }

    // beat grid
    g.fillStyle = css.getPropertyValue('--grid').trim();
    for (const b of this.beats) g.fillRect(Math.round(px(b)), 0, 1, h);
    g.fillStyle = css.getPropertyValue('--grid-strong').trim();
    for (const b of this.downbeats) g.fillRect(Math.round(px(b)), 0, Math.max(1, dpr), h);

    // wave
    const bufW = Math.max(1, Math.floor(px(this.buffer.duration)));
    const peaks = computePeaks(this.buffer, bufW);
    // played part full colour, the rest dimmed
    g.fillStyle = this.opts.color || css.getPropertyValue('--wave').trim();
    const mid = h / 2;
    const head = Math.round(px(this.playhead));
    for (let x = 0; x < bufW; x++) {
      const min = peaks[x * 2], max = peaks[x * 2 + 1];
      const y1 = mid - max * mid * 0.92, y2 = mid - min * mid * 0.92;
      g.globalAlpha = this.playhead > 0 && x > head ? 0.42 : 0.95;
      g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
    g.globalAlpha = 1;

    // playhead
    g.fillStyle = css.getPropertyValue('--playhead').trim();
    g.fillRect(Math.round(px(this.playhead)), 0, Math.max(1, dpr * 1.5), h);
  }
}

window.Waveform = Waveform;
