// Multi-layer player: every layer plays in sync, with mute/solo per layer.
// Monitoring mode (Headphones / Sound System) only changes what you HEAR -
// it sits after the master bus and is never applied to exports.

class GlorePlayer {
  constructor() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    this.layers = new Map(); // id -> { name, buffer, gain, muted, solo, volume }
    this.sources = [];
    this.playing = false;
    this.startedAt = 0;   // ctx time when playback started
    this.offset = 0;      // seconds into the track when playback started
    this.loop = null;     // { start, end } or null
    this.onEnded = null;

    this.master = this.ctx.createGain();
    this._buildMonitorChain();
    this.setMonitorMode('headphones');
  }

  _buildMonitorChain() {
    const c = this.ctx;
    // Sound System: a bit more weight in the lows, slightly tamed harsh mids,
    // and a gentle safety compressor so a loud room doesn't clip.
    this.lowShelf = c.createBiquadFilter(); this.lowShelf.type = 'lowshelf'; this.lowShelf.frequency.value = 90;
    this.presence = c.createBiquadFilter(); this.presence.type = 'peaking'; this.presence.frequency.value = 3200; this.presence.Q.value = 0.9;
    this.highShelf = c.createBiquadFilter(); this.highShelf.type = 'highshelf'; this.highShelf.frequency.value = 9000;
    this.safety = c.createDynamicsCompressor();
    this.safety.attack.value = 0.003; this.safety.release.value = 0.15; this.safety.ratio.value = 4; this.safety.knee.value = 6;
    this.outGain = c.createGain();
    this.analyser = c.createAnalyser(); this.analyser.fftSize = 2048;
    this.master.connect(this.lowShelf).connect(this.presence).connect(this.highShelf)
      .connect(this.safety).connect(this.outGain).connect(this.analyser).connect(c.destination);
  }

  setMonitorMode(mode) {
    this.monitorMode = mode;
    const t = this.ctx.currentTime;
    const set = (param, v) => param.setTargetAtTime(v, t, 0.05);
    if (mode === 'sound_system') {
      set(this.lowShelf.gain, 3.0);
      set(this.presence.gain, -1.5);
      set(this.highShelf.gain, -1.0);
      set(this.safety.threshold, -8);
      set(this.outGain.gain, 1.0);
    } else { // headphones: flat and detailed, safety only catches overs
      set(this.lowShelf.gain, 0);
      set(this.presence.gain, 0);
      set(this.highShelf.gain, 0);
      set(this.safety.threshold, -1);
      set(this.outGain.gain, 1.0);
    }
  }

  async load(id, name, url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Couldn't load ${name}`);
    const buffer = await this.ctx.decodeAudioData(await res.arrayBuffer());
    const prev = this.layers.get(id);
    const gain = prev ? prev.gain : this.ctx.createGain();
    if (!prev) gain.connect(this.master);
    this.layers.set(id, { name, buffer, gain, muted: prev?.muted ?? false, solo: prev?.solo ?? false, volume: prev?.volume ?? 1 });
    this._applyGains();
    if (this.playing) this.seek(this.position());
    return buffer;
  }

  remove(id) {
    const l = this.layers.get(id);
    if (!l) return;
    l.gain.disconnect();
    this.layers.delete(id);
    if (this.playing) this.seek(this.position());
  }

  has(id) { return this.layers.has(id); }

  duration() {
    let d = 0;
    for (const l of this.layers.values()) d = Math.max(d, l.buffer.duration);
    return d;
  }

  position() {
    if (!this.playing) return this.offset;
    let p = this.offset + (this.ctx.currentTime - this.startedAt);
    if (this.loop) {
      const len = this.loop.end - this.loop.start;
      if (p >= this.loop.end) p = this.loop.start + ((p - this.loop.start) % len);
    }
    return p;
  }

  play(from = this.offset) {
    if (this.ctx.state === 'suspended') this.ctx.resume();
    this._stopSources();
    const dur = this.duration();
    if (!dur) return;
    if (from >= dur) from = 0;
    this.offset = from;
    this.startedAt = this.ctx.currentTime + 0.02;
    for (const l of this.layers.values()) {
      if (from >= l.buffer.duration && !this.loop) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = l.buffer;
      if (this.loop) { src.loop = true; src.loopStart = this.loop.start; src.loopEnd = Math.min(this.loop.end, l.buffer.duration); }
      src.connect(l.gain);
      src.start(this.startedAt, Math.min(from, l.buffer.duration - 0.001));
      this.sources.push(src);
    }
    this.playing = true;
    clearTimeout(this._endTimer);
    if (!this.loop) {
      this._endTimer = setTimeout(() => { this.stop(); this.onEnded && this.onEnded(); }, (dur - from) * 1000 + 60);
    }
  }

  pause() { if (!this.playing) return; this.offset = this.position(); this._stopSources(); this.playing = false; clearTimeout(this._endTimer); }
  stop() { this._stopSources(); this.playing = false; this.offset = this.loop ? this.loop.start : 0; clearTimeout(this._endTimer); }
  toggle() { this.playing ? this.pause() : this.play(); }
  seek(t) { const was = this.playing; this._stopSources(); this.playing = false; this.offset = Math.max(0, t); if (was) this.play(this.offset); }

  setLoop(region) { this.loop = region; if (this.playing) this.seek(region ? region.start : this.position()); }

  setMute(id, muted) { const l = this.layers.get(id); if (l) { l.muted = muted; this._applyGains(); } }
  setSolo(id, solo) { const l = this.layers.get(id); if (l) { l.solo = solo; this._applyGains(); } }
  setVolume(id, v) { const l = this.layers.get(id); if (l) { l.volume = v; this._applyGains(); } }

  _applyGains() {
    const anySolo = [...this.layers.values()].some((l) => l.solo);
    const t = this.ctx.currentTime;
    for (const l of this.layers.values()) {
      const audible = !l.muted && (!anySolo || l.solo);
      l.gain.gain.setTargetAtTime(audible ? l.volume : 0, t, 0.01);
    }
  }

  _stopSources() {
    for (const s of this.sources) { try { s.stop(); } catch (_) { /* already stopped */ } s.disconnect(); }
    this.sources = [];
  }

  // Peak level of the output (0..1), for the meter
  level() {
    const buf = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(buf);
    let peak = 0;
    for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]));
    return peak;
  }
}

window.GlorePlayer = GlorePlayer;
