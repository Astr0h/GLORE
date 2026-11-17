// GLORE renderer: step-by-step flow UI talking to the local engine.
/* global GlorePlayer, Waveform */

const STEPS = [
  { id: 'import', label: 'Import', ready: true },
  { id: 'analyse', label: 'Analyse', ready: true },
  { id: 'lock', label: 'Vocal Lock', ready: true },
  { id: 'arrange', label: 'Arrange' },
  { id: 'generate', label: 'Generate' },
  { id: 'chops', label: 'Chops' },
  { id: 'vocal_fx', label: 'Vocal FX' },
  { id: 'mix', label: 'Mix' },
  { id: 'export', label: 'Export' },
];

const LATER = {
  arrange: ['Auto-places your vocal from its phrases (hook vs verse)', 'Genre-pack section templates', 'Drag, cut and duplicate vocal blocks on a timeline'],
  generate: ['ACE-Step instrumental in your key and BPM', 'Style prompt + genre LoRA', '2–4 variations, regenerate any single section', 'Beat as reference mode'],
  chops: ['Auto-slice phrases at transients', 'Pitch, reverse, stutter, gate, filter per slice', 'Drop patterns you can reshuffle', 'Select-to-loop for drums'],
  vocal_fx: ['EQ, compression, de-ess, reverb, delay', 'Clean / Airy / Club presets + sliders'],
  mix: ['Kick sidechain', 'Vocal pocket EQ carve', 'Master limiter'],
  export: ['WAV 24-bit, MP3 320', 'MIDI from melodic stems', 'Stems ZIP for Ableton'],
};

const BPM_PRESETS = {
  emotive_house: [128, 130, 132, 134],
  melodic_rap: [140, 145, 150, 155, 160],
};

const KIND_LABEL = { vocal: 'Vocal', full_song: 'Full song', beat: 'Beat' };

const state = {
  project: null,
  step: 'import',
  selected: null,          // source id on the Analyse step
  jobs: {},                // source id -> { progress, message }
  lockJob: null,
  layerVersion: {},        // layer id -> cache-buster
  layerWaves: new Map(),   // layer id -> Waveform (transport rows)
  viewWaves: [],           // waveforms in the main view (need playhead updates)
};

const player = new GlorePlayer();
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k === 'style') n.style.cssText = v; // CSSOM, allowed by the page's CSP
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, '');
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
};

// ------------------------------------------------------------------ engine API
async function api(path, opts = {}) {
  const res = await fetch(window.glore.engineUrl() + path, {
    method: opts.method || 'GET',
    headers: opts.body ? { 'Content-Type': 'application/json' } : undefined,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : `Engine error ${res.status}`);
  return data;
}

async function waitJob(jobId, onProgress) {
  for (;;) {
    const j = await api(`/jobs/${jobId}`);
    onProgress && onProgress(j);
    if (j.status === 'done') return j.result;
    if (j.status === 'error') throw new Error(j.error);
    await new Promise((r) => setTimeout(r, 300));
  }
}

function toast(msg, isError = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), isError ? 6000 : 2500);
}

async function refresh() {
  state.project = await api('/project');
  if (!state.selected || !state.project.sources.find((s) => s.id === state.selected)) {
    state.selected = state.project.sources[0]?.id || null;
  }
  await syncLayers();
  render();
  // Work started by the engine itself (e.g. analysing an extracted vocal) - keep checking
  const pending = state.project.sources.some((s) => !s.analysis && !state.jobs[s.id]);
  clearTimeout(refresh._t);
  if (pending) refresh._t = setTimeout(refresh, 1200);
}

// ------------------------------------------------------------------ boot
async function boot() {
  const msg = $('#boot-msg');
  $('#boot-actions').classList.add('hidden');
  msg.textContent = 'Starting engine…';
  const t0 = Date.now();
  for (;;) {
    try {
      const h = await api('/health');
      const gpu = $('#gpu');
      if (h.gpu.cuda) { gpu.textContent = h.gpu.device; gpu.className = 'gpu ok'; }
      else if (h.gpu.torch) { gpu.textContent = 'CPU only — CUDA not found'; gpu.className = 'gpu warn'; }
      else { gpu.textContent = 'PyTorch missing'; gpu.className = 'gpu warn'; }
      break;
    } catch (_) {
      if (Date.now() - t0 > 45000) {
        msg.textContent = "The engine didn't start. The log will say why — usually setup.ps1 hasn't been run.";
        $('#boot-actions').classList.remove('hidden');
        return;
      }
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  await refresh();
  state.step = state.project.step || 'import';
  render();
  $('#boot').classList.add('hidden');
  requestAnimationFrame(tick);
}

$('#boot-retry').addEventListener('click', async () => { await window.glore.restartEngine(); boot(); });
$('#boot-log').addEventListener('click', () => window.glore.openLog());
window.glore.onEngineExit(() => toast('The engine stopped. Restart GLORE, or check the log.', true));

// ------------------------------------------------------------------ layers / transport
function layerUrl(layer) {
  return `${window.glore.engineUrl()}/audio/${layer.file}?v=${state.layerVersion[layer.id] || 0}`;
}

async function syncLayers() {
  const layers = state.project.layers;
  const ids = new Set(layers.map((l) => l.id));
  for (const id of [...player.layers.keys()]) if (!ids.has(id)) player.remove(id);
  for (const l of layers) {
    const key = l.file + '|' + (state.layerVersion[l.id] || 0);
    if (player.layers.get(l.id)?.key === key) continue;
    try {
      await player.load(l.id, l.name, layerUrl(l));
      player.layers.get(l.id).key = key;
      // Sources start muted except the newest, so stacking samples doesn't get loud
    } catch (e) { toast(e.message, true); }
  }
  renderLayers();
}

function renderLayers() {
  const box = $('#layers');
  box.innerHTML = '';
  state.layerWaves.clear();
  const layers = state.project?.layers || [];
  if (!layers.length) { box.append(el('div', { class: 'empty' }, 'Layers appear here as you add samples.')); return; }
  for (const l of layers) {
    const pl = player.layers.get(l.id);
    if (!pl) continue;
    const canvas = el('canvas');
    const mute = el('button', { class: 'm' + (pl.muted ? ' on' : ''), title: 'Mute', onclick: () => { player.setMute(l.id, !pl.muted); renderLayers(); } }, 'M');
    const solo = el('button', { class: 's' + (pl.solo ? ' on' : ''), title: 'Solo', onclick: () => { player.setSolo(l.id, !pl.solo); renderLayers(); } }, 'S');
    const vol = el('input', { type: 'range', min: 0, max: 1.5, step: 0.01, value: pl.volume, title: 'Volume' });
    vol.addEventListener('input', () => player.setVolume(l.id, Number(vol.value)));
    box.append(el('div', { class: 'layer' }, el('div', { class: 'lname', title: l.name }, l.name), mute, solo, vol, canvas));
    const src = state.project.sources.find((x) => x.id === l.source_id);
    const isVocal = l.role === 'vocal_locked' || (l.role === 'source' && src?.kind === 'vocal') || (l.role === 'stem' && /vocals$/.test(l.id));
    const color = isVocal
      ? getComputedStyle(document.documentElement).getPropertyValue('--wave-vocal').trim() : undefined;
    const w = new Waveform(canvas, { color, onSeek: (t) => player.seek(t), duration: () => player.duration() });
    w.set({ buffer: pl.buffer });
    state.layerWaves.set(l.id, w);
  }
}

function soloOnly(layerId) {
  for (const [id] of player.layers) player.setSolo(id, id === layerId);
  renderLayers();
}

function fmt(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

function tick() {
  const pos = player.position();
  $('#time-pos').textContent = fmt(pos);
  $('#time-dur').textContent = fmt(player.duration());
  $('#btn-play').textContent = player.playing ? '❚❚' : '▶';
  $('#meter-fill').style.width = Math.min(100, player.level() * 100) + '%';
  for (const w of state.layerWaves.values()) { w.playhead = pos; w.draw(); }
  for (const w of state.viewWaves) { w.playhead = pos; w.draw(); }
  requestAnimationFrame(tick);
}

$('#btn-play').addEventListener('click', () => player.toggle());
$('#btn-stop').addEventListener('click', () => player.stop());
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    e.preventDefault(); player.toggle();
  }
});

// ------------------------------------------------------------------ top bar
function renderTop() {
  const p = state.project;
  const name = $('#project-name');
  if (document.activeElement !== name) name.value = p.name;
  const bpm = $('#project-bpm');
  if (document.activeElement !== bpm) bpm.value = p.bpm ?? '';
  $('#bpm-follow').classList.toggle('hidden', !p.bpm_locked_by_user);
  $('#project-key').textContent = p.key || '—';
  $('#genre-pack').value = p.genre_pack;
  for (const b of $('#monitor-mode').children) b.classList.toggle('on', b.dataset.mode === p.monitor_mode);
}

async function patchProject(body) {
  try { state.project = await api('/project', { method: 'PATCH', body }); render(); } catch (e) { toast(e.message, true); }
}

$('#project-name').addEventListener('change', (e) => patchProject({ name: e.target.value || 'Untitled' }));
$('#project-bpm').addEventListener('change', (e) => { const v = Number(e.target.value); if (v) patchProject({ bpm: v }); });
$('#bpm-follow').addEventListener('click', () => patchProject({ follow_vocal: true }));
$('#genre-pack').addEventListener('change', (e) => patchProject({ genre_pack: e.target.value }));
for (const b of $('#monitor-mode').children) {
  b.addEventListener('click', () => { player.setMonitorMode(b.dataset.mode); patchProject({ monitor_mode: b.dataset.mode }); });
}

// ------------------------------------------------------------------ steps rail
function stepDone(id) {
  const p = state.project;
  if (id === 'import') return p.sources.length > 0;
  if (id === 'analyse') return p.sources.length > 0 && p.sources.every((s) => s.analysis);
  if (id === 'lock') return !!p.vocal_lock;
  return false;
}

function renderSteps() {
  const box = $('#steps');
  box.innerHTML = '';
  STEPS.forEach((s, i) => {
    box.append(el('button', {
      class: `step${state.step === s.id ? ' on' : ''}${stepDone(s.id) ? ' done' : ''}${s.ready ? '' : ' later'}`,
      onclick: () => goStep(s.id),
    }, el('span', { class: 'n' }, i + 1), s.label, s.ready ? null : el('span', { class: 'soon' }, 'soon')));
  });
}

function goStep(id) {
  state.step = id;
  api('/project', { method: 'PATCH', body: { step: id } }).catch(() => {});
  render();
}

// ------------------------------------------------------------------ views
function render() {
  if (!state.project) return;
  renderTop();
  renderSteps();
  state.viewWaves = [];
  const view = $('#view');
  view.innerHTML = '';
  const fn = { import: viewImport, analyse: viewAnalyse, lock: viewLock }[state.step] || viewLater;
  fn(view);
}

function nextButton(step, label) {
  return el('button', { class: 'btn', onclick: () => goStep(step) }, label + ' →');
}

// ---------- 1. Import
async function importPaths(paths) {
  for (const path of paths) {
    try {
      const r = await api('/sources/import', { method: 'POST', body: { path } });
      await refresh();
      trackJob(r.source.id, r.job_id);
    } catch (e) { toast(e.message, true); }
  }
}

async function trackJob(sourceId, jobId) {
  state.jobs[sourceId] = { progress: 0, message: 'Queued' };
  render();
  try {
    await waitJob(jobId, (j) => { state.jobs[sourceId] = { progress: j.progress, message: j.message }; updateJobUi(sourceId); });
  } catch (e) { toast(e.message, true); }
  delete state.jobs[sourceId];
  await refresh();
}

function updateJobUi(sourceId) {
  const j = state.jobs[sourceId];
  const node = document.querySelector(`[data-job="${sourceId}"]`);
  if (!node || !j) return;
  node.querySelector('.progress div').style.width = (j.progress * 100) + '%';
  node.querySelector('.msg').textContent = j.message || 'Working…';
}

function jobStatus(sourceId) {
  const j = state.jobs[sourceId];
  if (!j) return null;
  return el('div', { class: 'status', 'data-job': sourceId },
    el('div', { class: 'progress' }, el('div', { style: `width:${j.progress * 100}%` })),
    el('span', { class: 'msg' }, j.message || 'Working…'));
}

function sampleFacts(s) {
  const a = s.analysis;
  const eff = { ...(a || {}), ...(s.overrides || {}) };
  if (!a) return 'Analysing…';
  const keyText = s.kind === 'beat' && !s.overrides?.key && a.key_confidence < 0.4 ? 'no clear key' : `${eff.key} (${eff.camelot})`;
  return `${eff.bpm} BPM · ${keyText} · ${fmt(s.duration)}`;
}

function viewImport(view) {
  const p = state.project;
  view.append(
    el('h1', {}, 'Import'),
    el('p', { class: 'sub' }, 'Drop in vocals, full songs or beats. GLORE guesses what each one is — tap to correct it.'),
  );
  const dz = el('div', { class: 'dropzone panel' },
    el('strong', {}, 'Drop audio files here'),
    'WAV · MP3 · FLAC · AIFF · OGG', el('br'), el('br'),
    el('button', { class: 'btn ghost', onclick: async () => importPaths(await window.glore.pickFiles()) }, 'Browse…'));
  view.append(dz);

  if (p.sources.length) {
    const list = el('div', { class: 'panel' });
    for (const s of p.sources) {
      const kinds = el('div', { class: 'row' },
        ...Object.entries(KIND_LABEL).map(([k, label]) => el('button', {
          class: `chip${k === 'vocal' ? ' vocal' : ''}${s.kind === k ? ' on' : ''}`,
          onclick: async () => {
            if (s.kind === k) return;
            try { const r = await api(`/sources/${s.id}/kind`, { method: 'POST', body: { kind: k } }); trackJob(s.id, r.job_id); } catch (e) { toast(e.message, true); }
          },
        }, label)),
        s.analysis && !s.derived_from && s.analysis.kind_guess === s.kind ? el('span', { class: 'badge' }, 'auto-detected') : null,
      );
      const actions = el('div', { class: 'row' },
        s.kind === 'full_song' && !state.jobs[s.id] ? el('button', {
          class: 'btn small', title: 'Pulls the acapella out with Demucs (GPU)',
          onclick: () => extractVocal(s),
        }, 'Extract vocal') : null,
        el('button', { class: 'btn small ghost', onclick: () => { soloOnly(`src-${s.id}`); player.play(0); } }, 'Preview'),
        el('button', { class: 'btn small danger', onclick: async () => { await api(`/sources/${s.id}`, { method: 'DELETE' }); refresh(); } }, 'Remove'),
      );
      const canvas = el('canvas');
      list.append(el('div', { class: 'sample' },
        el('div', {},
          el('div', { class: 'name' }, s.name, s.derived_from ? el('span', { class: 'badge', style: 'margin-left:8px' }, 'extracted') : null),
          el('div', { class: 'facts' }, sampleFacts(s)),
          el('div', { style: 'margin-top:8px' }, kinds),
          jobStatus(s.id)),
        actions, canvas));
      const pl = player.layers.get(`src-${s.id}`);
      if (pl) {
        const w = new Waveform(canvas, { color: s.kind === 'vocal' ? getComputedStyle(document.documentElement).getPropertyValue('--wave-vocal').trim() : undefined, onSeek: (t) => player.seek(t) });
        w.set({ buffer: pl.buffer, phrases: s.analysis?.phrases || [] });
        state.viewWaves.push(w);
      }
    }
    view.append(list, nextButton('analyse', 'Analyse'));
  }
}

async function extractVocal(s) {
  try {
    const r = await api(`/sources/${s.id}/extract-vocal`, { method: 'POST' });
    toast('Extracting vocal — first run downloads the model (~300 MB)');
    await trackJob(s.id, r.job_id);
    toast('Vocal extracted. Stems are in the layer list.');
  } catch (e) { toast(e.message, true); }
}

// ---------- 2. Analyse
function viewAnalyse(view) {
  const p = state.project;
  view.append(el('h1', {}, 'Analyse'),
    el('p', { class: 'sub' }, 'Best guesses for tempo and key. If one looks wrong, correct it — everything later uses these numbers.'));
  if (!p.sources.length) { view.append(el('div', { class: 'panel' }, 'Import a sample first.')); return; }

  view.append(el('div', { class: 'tabs' }, ...p.sources.map((s) => el('button', {
    class: `chip${s.kind === 'vocal' ? ' vocal' : ''}${state.selected === s.id ? ' on' : ''}`,
    onclick: () => { state.selected = s.id; render(); },
  }, `${s.name} · ${KIND_LABEL[s.kind]}`))));

  const s = p.sources.find((x) => x.id === state.selected);
  if (!s) return;
  const a = s.analysis;
  if (!a) { view.append(el('div', { class: 'panel' }, jobStatus(s.id) || 'Analysing…')); return; }
  const eff = { ...a, ...s.overrides };

  const canvas = el('canvas', { class: 'big-wave' });
  view.append(el('div', { class: 'panel' },
    canvas,
    el('div', { class: 'row', style: 'margin-top:10px' },
      el('button', { class: 'btn small ghost', onclick: () => { soloOnly(`src-${s.id}`); player.play(0); } }, '▶ Play this sample'),
      el('span', { class: 'dim' }, s.kind === 'vocal'
        ? `${a.phrases.length} phrases detected (shaded) · bar lines from the beat grid`
        : 'Bright lines = bar starts, faint lines = beats'))));
  const pl = player.layers.get(`src-${s.id}`);
  if (pl) {
    const w = new Waveform(canvas, { color: s.kind === 'vocal' ? getComputedStyle(document.documentElement).getPropertyValue('--wave-vocal').trim() : undefined, onSeek: (t) => player.seek(t) });
    w.set({ buffer: pl.buffer, beats: eff.beats || [], downbeats: eff.downbeats || [], phrases: a.phrases || [] });
    state.viewWaves.push(w);
  }

  const override = async (body) => {
    try { await api(`/sources/${s.id}/analysis`, { method: 'PATCH', body }); await refresh(); } catch (e) { toast(e.message, true); }
  };

  const bpmInput = el('input', { type: 'number', step: '0.01', value: eff.bpm });
  bpmInput.addEventListener('change', () => override({ bpm: Number(bpmInput.value) }));
  const keyInput = el('input', { value: eff.key, placeholder: 'e.g. A minor' });
  keyInput.addEventListener('change', () => override({ key: keyInput.value }));

  const confBar = (v) => el('div', { class: 'conf', title: `confidence ${Math.round(v * 100)}%` }, el('div', { style: `width:${v * 100}%` }));
  view.append(el('div', { class: 'grid2' },
    el('div', { class: 'panel stat' },
      el('span', { class: 'label' }, 'Tempo'),
      el('span', { class: 'big' }, `${eff.bpm}`),
      s.overrides.bpm ? el('span', { class: 'badge' }, 'set by you') : confBar(a.bpm_confidence),
      el('div', { class: 'row' }, ...a.bpm_candidates.filter((c) => c !== eff.bpm).map((c) =>
        el('button', { class: 'chip', onclick: () => override({ bpm: c }) }, `${c}`))),
      el('div', { class: 'field' }, el('span', { class: 'dim' }, 'Set:'), bpmInput),
      s.kind === 'vocal' && !s.derived_from ? el('span', { class: 'dim' }, 'Acapellas have no drums, so tempo is a harder guess. If it feels off, try ½ or 2×.') : null,
      s.derived_from ? el('span', { class: 'dim' }, "Taken from the original song's analysis (more reliable).") : null),
    el('div', { class: 'panel stat' },
      el('span', { class: 'label' }, 'Key'),
      el('span', { class: 'big' }, eff.key),
      el('span', { class: 'dim' }, `Camelot ${eff.camelot}`),
      s.overrides.key ? el('span', { class: 'badge' }, 'set by you') : confBar(a.key_confidence),
      el('div', { class: 'field' }, el('span', { class: 'dim' }, 'Set:'), keyInput),
      el('div', { class: 'row' },
        el('button', { class: 'btn small ghost', onclick: () => override({ clear: true }) }, 'Reset to detected'),
        el('button', { class: 'btn small ghost', onclick: async () => { const r = await api(`/sources/${s.id}/analyse`, { method: 'POST' }); trackJob(s.id, r.job_id); } }, 'Re-analyse')))));

  if (p.sources.some((x) => x.kind === 'vocal')) view.append(nextButton('lock', 'Vocal Lock'));
}

// ---------- 3. Vocal lock
function viewLock(view) {
  const p = state.project;
  view.append(el('h1', {}, 'Vocal Lock'),
    el('p', { class: 'sub' }, 'Your vocal is never regenerated. It is only stretched onto the project tempo — the pitch stays exactly as recorded unless you choose to shift it.'));
  const vocals = p.sources.filter((s) => s.kind === 'vocal' && s.analysis);
  if (!vocals.length) {
    view.append(el('div', { class: 'panel' }, 'No vocal yet. Import an acapella, or use "Extract vocal" on a full song in Import.'));
    return;
  }
  let src = vocals.find((v) => v.id === (state.lockSource || p.vocal_lock?.source_id)) || vocals[0];
  const eff = { ...src.analysis, ...src.overrides };
  const target = { bpm: p.bpm || eff.bpm, pitch: p.vocal_lock?.source_id === src.id ? p.vocal_lock.pitch_semitones : 0 };

  const sel = el('select', {}, ...vocals.map((v) => el('option', { value: v.id, selected: v.id === src.id }, v.name)));
  sel.addEventListener('change', () => { state.lockSource = sel.value; render(); });

  const bpmIn = el('input', { type: 'number', step: '0.01', value: target.bpm });
  const ratio = el('span', { class: 'dim' });
  const updateRatio = () => {
    const r = Number(bpmIn.value) / eff.bpm;
    ratio.textContent = `${eff.bpm} → ${bpmIn.value} BPM · ${(r * 100).toFixed(1)}% speed` + (r < 0.5 || r > 2 ? ' — too far, try ½ or 2×' : '');
  };
  bpmIn.addEventListener('input', updateRatio);
  updateRatio();

  const NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const pitch = el('input', { type: 'range', min: -12, max: 12, step: 1, value: target.pitch });
  const pitchLabel = el('span', { class: 'dim' });
  const updatePitch = () => {
    const n = Number(pitch.value);
    const newKey = `${NOTE[((eff.key_tonic + n) % 12 + 12) % 12]} ${eff.key_mode}`;
    pitchLabel.textContent = n === 0 ? `Off — stays in ${eff.key}` : `${n > 0 ? '+' : ''}${n} semitones → ${newKey}`;
  };
  pitch.addEventListener('input', updatePitch);
  updatePitch();

  const presets = BPM_PRESETS[p.genre_pack] || [];
  const presetRow = el('div', { class: 'row' },
    el('button', { class: 'chip', onclick: () => { bpmIn.value = eff.bpm; updateRatio(); } }, `Natural ${eff.bpm}`),
    ...presets.map((b) => el('button', { class: 'chip', onclick: () => { bpmIn.value = b; updateRatio(); } }, String(b))),
    el('span', { class: 'dim' }, p.genre_pack === 'melodic_rap' ? 'Melodic Rap range' : 'Emotive House range'));

  const job = state.lockJob;
  const lockBtn = el('button', {
    class: 'btn', disabled: !!job,
    onclick: async () => {
      const bpm = Number(bpmIn.value);
      try {
        if (bpm !== p.bpm) await api('/project', { method: 'PATCH', body: { bpm } });
        const r = await api('/vocal-lock', { method: 'POST', body: { source_id: src.id, target_bpm: bpm, pitch_semitones: Number(pitch.value) } });
        state.lockJob = { progress: 0, message: 'Starting' };
        render();
        await waitJob(r.job_id, (j) => { state.lockJob = { progress: j.progress, message: j.message }; });
        state.layerVersion['vocal-locked'] = Date.now();
        state.lockJob = null;
        await refresh();
        soloOnly('vocal-locked');
        toast('Vocal locked');
      } catch (e) { state.lockJob = null; render(); toast(e.message, true); }
    },
  }, job ? `${job.message}…` : (p.vocal_lock ? 'Re-lock vocal' : 'Lock vocal'));

  view.append(el('div', { class: 'panel' },
    el('div', { class: 'row' }, el('span', { class: 'dim' }, 'Vocal:'), sel, el('span', { class: 'dim' }, `${eff.key} · ${eff.bpm} BPM natural`)),
    el('div', { class: 'grid2', style: 'margin-top:16px' },
      el('div', { class: 'stat' }, el('span', { class: 'label' }, 'Target tempo'), el('div', { class: 'field' }, bpmIn), presetRow, ratio),
      el('div', { class: 'stat' }, el('span', { class: 'label' }, 'Pitch (optional)'), pitch, pitchLabel,
        el('span', { class: 'dim' }, 'Formants are preserved, so the voice keeps its character.'))),
    el('div', { class: 'row', style: 'margin-top:16px' }, lockBtn)));

  if (p.vocal_lock && player.has('vocal-locked')) {
    const canvas = el('canvas', { class: 'big-wave' });
    view.append(el('div', { class: 'panel' },
      el('div', { class: 'row', style: 'margin-bottom:10px' },
        el('strong', {}, 'Locked vocal'),
        el('span', { class: 'dim' }, `${p.vocal_lock.source_bpm} → ${p.vocal_lock.target_bpm} BPM · ${fmt(p.vocal_lock.duration)}`),
        el('div', { class: 'spacer', style: 'flex:1' }),
        el('button', { class: 'btn small ghost', onclick: () => { soloOnly(`src-${p.vocal_lock.source_id}`); player.play(0); } }, 'A · Original'),
        el('button', { class: 'btn small', onclick: () => { soloOnly('vocal-locked'); player.play(0); } }, 'B · Locked')),
      canvas));
    const w = new Waveform(canvas, { color: getComputedStyle(document.documentElement).getPropertyValue('--wave-vocal').trim(), onSeek: (t) => player.seek(t) });
    const beat = 60 / p.vocal_lock.target_bpm;
    const dur = p.vocal_lock.duration;
    const beats = []; for (let t = 0; t < dur; t += beat) beats.push(t);
    w.set({ buffer: player.layers.get('vocal-locked').buffer, phrases: p.vocal_lock.phrases, beats, downbeats: beats.filter((_, i) => i % 4 === 0) });
    state.viewWaves.push(w);
    view.append(el('p', { class: 'dim' }, 'Next session: Arrange — GLORE places these phrases into the song structure for you.'));
  }
}

// ---------- later steps
function viewLater(view) {
  const s = STEPS.find((x) => x.id === state.step);
  view.append(el('h1', {}, s.label), el('p', { class: 'sub' }, 'Coming in a later build session.'),
    el('div', { class: 'panel later-card' }, el('strong', {}, 'What this step will do'),
      el('ul', {}, ...(LATER[s.id] || []).map((t) => el('li', {}, t)))));
}

// ------------------------------------------------------------------ drag & drop anywhere
let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  const paths = [...e.dataTransfer.files].map((f) => window.glore.pathForFile(f)).filter(Boolean);
  if (paths.length) {
    if (state.step !== 'import') goStep('import');
    importPaths(paths);
  }
});

boot();
