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


const KIND_LABEL = { vocal: 'Vocal', full_song: 'Full song', beat: 'Beat' };

const state = {
  linkJobs: [],            // { id, url, progress, message }
  linkDraft: '',
  open: {},                // source id -> 'raw' | 'tag' (which drawer is open)
  rawStages: { backing: true, echo: true, reverb: true, noise: false },
  packs: [],
  packItems: {},           // pack id -> items (LoRA Packs page)
  cloud: { settings: null, status: null, open: false, draft: {}, job: null, following: {} },
  trainOpts: {},           // pack id -> { epochs, rank } typed on the card
  fit: {},                 // pack id -> last 8 GB test result
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
  await loadPacks();
  await loadCloud();
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
    const mute = el('button', { class: 'm' + (pl.muted ? ' on' : ''), 'data-tip': 'Mute', onclick: () => { player.setMute(l.id, !pl.muted); renderLayers(); } }, 'M');
    const solo = el('button', { class: 's' + (pl.solo ? ' on' : ''), 'data-tip': 'Solo', onclick: () => { player.setSolo(l.id, !pl.solo); renderLayers(); } }, 'S');
    const vol = el('input', { type: 'range', min: 0, max: 1.5, step: 0.01, value: pl.volume, 'data-tip': 'Volume' });
    vol.addEventListener('input', () => player.setVolume(l.id, Number(vol.value)));
    box.append(el('div', { class: 'layer' }, el('div', { class: 'lname', 'data-tip': l.name }, l.name), mute, solo, vol, canvas));
    const src = state.project.sources.find((x) => x.id === l.source_id);
    const isVocal = l.role === 'vocal_locked' || (l.role === 'source' && src?.kind === 'vocal') || (l.role === 'stem' && /vocals$/.test(l.id));
    const color = isVocal
      ? cssVar('--wave-vocal') : undefined;
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
  box.append(el('div', { class: 'rail-sep' }),
    el('button', { class: `step${state.step === 'packs' ? ' on' : ''}`, onclick: () => goStep('packs') },
      el('span', { class: 'n icon' }, '◆'), 'LoRA Packs'));
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
  const view = $('#view');
  const active = document.activeElement;
  if (active && view.contains(active) && active.tagName === 'INPUT' && active.type !== 'range') {
    state.renderDeferred = true;   // finish typing first; re-render on blur
    return;
  }
  state.renderDeferred = false;
  state.viewWaves = [];
  view.innerHTML = '';
  const fn = { import: viewImport, analyse: viewAnalyse, lock: viewLock, packs: viewPacks }[state.step] || viewLater;
  fn(view);
}
document.addEventListener('focusout', () => setTimeout(() => { if (state.renderDeferred) render(); }, 0));

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

const PACK_LABEL = { house: 'House', rnb_rap: 'R&B / Rap', gospel_soul: 'Gospel / Soul' };
const ORIGIN_BADGE = { link: ['link', 'Imported from a link'], extracted: ['extracted', 'Extracted from a full song'], raw: ['raw', 'Effects removed'] };

async function importLink(url) {
  url = url.trim();
  if (!url) return;
  try {
    const r = await api('/sources/import-link', { method: 'POST', body: { url } });
    const job = { id: r.job_id, url, service: r.service, progress: 0, message: 'Starting' };
    state.linkJobs.push(job);
    state.linkDraft = '';
    render();
    try {
      const res = await waitJob(r.job_id, (j) => { job.progress = j.progress; job.message = j.message; updateLinkJobUi(job); });
      if (res.matched && (r.service === 'spotify' || r.service === 'apple')) toast(`Matched on YouTube: ${res.matched}`);
    } catch (e) { toast(e.message, true); }
    state.linkJobs = state.linkJobs.filter((j) => j !== job);
    await refresh();
  } catch (e) { toast(e.message, true); }
}

function updateLinkJobUi(job) {
  const node = document.querySelector(`[data-link="${job.id}"]`);
  if (!node) return;
  node.querySelector('.progress div').style.width = (job.progress * 100) + '%';
  node.querySelector('.msg').textContent = job.message || 'Working…';
}

function viewImport(view) {
  const p = state.project;
  view.append(el('h1', { 'data-tip': 'Drop in vocals, full songs or beats, or paste a link. GLORE guesses what each one is — click a tag to correct it.' }, 'Import'));

  const link = el('input', { class: 'link-input', placeholder: 'Paste a YouTube, SoundCloud, Spotify or Apple Music link', value: state.linkDraft, spellcheck: 'false' });
  link.addEventListener('input', () => { state.linkDraft = link.value; });
  link.addEventListener('keydown', (e) => { if (e.key === 'Enter') { link.blur(); importLink(link.value); } });
  const dz = el('div', { class: 'dropzone panel' },
    el('strong', { 'data-tip': 'WAV · MP3 · FLAC · AIFF · OGG' }, 'Drop audio here'),
    el('button', { class: 'btn ghost', onclick: async () => importPaths(await window.glore.pickFiles()) }, 'Browse…'),
    el('div', { class: 'or' }, 'or'),
    el('div', { class: 'link-row', 'data-tip': 'Spotify and Apple Music are protected, so GLORE finds the same track on YouTube.' },
      link, el('button', { class: 'btn', onclick: () => { link.blur(); importLink(link.value); } }, 'Import')));
  view.append(dz);

  if (state.linkJobs.length) {
    view.append(el('div', { class: 'panel' }, ...state.linkJobs.map((j) => el('div', { class: 'link-job', 'data-link': j.id },
      el('span', { class: 'badge' }, j.service), el('span', { class: 'url' }, j.url),
      el('div', { class: 'status' }, el('div', { class: 'progress' }, el('div', { style: `width:${j.progress * 100}%` })), el('span', { class: 'msg' }, j.message))))));
  }

  if (p.sources.length) {
    const list = el('div', { class: 'panel' });
    for (const s of p.sources) list.append(sampleCard(s));
    view.append(list, nextButton('analyse', 'Analyse'));
  }
}

function sampleCard(s) {
  const auto = s.analysis && !s.derived_from && s.analysis.kind_guess === s.kind;
  const busy = !!state.jobs[s.id];
  const drawer = state.open[s.id];
  const toggle = (name) => { state.open[s.id] = drawer === name ? null : name; render(); };

  const kinds = el('div', { class: 'row' },
    ...Object.entries(KIND_LABEL).map(([k, label]) => el('button', {
      class: `chip${k === 'vocal' ? ' vocal' : ''}${s.kind === k ? ' on' : ''}`,
      'data-tip': s.kind === k && auto ? 'Auto-detected' : null,
      onclick: async () => {
        if (s.kind === k) return;
        try { const r = await api(`/sources/${s.id}/kind`, { method: 'POST', body: { kind: k } }); trackJob(s.id, r.job_id); } catch (e) { toast(e.message, true); }
      },
    }, label)),
    el('span', { class: 'vsep' }),
    el('button', {
      class: `chip tag${s.pack ? ' on' : ''}${drawer === 'tag' ? ' open' : ''}`,
      'data-tip': 'Add this track to a LoRA training set',
      onclick: () => toggle('tag'),
    }, s.pack ? `${PACK_LABEL[s.pack]}${s.moods.length ? ' · ' + s.moods.join(', ') : ''}` : '+ Train'));

  const actions = el('div', { class: 'row' },
    s.kind === 'full_song' && !busy ? el('button', {
      class: 'btn small', 'data-tip': 'Pulls the acapella out on your GPU. The other stems land in the layer list.',
      onclick: () => extractVocal(s),
    }, 'Extract vocal') : null,
    s.kind === 'vocal' && !busy ? el('button', {
      class: `btn small${drawer === 'raw' ? '' : ' ghost'}`, 'data-tip': 'Strip reverb, echo, backing vocals and noise to get a dry lead',
      onclick: () => toggle('raw'),
    }, 'Raw vocal') : null,
    el('button', { class: 'btn small ghost', onclick: () => { soloOnly(`src-${s.id}`); player.play(0); } }, 'Preview'),
    el('button', { class: 'btn small danger', onclick: async () => { await api(`/sources/${s.id}`, { method: 'DELETE' }); refresh(); } }, 'Remove'),
  );
  const badge = ORIGIN_BADGE[s.origin] || (s.derived_from ? ORIGIN_BADGE.extracted : null);
  const canvas = el('canvas');
  const card = el('div', { class: 'sample' },
    el('div', {},
      el('div', { class: 'name' }, s.name, badge ? el('span', { class: 'badge', style: 'margin-left:8px', 'data-tip': badge[1] }, badge[0]) : null),
      el('div', { class: 'facts' }, sampleFacts(s)),
      el('div', { style: 'margin-top:8px' }, kinds),
      jobStatus(s.id)),
    actions,
    drawer === 'raw' && s.kind === 'vocal' && !busy ? rawDrawer(s) : null,
    drawer === 'tag' ? tagDrawer(s) : null,
    canvas);
  const pl = player.layers.get(`src-${s.id}`);
  if (pl) {
    const w = new Waveform(canvas, { color: s.kind === 'vocal' ? cssVar('--wave-vocal') : undefined, onSeek: (t) => player.seek(t) });
    w.set({ buffer: pl.buffer, phrases: s.analysis?.phrases || [] });
    state.viewWaves.push(w);
  }
  return card;
}

const RAW_STAGES = [
  ['backing', 'Backing vocals', 'Keeps the lead, removes doubles, harmonies and ad-lib layers'],
  ['echo', 'Echo / delay', 'Removes delay throws and echoes'],
  ['reverb', 'Reverb', 'Removes room, plate and hall reverb'],
  ['noise', 'Noise', 'Removes hiss and leftover bleed from extraction'],
];

function rawDrawer(s) {
  const go = el('button', {
    class: 'btn small',
    onclick: async () => {
      const stages = Object.entries(state.rawStages).filter(([, on]) => on).map(([k]) => k);
      if (!stages.length) { toast('Pick at least one thing to remove', true); return; }
      try {
        const r = await api(`/sources/${s.id}/raw-vocal`, { method: 'POST', body: { stages } });
        state.open[s.id] = null;
        await trackJob(s.id, r.job_id);
        toast('Raw vocal ready');
      } catch (e) { toast(e.message, true); }
    },
  }, 'Make raw');
  return el('div', { class: 'drawer' },
    el('span', { class: 'label', 'data-tip': "Autotune can't be undone — it changes the notes themselves." }, 'Remove'),
    el('div', { class: 'row' }, ...RAW_STAGES.map(([id, label, tip]) => el('button', {
      class: `chip${state.rawStages[id] ? ' on' : ''}`, 'data-tip': tip,
      onclick: () => { state.rawStages[id] = !state.rawStages[id]; render(); },
    }, label))),
    go);
}

function tagDrawer(s) {
  const pack = s.pack;
  const info = state.packs.find((p) => p.id === pack);
  const save = async (newPack, moods) => {
    try {
      const r = await api(`/sources/${s.id}/tag`, { method: 'POST', body: { pack: newPack, moods } });
      state.packs = r.packs;
      await refresh();
    } catch (e) { toast(e.message, true); }
  };
  const packRow = el('div', { class: 'row' },
    ...Object.entries(PACK_LABEL).map(([id, label]) => el('button', {
      class: `chip${pack === id ? ' on' : ''}`,
      onclick: () => save(pack === id ? null : id, pack === id ? [] : s.moods),
    }, label)));
  const parts = [el('span', { class: 'label', 'data-tip': 'Tagged tracks build that pack\'s LoRA. Mood tags become the training captions, so the prompt can pick a vibe later.' }, 'Training set'), packRow];
  if (pack && info) {
    const moods = [...new Set([...info.moods, ...s.moods])];
    const custom = el('input', { class: 'mood-input', placeholder: '+ mood' });
    custom.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && custom.value.trim()) { const v = custom.value.trim(); custom.value = ''; custom.blur(); save(pack, [...s.moods, v]); }
    });
    parts.push(el('span', { class: 'label' }, 'Mood'),
      el('div', { class: 'row' }, ...moods.map((m) => el('button', {
        class: `chip small${s.moods.includes(m) ? ' on' : ''}`,
        onclick: () => save(pack, s.moods.includes(m) ? s.moods.filter((x) => x !== m) : [...s.moods, m]),
      }, m)), custom));
  }
  return el('div', { class: 'drawer' }, ...parts);
}

// ---------- Cloud GPU (RunPod) + LoRA training
const money = (v) => `$${Number(v || 0).toFixed(2)}`;
const mins = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m` : `${Math.max(0, Math.floor(s / 60))} min`);
const defaultEpochs = (n) => (n <= 10 ? 300 : n <= 50 ? 150 : 80);

async function loadCloud() {
  try {
    state.cloud.settings = await api('/cloud/settings');
    state.cloud.status = await api('/cloud/status');
    for (const [key, jobId] of Object.entries(state.cloud.status.jobs || {})) {
      if (!state.cloud.following[key]) followCloudJob(key, jobId);   // picked up again after a restart
    }
  } catch (e) { toast(e.message, true); }
}

// Keep cost and pod state live while you're on the LoRA Packs page or something is running
setInterval(async () => {
  const busy = Object.keys(state.cloud.following).length || (state.cloud.status?.active || []).length;
  if (!state.project || !(state.step === 'packs' || busy)) return;
  try { state.cloud.status = await api('/cloud/status'); } catch (_) { return; }
  updateCloudPanel();
}, 5000);

function updateCloudPanel() {
  const cur = document.getElementById('cloud-panel');
  if (!cur || cur.contains(document.activeElement)) return;   // don't disturb typing
  cur.replaceWith(cloudPanel());
}

async function followCloudJob(key, jobId) {
  state.cloud.following[key] = jobId;
  state.packJobs = state.packJobs || {};
  const pack = key.includes(':') ? key.split(':')[1] : null;
  const set = (j) => {
    if (pack) { state.packJobs[pack] = { progress: j.progress, message: j.message || '' }; updatePackJob(pack); }
    else { state.cloud.job = { progress: j.progress, message: j.message || '' }; updateCloudPanel(); }
  };
  set({ progress: 0, message: 'Starting' });
  if (pack) render(); else updateCloudPanel();
  try {
    return await waitJob(jobId, set);
  } finally {
    if (pack) delete state.packJobs[pack]; else state.cloud.job = null;
    delete state.cloud.following[key];
    await loadPacks();
    await loadCloud();
    render();
  }
}

function field(label, tip, input) {
  return el('label', { class: 'field' }, el('span', { class: 'label', 'data-tip': tip }, label), input);
}

function settingInput(key, attrs = {}) {
  const s = state.cloud.settings || {};
  const d = state.cloud.draft;
  const value = key in d ? d[key] : (s[key] ?? '');
  return el('input', {
    class: 'field-input' + (attrs.type === 'number' ? ' mono' : ''), spellcheck: 'false', ...attrs, value,
    oninput: (e) => { d[key] = e.target.value; },
  });
}

function settingSelect(key, options) {
  const s = state.cloud.settings || {};
  const d = state.cloud.draft;
  const value = key in d ? d[key] : (s[key] ?? '');
  const sel = el('select', { class: 'field-input', onchange: (e) => { d[key] = e.target.value; } },
    ...options.map(([v, label]) => el('option', { value: v }, label)));
  sel.value = value;
  return sel;
}

async function saveCloudSettings() {
  const body = {};
  for (const [k, v] of Object.entries(state.cloud.draft)) {
    if (['volume_gb', 'idle_minutes', 'hard_cap_hours', 'monthly_limit'].includes(k)) { if (v !== '') body[k] = Number(v); }
    else body[k] = v;
  }
  state.cloud.settings = await api('/cloud/settings', { method: 'PUT', body });
  state.cloud.draft = {};
  state.cloud.status = await api('/cloud/status');
}

function cloudSettingsForm() {
  const s = state.cloud.settings;
  const dcs = [['', 'Auto (EU first)'], ...s.s3_datacenters.map((d) => [d, d])];
  const save = async () => {
    try { await saveCloudSettings(); toast('Saved'); if (!state.cloud.settings.missing.length) state.cloud.open = false; render(); }
    catch (e) { toast(e.message, true); }
  };
  const createVolume = async () => {
    try {
      await saveCloudSettings();
      const r = await api('/cloud/volume', { method: 'POST' });
      await followCloudJob('volume', r.job_id);
      toast('Network volume created');
    } catch (e) { toast(e.message, true); }
  };
  return el('div', { class: 'cloud-form' },
    el('div', { class: 'fields' },
      field('RunPod API key', 'RunPod console → Settings → API Keys. Starts and stops the GPU.', settingInput('api_key', { type: 'password', placeholder: 'rpa_…' })),
      field('S3 access key', 'Credentials → S3 API Keys: your user id (user_…), shown with the key. Lets GLORE upload without a running GPU.', settingInput('s3_access_key', { placeholder: 'user_…' })),
      field('S3 secret', 'The secret shown once when you create the S3 API key.', settingInput('s3_secret', { type: 'password' })),
      field('Datacenter', 'Must support RunPod\'s S3 API. Auto picks the one with the most H100s free, EU first.', settingSelect('datacenter', dcs)),
      field('Network volume', 'Where training audio, models and LoRAs live between runs. Paste an existing volume id, or create one.',
        el('div', { class: 'row tight' }, settingInput('volume_id', { placeholder: 'none yet' }),
          s.volume_id ? null : el('button', { class: 'btn small ghost', disabled: !!state.cloud.job, onclick: createVolume, 'data-tip': 'Creates it in the datacenter above. Storage is billed monthly, about $0.07 per GB.' }, 'Create'))),
      field('Volume GB', 'Size for a new volume. 50 GB holds ACE-Step, its models and all three packs.', settingInput('volume_gb', { type: 'number', min: 10, max: 4096 })),
      field('Idle stop · min', 'The GPU shuts itself down after this long with nothing to do.', settingInput('idle_minutes', { type: 'number', min: 1, max: 120 })),
      field('Hard cap · h', 'No GPU session ever runs longer than this, whatever happens. A run cut short resumes from its last checkpoint.', settingInput('hard_cap_hours', { type: 'number', min: 0.5, max: 48, step: 0.5 })),
      field('Monthly limit · $', 'GLORE won\'t start a run that could go over this, and stops the GPU if it\'s reached. 0 = no limit.', settingInput('monthly_limit', { type: 'number', min: 0, step: 1 })),
      field('GPU', 'Picked by name: the cheapest one in stock whose name contains this.', settingInput('gpu', { placeholder: 'H100' })),
      field('Base model', 'ACE-Step 1.5 standard (2B). LoRAs only work on the model they were trained on: turbo is what runs on your 8 GB card.',
        settingSelect('variant', [['turbo', 'Standard · turbo'], ['base', 'Standard · base'], ['sft', 'Standard · sft']]))),
    el('div', { class: 'row' }, el('button', { class: 'btn small', onclick: save }, 'Save')));
}

function cloudPanel() {
  const c = state.cloud;
  const s = c.settings;
  if (!s) return el('div', { class: 'panel cloud', id: 'cloud-panel' }, el('span', { class: 'dim small' }, 'Cloud GPU…'));
  const st = c.status || { active: [], month_spend: 0, monthly_limit: 0 };
  const ready = !s.missing.length;
  const active = st.active || [];
  const live = active.reduce((a, r) => a + r.cost_so_far, 0);
  const chip = !ready ? el('span', { class: 'badge warn' }, 'not set up')
    : active.length ? el('span', { class: 'badge live' }, `${active.length} running · ${money(live)}`)
      : el('span', { class: 'badge', 'data-tip': `Volume ${s.volume_id}` }, `ready · ${s.datacenter}`);
  const test = async () => {
    try {
      const r = await api('/cloud/test', { method: 'POST' });
      const res = await followCloudJob('test', r.job_id);
      toast(`Cloud OK · ${res.gpu} · ${res.seconds} s · ${money(res.cost)}`);
    } catch (e) { toast(e.message, true); }
  };
  const stop = async () => {
    if (!confirm('Stop every GLORE cloud GPU now? A training run in progress keeps its last checkpoint.')) return;
    try { const r = await api('/cloud/stop', { method: 'POST' }); toast(`Stopped ${r.stopped}`); await loadCloud(); updateCloudPanel(); }
    catch (e) { toast(e.message, true); }
  };
  return el('div', { class: 'panel cloud', id: 'cloud-panel' },
    el('div', { class: 'cloud-head' },
      el('strong', {}, 'Cloud GPU'), chip,
      el('span', { class: 'mono small dim', 'data-tip': 'GLORE GPU time this month (storage is billed separately by RunPod)' },
        `${money(st.month_spend)}${st.monthly_limit ? ` / ${money(st.monthly_limit)}` : ''} this month`),
      el('div', { class: 'spacer' }),
      c.job ? el('div', { class: 'status' }, el('div', { class: 'progress' }, el('div', { style: `width:${c.job.progress * 100}%` })), el('span', { class: 'msg' }, c.job.message)) : null,
      active.length ? el('button', { class: 'btn small danger', onclick: stop, 'data-tip': 'Shut down every running GLORE pod now' }, 'Stop') : null,
      el('button', { class: 'btn small ghost', disabled: !ready || !!c.job, onclick: test, 'data-tip': 'Starts the cheapest GPU for about 2 minutes to check keys, storage and the pod script. Costs about a cent.' }, 'Test'),
      el('button', { class: `btn small ghost${c.open ? ' on' : ''}`, onclick: () => { c.open = !c.open; render(); } }, 'Settings')),
    ...active.map((r) => el('div', { class: 'pod-row mono small' },
      el('span', {}, r.kind === 'test' ? 'test' : (r.pack || r.kind)),
      el('span', { class: 'dim' }, r.gpu),
      el('span', {}, mins(r.runtime_s)),
      el('span', {}, money(r.cost_so_far)),
      el('span', { class: 'dim', 'data-tip': 'Time left before the hard cap stops it' }, `cap ${mins(Math.max(0, r.cap_left_s))}`))),
    (c.open || !ready) ? cloudSettingsForm() : null);
}

function trainRow(p, busy) {
  const o = state.trainOpts[p.id] = state.trainOpts[p.id] || {};
  const ready = state.cloud.settings && !state.cloud.settings.missing.length;
  const enough = p.prepared >= p.min_tracks;
  const tip = !ready ? 'Set up the cloud GPU above first' : !enough ? `Prepare at least ${p.min_tracks} tracks first` : 'Upload the prepared tracks and train on a cloud H100';
  return el('div', { class: 'row train-row' },
    el('span', { class: 'label', 'data-tip': `Passes over the whole set. Empty = picked from the track count (${defaultEpochs(p.prepared)} for ${p.prepared}).` }, 'epochs'),
    el('input', { class: 'num sm', type: 'number', min: 1, max: 2000, placeholder: String(defaultEpochs(p.prepared)), value: o.epochs || '', oninput: (e) => { o.epochs = e.target.value; } }),
    el('span', { class: 'label', 'data-tip': 'LoRA rank. 64 is the usual; 128 learns more detail, 16–32 is lighter. 8, 16, 32, 64 or 128.' }, 'rank'),
    el('input', { class: 'num sm', type: 'number', min: 8, max: 128, placeholder: '64', value: o.rank || '', oninput: (e) => { o.rank = e.target.value; } }),
    el('button', { class: 'btn small ghost', disabled: busy || p.prepared < 3, onclick: () => fitTest(p), 'data-tip': 'Trains on 3 tracks on this PC for 5 minutes with low-VRAM settings and measures VRAM. First run downloads ACE-Step and its models (~13 GB).' }, '8 GB test'),
    el('button', { class: 'btn small', disabled: busy || !ready || !enough, onclick: () => trainPack(p), 'data-tip': tip }, 'Train LoRA'));
}

function loraLine(p) {
  const l = (p.loras || [])[0];
  if (!l) return null;
  const when = new Date(l.trained_at * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
  return el('div', { class: 'lora-line small', 'data-tip': l.path },
    el('span', { class: 'ok-dot' }), `LoRA · ${when} · ${l.tracks} tracks · ${l.epochs} epochs`,
    l.loss != null ? el('span', { class: 'mono dim' }, ` · loss ${Number(l.loss).toFixed(3)}`) : null,
    p.loras.length > 1 ? el('span', { class: 'dim' }, ` · +${p.loras.length - 1} older`) : null);
}

function fitLine(id) {
  const r = state.fit[id];
  if (!r) return null;
  const verdict = r.fits === 'yes' ? 'Fits on this GPU' : r.fits === 'no' ? 'Out of memory on this GPU' : 'No result';
  return el('div', { class: `fit-line small${r.fits === 'no' ? ' bad' : ''}`, 'data-tip': `Rank 16, 8-bit optimiser, encoder offloaded. Log: ${r.log}` },
    verdict, el('span', { class: 'mono dim' },
      ` · ${r.peak_vram_gb} / ${r.total_vram_gb} GB`,
      r.sec_per_step ? ` · ${r.sec_per_step} s/step` : '',
      r.full_run_hours_here ? ` · full run here ~${r.full_run_hours_here} h` : ''));
}

async function trainPack(p) {
  const o = state.trainOpts[p.id] || {};
  try {
    const q = await api('/cloud/quote');
    const lim = q.monthly_limit ? ` of ${money(q.monthly_limit)}` : '';
    if (!confirm(`Train ${p.label} on ${q.gpu} (${money(q.price_per_hr)}/h)?\n\nAt most ${money(q.max_cost)} (${q.hard_cap_hours} h cap). This month so far: ${money(q.month_spend)}${lim}.`)) return;
    const body = {};
    if (o.epochs) body.epochs = Number(o.epochs);
    if (o.rank) body.rank = Number(o.rank);
    const r = await api(`/datasets/${p.id}/train`, { method: 'POST', body });
    const info = await followCloudJob(`train:${p.id}`, r.job_id);
    toast(`${p.label} LoRA ready · ${info.epochs} epochs`);
  } catch (e) { toast(e.message, true); }
}

async function fitTest(p) {
  if (!confirm('Run the 8 GB test on this PC?\n\nAbout 5 minutes of training on 3 tracks. The first time it also installs ACE-Step and downloads its models (~13 GB).')) return;
  try {
    const r = await api(`/datasets/${p.id}/fit-test`, { method: 'POST' });
    const res = await followCloudJob(`fit:${p.id}`, r.job_id);
    state.fit[p.id] = res;
    render();
  } catch (e) { toast(e.message, true); }
}

// ---------- LoRA Packs page
async function loadPacks() {
  try {
    state.packs = await api('/packs');
    for (const p of state.packs) state.packItems[p.id] = await api(`/datasets/${p.id}`);
  } catch (e) { toast(e.message, true); }
}

function viewPacks(view) {
  view.append(el('h1', { 'data-tip': 'One LoRA per pack, shared across all your projects. Tag tracks on the Import page to add them.' }, 'LoRA Packs'));
  if (!state.packs.length) { loadPacks().then(() => { if (state.step === 'packs') render(); }); return; }
  const grid = el('div', { class: 'packs' });
  for (const p of state.packs) {
    const items = state.packItems[p.id] || [];
    const pct = Math.min(100, (p.count / p.min_tracks) * 100);
    const job = state.packJobs?.[p.id];
    grid.append(el('div', { class: 'panel pack', 'data-pack': p.id },
      el('div', { class: 'pack-head' },
        el('strong', {}, p.label),
        el('span', { class: 'badge', 'data-tip': p.training_audio === 'keep_backing' ? 'Lead vocal removed, choirs and backing vocals kept' : 'All vocals removed before training' },
          p.training_audio === 'keep_backing' ? 'keeps backing vox' : 'instrumental')),
      el('div', { class: 'pack-count mono', 'data-tip': p.count >= p.min_tracks ? 'Enough to train. More is fine: 50–150 is the sweet spot.' : `At least ${p.min_tracks} tracks needed to train` },
        `${p.count}`, el('span', { class: 'dim' }, p.count >= p.min_tracks ? ' tracks · ready' : ` / ${p.min_tracks}`)),
      el('div', { class: 'progress wide', 'data-tip': `${p.prepared} of ${p.count} prepared for training` }, el('div', { style: `width:${pct}%` })),
      el('div', { class: 'pack-items' }, ...(items.length ? items.map((it) => el('div', { class: 'pack-item' },
        el('span', { class: 'pi-name', 'data-tip': it.caption }, it.name),
        it.prepared ? el('span', { class: 'ok-dot', 'data-tip': 'Prepared' }) : null,
        el('button', { class: 'x', 'data-tip': 'Remove from this pack', onclick: async () => { state.packs = await api(`/datasets/${p.id}/${it.id}`, { method: 'DELETE' }); await loadPacks(); await refresh(); } }, '×')))
        : [el('div', { class: 'dim small' }, 'No tracks yet')])),
      el('div', { class: 'row' },
        job ? el('div', { class: 'status' }, el('div', { class: 'progress' }, el('div', { style: `width:${job.progress * 100}%` })), el('span', { class: 'msg' }, job.message))
          : el('button', {
            class: 'btn small ghost', disabled: p.prepared >= p.count,
            'data-tip': 'Strip vocals from the tracks so they\'re ready to train on',
            onclick: () => preparePack(p.id),
          }, 'Prepare'),
        job ? null : el('button', {
          class: 'btn small ghost', 'data-tip': 'Download every ticked track from a track-list CSV (export it from the sheet). Tracks already in the pack are skipped.',
          onclick: () => importTrackList(p.id),
        }, 'Import list'),
        null),
      trainRow(p, !!job),
      loraLine(p),
      fitLine(p.id),
      listReport(p.id)));
  }
  view.append(cloudPanel(), grid);
}

async function importTrackList(id) {
  const path = await window.glore.pickCsv();
  if (!path) return;
  state.packJobs = state.packJobs || {};
  state.listReports = state.listReports || {};
  try {
    const r = await api(`/datasets/${id}/import-list`, { method: 'POST', body: { csv_path: path } });
    state.packJobs[id] = { progress: 0, message: `Starting ${r.tracks} tracks` };
    render();
    const res = await waitJob(r.job_id, (j) => { state.packJobs[id] = { progress: j.progress, message: j.message }; updatePackJob(id); });
    state.listReports[id] = res;
    toast(`${res.added.length} added · ${res.skipped.length} already there · ${res.failed.length} failed`, res.failed.length > 0);
  } catch (e) { toast(e.message, true); }
  delete state.packJobs[id];
  await loadPacks();
  render();
}

function updatePackJob(id) {
  const j = state.packJobs?.[id];
  const card = document.querySelector(`[data-pack="${id}"]`);
  if (!j || !card) { render(); return; }
  const bar = card.querySelector('.status .progress div');
  const msg = card.querySelector('.status .msg');
  if (bar) bar.style.width = (j.progress * 100) + '%';
  if (msg) msg.textContent = j.message;
}

function listReport(id) {
  const rep = state.listReports?.[id];
  if (!rep || !rep.failed.length) return null;
  return el('div', { class: 'list-report' },
    el('span', { class: 'label', 'data-tip': 'Import these by hand with a link on the Import page, or fix the name in the sheet and run the list again.' }, `${rep.failed.length} not found`),
    ...rep.failed.map((f) => el('div', { class: 'fail', 'data-tip': f.reason }, f.name)));
}

async function preparePack(id) {
  state.packJobs = state.packJobs || {};
  try {
    const r = await api(`/datasets/${id}/prepare`, { method: 'POST' });
    state.packJobs[id] = { progress: 0, message: 'Starting' };
    render();
    await waitJob(r.job_id, (j) => { state.packJobs[id] = { progress: j.progress, message: j.message }; updatePackJob(id); });
  } catch (e) { toast(e.message, true); }
  delete state.packJobs[id];
  await loadPacks();
  render();
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

async function extractVocal(s) {
  try {
    const r = await api(`/sources/${s.id}/extract-vocal`, { method: 'POST' });
    await trackJob(s.id, r.job_id);
    toast('Vocal extracted');
  } catch (e) { toast(e.message, true); }
}

// ---------- 2. Analyse
function viewAnalyse(view) {
  const p = state.project;
  view.append(el('h1', { 'data-tip': 'Best guesses for tempo and key. Correct anything that looks wrong — every later step uses these numbers.' }, 'Analyse'));
  if (!p.sources.length) { view.append(el('div', { class: 'panel dim' }, 'Import a sample first.')); return; }

  view.append(el('div', { class: 'tabs' }, ...p.sources.map((s) => el('button', {
    class: `chip${s.kind === 'vocal' ? ' vocal' : ''}${state.selected === s.id ? ' on' : ''}`,
    onclick: () => { state.selected = s.id; render(); },
  }, `${s.name} · ${KIND_LABEL[s.kind]}`))));

  const s = p.sources.find((x) => x.id === state.selected);
  if (!s) return;
  const a = s.analysis;
  if (!a) { view.append(el('div', { class: 'panel' }, jobStatus(s.id) || 'Analysing…')); return; }
  const eff = { ...a, ...s.overrides };

  const canvas = el('canvas', {
    class: 'big-wave',
    'data-tip': s.kind === 'vocal'
      ? `${a.phrases.length} phrases (shaded). Bright lines = bars, faint = beats. Click to jump.`
      : 'Bright lines = bars, faint = beats. Click to jump.',
  });
  view.append(el('div', { class: 'panel' }, canvas,
    el('div', { class: 'row', style: 'margin-top:10px' },
      el('button', { class: 'btn small ghost', onclick: () => { soloOnly(`src-${s.id}`); player.play(0); } }, '▶ Play'))));
  const pl = player.layers.get(`src-${s.id}`);
  if (pl) {
    const w = new Waveform(canvas, { color: s.kind === 'vocal' ? cssVar('--wave-vocal') : undefined, onSeek: (t) => player.seek(t) });
    w.set({ buffer: pl.buffer, beats: eff.beats || [], downbeats: eff.downbeats || [], phrases: a.phrases || [] });
    state.viewWaves.push(w);
  }

  const override = async (body) => {
    try { await api(`/sources/${s.id}/analysis`, { method: 'PATCH', body }); await refresh(); } catch (e) { toast(e.message, true); }
  };

  let tempoTip = 'Type a BPM to override.';
  if (s.derived_from) tempoTip = "Taken from the original song (more reliable than the acapella). Type to override.";
  else if (s.kind === 'vocal') tempoTip = 'Acapellas have no drums, so tempo is harder to read. If it feels off, try half or double. Type to override.';
  const bpmInput = el('input', { type: 'number', step: '0.01', value: eff.bpm, class: 'big-input', 'data-tip': tempoTip });
  bpmInput.addEventListener('change', () => override({ bpm: Number(bpmInput.value) }));
  const keyInput = el('input', { value: eff.key, class: 'big-input', 'data-tip': 'Type a key to override, e.g. "A minor", "F#", "Bbm".' });
  keyInput.addEventListener('change', () => override({ key: keyInput.value }));

  const confBar = (v, isSet) => isSet
    ? el('span', { class: 'badge' }, 'set by you')
    : el('div', { class: 'conf', 'data-tip': `Detection confidence: ${Math.round(v * 100)}%` }, el('div', { style: `width:${v * 100}%` }));
  view.append(el('div', { class: 'grid2' },
    el('div', { class: 'panel stat' },
      el('span', { class: 'label' }, 'Tempo'), bpmInput, confBar(a.bpm_confidence, !!s.overrides.bpm)),
    el('div', { class: 'panel stat' },
      el('span', { class: 'label' }, 'Key'), keyInput,
      el('span', { class: 'dim', 'data-tip': 'Camelot wheel code — neighbouring numbers mix well together.' }, `Camelot ${eff.camelot}`),
      confBar(a.key_confidence, !!s.overrides.key))));
  view.append(el('div', { class: 'row' },
    p.sources.some((x) => x.kind === 'vocal') ? nextButton('lock', 'Vocal Lock') : null,
    el('button', { class: 'btn ghost', onclick: () => override({ clear: true }), 'data-tip': 'Undo your tempo/key edits for this sample' }, 'Reset'),
    el('button', { class: 'btn ghost', onclick: async () => { const r = await api(`/sources/${s.id}/analyse`, { method: 'POST' }); trackJob(s.id, r.job_id); } }, 'Re-analyse')));
}

// ---------- 3. Vocal lock
function viewLock(view) {
  const p = state.project;
  view.append(el('h1', { 'data-tip': 'Your vocal is never regenerated — only stretched onto the project tempo. Pitch stays as recorded unless you shift it.' }, 'Vocal Lock'));
  const vocals = p.sources.filter((s) => s.kind === 'vocal' && s.analysis);
  if (!vocals.length) {
    view.append(el('div', { class: 'panel dim' }, 'No vocal yet — import an acapella or extract one from a full song.'));
    return;
  }
  const src = vocals.find((v) => v.id === (state.lockSource || p.vocal_lock?.source_id)) || vocals[0];
  const eff = { ...src.analysis, ...src.overrides };
  const target = { bpm: p.bpm || eff.bpm, pitch: p.vocal_lock?.source_id === src.id ? p.vocal_lock.pitch_semitones : 0 };

  const sel = el('select', {}, ...vocals.map((v) => el('option', { value: v.id, selected: v.id === src.id }, v.name)));
  sel.addEventListener('change', () => { state.lockSource = sel.value; render(); });

  const bpmIn = el('input', { type: 'number', step: '0.01', value: target.bpm, class: 'big-input' });
  const warn = el('span', { class: 'warn' });
  const updateRatio = () => {
    const r = Number(bpmIn.value) / eff.bpm;
    bpmIn.dataset.tip = `${eff.bpm} → ${bpmIn.value} BPM (${(r * 100).toFixed(1)}% speed)`;
    warn.textContent = r < 0.5 || r > 2 ? 'Too big a jump — try half or double' : '';
  };
  bpmIn.addEventListener('input', updateRatio);
  updateRatio();

  const NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const pitch = el('input', { type: 'range', min: -12, max: 12, step: 1, value: target.pitch, 'data-tip': 'Formants are preserved, so the voice keeps its character.' });
  const pitchLabel = el('span', { class: 'dim mono' });
  const updatePitch = () => {
    const n = Number(pitch.value);
    const newKey = `${NOTE[((eff.key_tonic + n) % 12 + 12) % 12]} ${eff.key_mode}`;
    pitchLabel.textContent = n === 0 ? `0 · ${eff.key}` : `${n > 0 ? '+' : ''}${n} · ${newKey}`;
  };
  pitch.addEventListener('input', updatePitch);
  updatePitch();

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
  }, job ? `${job.message}…` : (p.vocal_lock ? 'Re-lock' : 'Lock vocal'));

  view.append(el('div', { class: 'panel' },
    el('div', { class: 'row' }, sel, el('span', { class: 'dim mono' }, `${eff.key} · ${eff.bpm} BPM`)),
    el('div', { class: 'grid2', style: 'margin-top:16px' },
      el('div', { class: 'stat' }, el('span', { class: 'label' }, 'Target BPM'), bpmIn, warn),
      el('div', { class: 'stat' }, el('span', { class: 'label' }, 'Pitch'), pitch, pitchLabel)),
    el('div', { class: 'row', style: 'margin-top:16px' }, lockBtn)));

  if (p.vocal_lock && player.has('vocal-locked')) {
    const canvas = el('canvas', { class: 'big-wave' });
    view.append(el('div', { class: 'panel' },
      el('div', { class: 'row', style: 'margin-bottom:10px' },
        el('strong', {}, 'Locked'),
        el('span', { class: 'dim mono' }, `${p.vocal_lock.source_bpm} → ${p.vocal_lock.target_bpm} BPM · ${fmt(p.vocal_lock.duration)}`),
        el('div', { style: 'flex:1' }),
        el('button', { class: 'btn small ghost', 'data-tip': 'Play the original', onclick: () => { soloOnly(`src-${p.vocal_lock.source_id}`); player.play(0); } }, 'A'),
        el('button', { class: 'btn small', 'data-tip': 'Play the locked version', onclick: () => { soloOnly('vocal-locked'); player.play(0); } }, 'B')),
      canvas));
    const w = new Waveform(canvas, { color: cssVar('--wave-vocal'), onSeek: (t) => player.seek(t) });
    const beat = 60 / p.vocal_lock.target_bpm;
    const beats = []; for (let t = 0; t < p.vocal_lock.duration; t += beat) beats.push(t);
    w.set({ buffer: player.layers.get('vocal-locked').buffer, phrases: p.vocal_lock.phrases, beats, downbeats: beats.filter((_, i) => i % 4 === 0) });
    state.viewWaves.push(w);
  }
}

// ---------- later steps
function viewLater(view) {
  const s = STEPS.find((x) => x.id === state.step);
  view.append(el('h1', {}, s.label),
    el('div', { class: 'panel later-card' }, el('span', { class: 'badge' }, 'Coming soon'),
      el('ul', {}, ...(LATER[s.id] || []).map((t) => el('li', {}, t)))));
}

// ------------------------------------------------------------------ hover tips
// Any element with data-tip gets a small popup on hover instead of on-screen text.
const tipEl = el('div', { class: 'tip hidden' });
document.body.append(tipEl);
let tipTimer = null;
let tipTarget = null;
document.addEventListener('mouseover', (e) => {
  const t = e.target.closest('[data-tip]');
  if (t === tipTarget) return;
  tipTarget = t;
  clearTimeout(tipTimer);
  tipEl.classList.add('hidden');
  if (!t || !t.dataset.tip) return;
  tipTimer = setTimeout(() => {
    tipEl.textContent = t.dataset.tip;
    tipEl.classList.remove('hidden');
    const r = t.getBoundingClientRect();
    const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
    let x = Math.min(Math.max(8, r.left + r.width / 2 - tw / 2), window.innerWidth - tw - 8);
    let y = r.bottom + 8;
    if (y + th > window.innerHeight - 8) y = r.top - th - 8;
    tipEl.style.left = `${x}px`;
    tipEl.style.top = `${y}px`;
  }, 400);
});
document.addEventListener('mousedown', () => { clearTimeout(tipTimer); tipEl.classList.add('hidden'); });

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
