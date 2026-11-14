/* Unlockurr shared module: storage, PIN hashing, face models, avatars. */
const UL = (() => {
  const KEY = 'unlockurr.v1';
  const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/';
  const defaults = () => ({ room: '1208', users: [], log: [] });

  function load() {
    try {
      const s = JSON.parse(localStorage.getItem(KEY));
      return s ? { ...defaults(), ...s } : defaults();
    } catch { return defaults(); }
  }
  function save(s) {
    try { localStorage.setItem(KEY, JSON.stringify(s)); return true; }
    catch (e) { console.error('Unlockurr: save failed', e); alert('Could not save — browser storage is full or blocked.'); return false; }
  }
  function addLog(entry) {
    const s = load();
    s.log.unshift({ t: Date.now(), ...entry });
    s.log = s.log.slice(0, 100);
    save(s);
  }

  async function hashPin(pin) {
    const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode('unlockurr:' + pin));
    return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
  }
  const validPin = p => /^\d{4,8}$/.test(p);
  const uid = () => 'g_' + Math.random().toString(36).slice(2, 10);
  const fullName = u => `${u.first || ''} ${u.last || ''}`.trim() || 'User';

  /* Access period: { type: 'permanent' } or { type: 'scheduled', start: ms, end: ms }. Missing = permanent. */
  function accessStatus(u, now = Date.now()) {
    const a = u.access;
    if (!a || a.type !== 'scheduled') return { active: true, state: 'permanent' };
    if (a.start && now < a.start) return { active: false, state: 'upcoming' };
    if (a.end && now > a.end) return { active: false, state: 'expired' };
    return { active: true, state: 'active' };
  }
  const fmtWhen = ms => new Date(ms).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  const fmtDay = ms => new Date(ms).toLocaleDateString([], { month: 'short', day: 'numeric' });
  function accessLabel(u, now = Date.now(), short = false) {
    const st = accessStatus(u, now).state, a = u.access || {}, f = short ? fmtDay : fmtWhen;
    if (st === 'permanent') return 'Permanent';
    if (st === 'upcoming') return `Starts ${f(a.start)}`;
    if (st === 'expired') return `Expired ${f(a.end)}`;
    return `Until ${f(a.end)}`;
  }
  const initials = u => (((u.first || '')[0] || '') + ((u.last || '')[0] || '')).toUpperCase() || '?';

  const PALETTE = ['#2f6fed', '#0f9d76', '#c2410c', '#9333ea', '#be185d', '#0e7490', '#4d7c0f', '#b45309'];
  function colorFor(u) {
    let h = 0; for (const ch of fullName(u)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return PALETTE[h % PALETTE.length];
  }
  function avatar(u, cls = '') {
    const el = document.createElement('div');
    el.className = 'avatar ' + cls;
    if (u.pfp) {
      const img = new Image(); img.src = u.pfp; img.alt = fullName(u); el.appendChild(img);
    } else {
      el.textContent = initials(u);
      el.style.background = colorFor(u);
    }
    return el;
  }

  /* ---------- face models ---------- */
  let ready;
  function loadFace() {
    if (!ready) ready = (async () => {
      if (!window.faceapi) throw new Error('face-api.js failed to load');
      // reuse compiled GPU programs across input sizes instead of compiling a new one per shape (big first-use stalls)
      try { faceapi.tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true); } catch {}
      try { await faceapi.tf.setBackend('webgl'); } catch {}
      await faceapi.tf.ready();
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);
    })();
    ready.catch(() => { ready = null; });
    return ready;
  }
  // run every network once so their GPU programs compile now (while nothing is animating), not on first use
  async function warmUp(sizes = [224], video = null) {
    const c = document.createElement('canvas'); c.width = 160; c.height = 160;
    const g = c.getContext('2d'); g.fillStyle = '#777'; g.fillRect(0, 0, 160, 160);
    for (const inputSize of sizes) for (const src of video ? [c, video] : [c])
      await faceapi.detectSingleFace(src, new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: 0.5 })).withFaceLandmarks().withFaceDescriptor();
    await faceapi.nets.faceLandmark68Net.detectLandmarks(c);
    await faceapi.nets.faceRecognitionNet.computeFaceDescriptor(c);
  }
  async function describe(input, inputSize = 320) {
    const r = await faceapi
      .detectSingleFace(input, new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: 0.5 }))
      .withFaceLandmarks().withFaceDescriptor();
    if (!r) return null;
    return { d: Array.from(r.descriptor, v => +v.toFixed(5)), score: r.detection.score, box: r.detection.box };
  }
  function dist(a, b) {
    let s = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - b[i]; s += x * x; }
    return Math.sqrt(s);
  }

  /* ---------- images ---------- */
  function squareCrop(src, sw, sh, box, size, pad = 0.35) {
    let x, y, side;
    if (box) {
      side = Math.max(box.width, box.height) * (1 + pad * 2);
      x = box.x + box.width / 2 - side / 2; y = box.y + box.height / 2 - side / 2;
    } else { side = Math.min(sw, sh); x = (sw - side) / 2; y = (sh - side) / 2; }
    const c = document.createElement('canvas'); c.width = c.height = size;
    c.getContext('2d').drawImage(src, x, y, side, side, 0, 0, size, size);
    return c.toDataURL('image/jpeg', 0.85);
  }
  function fileToImage(file) {
    return new Promise((res, rej) => {
      const url = URL.createObjectURL(file); const img = new Image();
      img.onload = () => res(img); img.onerror = rej; img.src = url;
    });
  }

  return { load, save, addLog, hashPin, validPin, uid, fullName, accessStatus, accessLabel, fmtWhen, initials, avatar,
    loadFace, warmUp, describe, dist, squareCrop, fileToImage, KEY };
})();
