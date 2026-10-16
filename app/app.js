'use strict';
(() => {
  const $ = (s) => document.querySelector(s);
  const el = {
    category: $('#category'), categoryInfo: $('#category-info'),
    modelStatus: $('#model-status'), modelFill: $('#model-fill'), modelLabel: $('#model-label'),
    drop: $('#drop'), file: $('#file'), samples: $('#samples'),
    preview: $('#preview'), previewImg: $('#preview-img'), previewCap: $('#preview-cap'),
    stepConvert: $('#step-convert'), progress: $('#progress'), convertBar: $('#convert-bar'),
    convertFill: $('#convert-fill'), convertLabel: $('#convert-label'),
    convertDone: $('#convert-done'), inputPreview: $('#input-preview'), inputDesc: $('#input-desc'),
    stepResult: $('#step-result'), resultBody: $('#result-body'),
    error: $('#error'), errorText: $('#error-text'), testOut: $('#test-output'),
  };
  const MAX_SIDE = 2048;
  const MAX_FILE_MB = 40;
  // each conversion step stays on screen at least this long so the progress labels are readable
  const MIN_STEP_MS = 160;
  const state = { registry: null, category: null, sessions: new Map(), loading: null, busy: false };

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));
  const show = (node, on = true) => node.classList.toggle('hidden', !on);
  const pct = (x, digits = 1) => `${(100 * x).toFixed(digits)}%`;
  const showError = (msg) => { el.errorText.textContent = msg; show(el.error); };
  const clearError = () => { el.errorText.textContent = ''; show(el.error, false); };

  // ---------- registry and models ----------
  async function init() {
    if (typeof ort === 'undefined') {
      showError('ONNX Runtime failed to load. Serve this folder over HTTP (for example: python3 -m http.server) rather than opening the file directly.');
      return;
    }
    ort.env.wasm.wasmPaths = new URL('vendor/ort/', location.href).href;
    try {
      const res = await fetch('registry.json', { cache: 'no-cache' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.registry = await res.json();
    } catch (e) {
      showError(`Could not read registry.json: ${e.message}`);
      return;
    }
    for (const c of state.registry.categories) {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.label;
      el.category.appendChild(opt);
    }
    el.category.addEventListener('change', () => selectCategory(el.category.value));
    el.file.addEventListener('change', () => {
      const f = el.file.files[0];
      el.file.value = '';
      if (f) handleImage(f, f.name);
    });
    for (const evt of ['dragenter', 'dragover']) {
      el.drop.addEventListener(evt, (e) => { e.preventDefault(); el.drop.classList.add('over'); });
    }
    el.drop.addEventListener('dragleave', () => el.drop.classList.remove('over'));
    el.drop.addEventListener('drop', (e) => {
      e.preventDefault();
      el.drop.classList.remove('over');
      const f = e.dataTransfer.files[0];
      if (f) handleImage(f, f.name);
    });
    await selectCategory(state.registry.categories[0].id);
    if (new URLSearchParams(location.search).has('test')) runSelfTest();
  }

  async function selectCategory(id) {
    const cat = state.registry.categories.find((c) => c.id === id);
    state.category = cat;
    el.category.value = id;
    el.categoryInfo.textContent = cat.description;
    el.samples.replaceChildren(...cat.samples.map((s) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn';
      b.textContent = s.label;
      b.addEventListener('click', () => loadSample(s));
      return b;
    }));
    show(el.preview, false);
    show(el.stepConvert, false);
    show(el.stepResult, false);
    clearError();
    try { await ensureModel(cat); } catch { /* reported by ensureModel */ }
  }

  function setModelStatus(kind, cat, frac, bytes) {
    el.modelStatus.className = `model-status ${kind}`;
    const mb = bytes ? ` (${(bytes / 1e6).toFixed(1)} MB)` : '';
    if (kind === 'loading') {
      el.modelFill.style.width = frac == null ? '15%' : pct(frac, 0);
      el.modelLabel.textContent = frac == null ? `Downloading model${mb}…` : `Downloading model… ${pct(frac, 0)}${mb}`;
    } else if (kind === 'compiling') {
      el.modelFill.style.width = '100%';
      el.modelLabel.textContent = 'Preparing model…';
    } else if (kind === 'ready') {
      el.modelFill.style.width = '100%';
      el.modelLabel.textContent = `Model ready: ${cat.label} (runs on this device).`;
    } else {
      el.modelFill.style.width = '100%';
      el.modelLabel.textContent = 'Model failed to load.';
    }
  }

  async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const total = Number(res.headers.get('content-length')) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(got, total);
    }
    const out = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
  }

  async function ensureModel(cat) {
    if (state.sessions.has(cat.id)) {
      setModelStatus('ready', cat);
      return state.sessions.get(cat.id);
    }
    if (state.loading && state.loading.id === cat.id) return state.loading.promise;
    const promise = (async () => {
      setModelStatus('loading', cat, 0);
      const bytes = await fetchWithProgress(cat.model, (got, total) => setModelStatus('loading', cat, total ? got / total : null, got));
      setModelStatus('compiling', cat);
      await nextFrame();
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
      state.sessions.set(cat.id, session);
      setModelStatus('ready', cat);
      return session;
    })();
    state.loading = { id: cat.id, promise };
    try {
      return await promise;
    } catch (e) {
      setModelStatus('error', cat);
      showError(`Could not load the model: ${e.message}`);
      throw e;
    } finally {
      state.loading = null;
    }
  }

  // ---------- image conversion (mirrors the training preprocessing: PIL grayscale -> Lanczos resize -> center crop -> ImageNet normalization) ----------
  function lanczos3(x) {
    if (x <= -3 || x >= 3) return 0;
    if (x === 0) return 1;
    const px = Math.PI * x;
    return (Math.sin(px) / px) * (Math.sin(px / 3) / (px / 3));
  }

  function resampleCoeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const filterScale = Math.max(scale, 1);
    const support = 3 * filterScale;
    const ss = 1 / filterScale;
    const bounds = new Int32Array(outSize * 2);
    const kernels = [];
    for (let i = 0; i < outSize; i++) {
      const center = (i + 0.5) * scale;
      let lo = Math.max(0, Math.trunc(center - support + 0.5));
      const hi = Math.min(inSize, Math.trunc(center + support + 0.5));
      const n = hi - lo;
      const k = new Float64Array(n);
      let sum = 0;
      for (let j = 0; j < n; j++) {
        const w = lanczos3((j + lo - center + 0.5) * ss);
        k[j] = w;
        sum += w;
      }
      if (sum !== 0) for (let j = 0; j < n; j++) k[j] /= sum;
      bounds[2 * i] = lo;
      bounds[2 * i + 1] = n;
      kernels.push(k);
    }
    return { bounds, kernels };
  }

  function resampleGray(src, w, h, outW, outH) {
    const hc = resampleCoeffs(w, outW);
    const tmp = new Uint8ClampedArray(outW * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < outW; x++) {
        const lo = hc.bounds[2 * x], n = hc.bounds[2 * x + 1], k = hc.kernels[x];
        let s = 0;
        for (let j = 0; j < n; j++) s += src[row + lo + j] * k[j];
        tmp[y * outW + x] = Math.round(s);
      }
    }
    const vc = resampleCoeffs(h, outH);
    const out = new Uint8ClampedArray(outW * outH);
    for (let y = 0; y < outH; y++) {
      const lo = vc.bounds[2 * y], n = vc.bounds[2 * y + 1], k = vc.kernels[y];
      for (let x = 0; x < outW; x++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += tmp[(lo + j) * outW + x] * k[j];
        out[y * outW + x] = Math.round(s);
      }
    }
    return out;
  }

  async function decodeBitmap(file) {
    try {
      return await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch {
      return createImageBitmap(file);
    }
  }

  function rasterize(bitmap) {
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const h = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, w, h);
    return { data: ctx.getImageData(0, 0, w, h).data, w, h, scaled: scale < 1 };
  }

  function toGray(rgba, w, h) {
    const out = new Uint8ClampedArray(w * h);
    for (let i = 0, p = 0; i < out.length; i++, p += 4) {
      out[i] = (rgba[p] * 19595 + rgba[p + 1] * 38470 + rgba[p + 2] * 7471 + 32768) >> 16;
    }
    return out;
  }

  function centerCrop(src, size, crop) {
    const off = (size - crop) >> 1;
    const out = new Uint8ClampedArray(crop * crop);
    for (let y = 0; y < crop; y++) out.set(src.subarray((y + off) * size + off, (y + off) * size + off + crop), y * crop);
    return out;
  }

  function toTensor(gray, spec) {
    const n = spec.crop * spec.crop;
    const data = new Float32Array(spec.channels * n);
    for (let c = 0; c < spec.channels; c++) {
      const m = spec.mean[c], s = spec.std[c], base = c * n;
      for (let i = 0; i < n; i++) data[base + i] = (gray[i] / 255 - m) / s;
    }
    return new ort.Tensor('float32', data, [1, spec.channels, spec.crop, spec.crop]);
  }

  async function convertImage(file, spec, onProgress) {
    const step = async (frac, label, fn) => {
      onProgress(frac, label);
      await nextFrame();
      const t = performance.now();
      const out = await fn();
      const dt = performance.now() - t;
      if (dt < MIN_STEP_MS) await sleep(MIN_STEP_MS - dt);
      return out;
    };
    const bitmap = await step(0.05, 'Reading and decoding the file', () => decodeBitmap(file));
    const srcW = bitmap.width, srcH = bitmap.height;
    const { data, w, h, scaled } = await step(0.25, `Rasterizing ${srcW}×${srcH} pixels`, () => rasterize(bitmap));
    if (bitmap.close) bitmap.close();
    const gray = await step(0.45, 'Converting to 8-bit grayscale', () => toGray(data, w, h));
    const resized = await step(0.62, `Resizing to ${spec.resize}×${spec.resize} (Lanczos)`, () => resampleGray(gray, w, h, spec.resize, spec.resize));
    const crop = await step(0.8, `Center-cropping to ${spec.crop}×${spec.crop}`, () => centerCrop(resized, spec.resize, spec.crop));
    const tensor = await step(0.92, 'Normalizing into a float32 tensor', () => toTensor(crop, spec));
    onProgress(1, 'Conversion complete');
    await nextFrame();
    return { tensor, crop, meta: { srcW, srcH, scaled } };
  }

  function setConvertProgress(frac, label) {
    const done = frac >= 1;
    el.progress.classList.toggle('done', done);
    el.convertFill.style.width = pct(frac, 0);
    el.convertBar.setAttribute('aria-valuenow', String(Math.round(frac * 100)));
    el.convertLabel.textContent = done ? '✓ Conversion complete' : `Converting image… ${Math.round(frac * 100)}% — ${label}`;
  }

  function showConvertDone(crop, meta, ms, spec) {
    const ctx = el.inputPreview.getContext('2d');
    const img = ctx.createImageData(spec.crop, spec.crop);
    for (let i = 0, p = 0; i < crop.length; i++, p += 4) {
      img.data[p] = img.data[p + 1] = img.data[p + 2] = crop[i];
      img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const capped = meta.scaled ? ` (downscaled to at most ${MAX_SIDE}px first)` : '';
    el.inputDesc.textContent = `Converted the ${meta.srcW}×${meta.srcH} image${capped} to 8-bit grayscale, resized it to ${spec.resize}×${spec.resize}, `
      + `center-cropped ${spec.crop}×${spec.crop}, and normalized it into a float32 tensor of shape [1, ${spec.channels}, ${spec.crop}, ${spec.crop}] `
      + `in ${Math.round(ms)} ms. The thumbnail is exactly what the model receives.`;
    show(el.convertDone);
  }

  // ---------- inference and result ----------
  async function infer(session, cat, tensor) {
    const out = await session.run({ [cat.input.name]: tensor });
    const logit = out[cat.output.name].data[0];
    return cat.output.activation === 'sigmoid' ? 1 / (1 + Math.exp(-logit)) : logit;
  }

  function renderResult(prob, ms) {
    const cat = state.category;
    const t = cat.output.threshold;
    const positive = prob >= t;
    const ev = cat.evaluation;
    el.resultBody.innerHTML = `
      <div class="verdict ${positive ? 'pos' : 'neg'}">
        <div class="verdict-head">
          <span class="badge"><span aria-hidden="true">${positive ? '!' : '✓'}</span> ${positive ? 'POSITIVE' : 'NEGATIVE'}</span>
          <span class="verdict-title">${positive ? cat.output.positive : cat.output.negative}</span>
        </div>
        <div class="score">
          <div class="score-track"><div class="score-fill" style="width:${pct(prob, 1)}"></div><div class="score-thresh" style="left:${pct(t, 1)}"></div></div>
          <div class="score-text">Model score ${prob.toFixed(3)} on a 0–1 scale; scores at or above the ${t.toFixed(2)} mark count as positive. Inference took ${Math.round(ms)} ms on this device.</div>
        </div>
        <p class="context">In a ${ev.set} of ${ev.n_images} images from ${ev.n_patients} patients, this setting caught ${pct(ev.sensitivity)} of true positives (sensitivity) and cleared ${pct(ev.specificity)} of negatives (specificity); AUROC ${ev.auroc.toFixed(3)}.</p>
        <p class="caveat">${cat.caveats} This result is not a diagnosis.</p>
      </div>`;
  }

  async function handleImage(file, name) {
    if (state.busy) return null;
    clearError();
    const looksLikeImage = (file.type && file.type.startsWith('image/')) || /\.(jpe?g|png|webp|gif|bmp|avif|tiff?)$/i.test(name || '');
    if (!looksLikeImage) { showError('Please choose an image file (JPEG, PNG, WebP…).'); return null; }
    if (file.size > MAX_FILE_MB * 1e6) { showError(`That file is larger than ${MAX_FILE_MB} MB. Please use a smaller image.`); return null; }
    state.busy = true;
    const cat = state.category;
    try {
      if (el.previewImg.src.startsWith('blob:')) URL.revokeObjectURL(el.previewImg.src);
      el.previewImg.src = URL.createObjectURL(file);
      el.previewCap.textContent = `${name} · ${(file.size / 1024).toFixed(0)} KB`;
      show(el.preview);
      show(el.convertDone, false);
      show(el.stepResult, false);
      setConvertProgress(0, 'Starting');
      show(el.stepConvert);
      el.stepConvert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      const t0 = performance.now();
      const { tensor, crop, meta } = await convertImage(file, cat.input, setConvertProgress);
      const convertMs = performance.now() - t0;
      showConvertDone(crop, meta, convertMs, cat.input);

      el.resultBody.innerHTML = '<p class="working"><span class="spinner" aria-hidden="true"></span> Analyzing…</p>';
      show(el.stepResult);
      const session = await ensureModel(cat);
      const t1 = performance.now();
      const prob = await infer(session, cat, tensor);
      const inferMs = performance.now() - t1;
      renderResult(prob, inferMs);
      el.stepResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      return { prob, convertMs, inferMs };
    } catch (e) {
      console.error(e);
      showError(`Could not process that image: ${e.message}`);
      return null;
    } finally {
      state.busy = false;
    }
  }

  async function loadSample(sample) {
    clearError();
    try {
      const res = await fetch(sample.file);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      return handleImage(blob, sample.file.split('/').pop());
    } catch (e) {
      showError(`Could not load the sample image: ${e.message}`);
      return null;
    }
  }

  // ---------- self test (?test): runs every sample through the full pipeline and prints machine-readable results ----------
  async function runSelfTest() {
    const results = [];
    for (const s of state.category.samples) {
      const r = await loadSample(s);
      results.push({ sample: s.file, truth: s.truth, prob: r ? Number(r.prob.toFixed(4)) : null,
        verdict: r ? (r.prob >= state.category.output.threshold ? 'positive' : 'negative') : 'error',
        convertMs: r ? Math.round(r.convertMs) : null, inferMs: r ? Math.round(r.inferMs) : null });
    }
    const summary = { category: state.category.id, ua: navigator.userAgent, threads: ort.env.wasm.numThreads, results };
    el.testOut.textContent = `TEST-RESULT ${JSON.stringify(summary)}`;
    show(el.testOut);
    console.log(`TEST-RESULT ${JSON.stringify(summary)}`);
    document.title = `TEST DONE - ${document.title}`;
  }

  init();
})();
