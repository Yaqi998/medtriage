'use strict';
// Offline support + cross-origin isolation for static hosts (GitHub Pages) that cannot set headers.
//  - install: precache every file the app needs, so after one visit it works with no network.
//  - fetch:   big immutable assets (model, wasm, images) come from the cache first; code and the
//             registry go to the network first so new deployments show up, falling back to the cache.
//  - every same-origin response gets COOP/COEP headers, which makes the page crossOriginIsolated
//    and lets ONNX Runtime Web use multi-threaded WebAssembly.
// __BUILD__ is replaced with the commit SHA by the Pages workflow; a new value refreshes the cache.
const CACHE = 'medtriage-__BUILD__';
const PRECACHE = [
  './',
  'index.html',
  'styles.css',
  'app.js',
  'registry.json',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'vendor/ort/ort.wasm.min.js',
  'vendor/ort/ort-wasm-simd-threaded.mjs',
  'vendor/ort/ort-wasm-simd-threaded.wasm',
  'models/cxr_pneumonia.onnx',
  'samples/normal.jpg',
  'samples/pneumonia.jpg',
];
const CACHE_FIRST = /\.(onnx|wasm|mjs|png|jpe?g|webp)$/i;

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll(PRECACHE.map((u) => new Request(u, { cache: 'reload' })));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key.startsWith('medtriage-') && key !== CACHE) await caches.delete(key);
    }
    await self.clients.claim();
  })());
});

function isolate(res) {
  // opaque or redirect responses cannot be rebuilt; pass them through untouched
  if (!res || res.status === 0 || res.type === 'opaqueredirect' || res.redirected) return res;
  const headers = new Headers(res.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

async function cacheFirst(req) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req, { ignoreSearch: true });
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) cache.put(req, res.clone());
  return res;
}

async function networkFirst(req) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    if (res.ok && !res.redirected) cache.put(req, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(req, { ignoreSearch: true })
      || (req.mode === 'navigate' ? await cache.match('./') : undefined);
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  const path = new URL(req.url).pathname;
  event.respondWith((CACHE_FIRST.test(path) ? cacheFirst(req) : networkFirst(req)).then(isolate));
});
