const CACHE_VERSION = 'v2.2.1'; 
const CACHE_NAME = `demucsstems-${CACHE_VERSION}`;

const FILES_TO_CACHE = [
  './', 
  './index.html',
  './manifest.json',
  './player-192x192.PNG',
  './player-512x512.PNG'
];

self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Install');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[ServiceWorker] Caching app shell');
        return cache.addAll(FILES_TO_CACHE);
      })
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activate');
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(keyList.map((key) => {
        if (key !== CACHE_NAME) {
          console.log('[ServiceWorker] Removing old cache', key);
          return caches.delete(key);
        }
      }));
    })
    .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  
  const url = new URL(event.request.url);

  if (url.hostname.endsWith('.ts.net') || event.request.method === 'POST') {
      return; 
  }

  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        return response || fetch(event.request);
      })
  );
});
