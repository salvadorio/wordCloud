#!/usr/bin/env node
// Tiny proxy for the word-cloud app.
// Forwards /datamuse/* → https://api.datamuse.com/* and adds CORS headers.
// Usage: node proxy.js [port]

const http  = require('http');
const https = require('https');

const PORT = parseInt(process.argv[2]) || 3001;

const UPSTREAMS = {
  '/datamuse/': 'https://api.datamuse.com/',
};

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  if (req.method !== 'GET')     { res.writeHead(405); res.end('Method not allowed'); return; }

  let upstream = null;
  for (const [prefix, base] of Object.entries(UPSTREAMS)) {
    if (req.url.startsWith(prefix)) {
      upstream = base + req.url.slice(prefix.length);
      break;
    }
  }

  if (!upstream) { res.writeHead(404); res.end('Not found'); return; }

  console.log(`→ ${upstream}`);
  https.get(upstream, { headers: { Accept: 'application/json' } }, (upRes) => {
    res.writeHead(upRes.statusCode, { 'Content-Type': 'application/json' });
    upRes.pipe(res);
  }).on('error', (err) => {
    console.error('Upstream error:', err.message);
    res.writeHead(502); res.end(JSON.stringify({ error: err.message }));
  });
});

server.listen(PORT, () => {
  console.log(`Word-cloud proxy running on http://localhost:${PORT}`);
  console.log(`  /datamuse/* → https://api.datamuse.com/*`);
});
