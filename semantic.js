// semantic.js — TensorFlow.js Universal Sentence Encoder
// Exposed as plain globals (no module bundler required).
//
// ── INTEGRATION GUIDE ────────────────────────────────────────
//
//  1. Include this file ONCE in your HTML (no other CDN tags needed —
//     TF.js and USE are loaded lazily on first call):
//
//       <script src="semantic.js"></script>
//
//  2. Settings object (add near your SIM config):
//
//       const settings = {
//         useEmbeddings:       true,   // local TF.js embeddings
//         useDatamuse:         false,  // online Datamuse API
//         similarityK:         5,      // top-K neighbours per word
//         similarityThreshold: 0.35,   // min cosine similarity for an edge
//       };
//
//  3. App initialisation — preload model in background immediately
//     after generate() / on DOMContentLoaded:
//
//       loadEmbeddingModel()
//         .then(() => { embeddingModelReady = true; updateModelStatus('ready'); })
//         .catch(() => updateModelStatus('error'));
//
//  4. Switch between backends in setSemanticMode():
//
//       async function setSemanticMode(enabled) {
//         semanticMode = enabled;
//         if (!enabled) {
//           springs = personSprings;
//           presimulate(1800);
//         } else if (nodes.length) {
//           if (settings.useEmbeddings) await fetchEmbeddingSprings();
//           else                        await fetchSemanticSprings();  // Datamuse
//           presimulate(1800);
//         }
//       }
//
// ─────────────────────────────────────────────────────────────
'use strict';

const _TFJS_CDN = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4/dist/tf.min.js';
const _USE_CDN  = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder@1/dist/universal-sentence-encoder.min.js';

let _useModel    = null;  // cached UniversalSentenceEncoder instance
let _loadPromise = null;  // in-flight Promise<model> while loading

// ─── private helper ──────────────────────────────────────────
function _loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src     = src;
    s.onload  = resolve;
    s.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(s);
  });
}

// ─────────────────────────────────────────────────────────────
//  loadEmbeddingModel()
//
//  Load TF.js + USE from CDN (once) and cache the model instance.
//  Subsequent calls return the cached model immediately.
//
//  @returns {Promise<UniversalSentenceEncoder>}
// ─────────────────────────────────────────────────────────────
async function loadEmbeddingModel() {
  if (_useModel)    return _useModel;
  if (_loadPromise) return _loadPromise;

  _loadPromise = (async () => {
    if (typeof tf  === 'undefined') await _loadScript(_TFJS_CDN);
    if (typeof use === 'undefined') await _loadScript(_USE_CDN);
    const model  = await use.load();
    _useModel    = model;
    _loadPromise = null;
    return model;
  })();

  return _loadPromise;
}

// ─────────────────────────────────────────────────────────────
//  embedTokens(tokens)
//
//  Embed an array of raw display tokens in ONE batch call.
//  Each token is wrapped in a template string to stabilise
//  short words and multi-word phrases in the embedding space:
//
//    "They describe me as: <TOKEN>."
//
//  @param  {string[]}   tokens  — raw display tokens
//  @returns {Promise<number[][]>}  N × 512 embedding matrix
// ─────────────────────────────────────────────────────────────
async function embedTokens(tokens) {
  const model     = await loadEmbeddingModel();
  const sentences = tokens.map(t => `They describe me as: ${t}.`);
  const tensor    = await model.embed(sentences);
  const matrix    = await tensor.array();   // shape [N, 512]
  tensor.dispose();
  return matrix;
}

// ─────────────────────────────────────────────────────────────
//  cosineSimilarity(a, b)
//
//  @param  {number[]} a  — 512-dim vector
//  @param  {number[]} b  — 512-dim vector
//  @returns {number}  cosine similarity in [-1, 1]
//                     (USE embeddings are typically in [0, 1])
// ─────────────────────────────────────────────────────────────
function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

// ─────────────────────────────────────────────────────────────
//  buildEmbeddingLinks(tokens, options)
//
//  Embed all tokens, build a pairwise cosine-similarity matrix,
//  then for each token keep only its top-K most-similar neighbours
//  that exceed `threshold`.  Duplicate edges are collapsed.
//
//  Edge object format:
//    {
//      source:   string,   // token label
//      target:   string,   // token label
//      weight:   number,   // cosine similarity (0–1)
//      distance: number,   // baseDistance * (1 - weight)  ← physics rest length
//      strength: number,   // clamp(weight * 2, 0.05, 1)   ← spring multiplier
//    }
//
//  @param  {string[]} tokens
//  @param  {{
//    K?:                number,    default 5
//    threshold?:        number,    default 0.40
//    baseDistance?:     number,    default 200
//    symmetric?:        boolean,   default true  — mutual top-K (see below)
//    fallbackK?:        number,    default 2     — max one-sided edges for isolated nodes
//    onesidedPenalty?:  number,    default 0.5   — weight multiplier for one-sided edges
//    onProgress?:       Function,  callback(loaded, total)
//  }} options
//
//  symmetric = true  (recommended, default)
//    An edge A↔B is only created when BOTH A ranks B in its top-K
//    AND B ranks A in its top-K.  This is "mutual k-NN" and
//    eliminates one-sided "reaches" that cause spurious connections.
//
//    Isolated-node fallback (symmetric mode only):
//    After the mutual pass, any node that ended up with zero connections
//    is allowed to claim up to `fallbackK` of its best one-sided edges,
//    weighted down by `onesidedPenalty`.  This lets niche or rare words
//    (proper nouns, slang, domain-specific terms) anchor themselves to
//    the graph without pulling in false positives for ordinary words.
//
//  symmetric = false
//    Union top-K: an edge exists if EITHER endpoint ranks the other
//    in its top-K.  Produces a denser graph, more false positives.
//
//  @returns {Promise<Array>}
// ─────────────────────────────────────────────────────────────
async function buildEmbeddingLinks(tokens, options = {}) {
  const {
    K               = 5,
    threshold       = 0.40,
    baseDistance    = 200,
    symmetric       = true,
    fallbackK       = 2,
    onesidedPenalty = 0.5,
    onProgress      = null,
  } = options;

  const n = tokens.length;
  if (n < 2) return [];

  if (onProgress) onProgress(0, n);

  // ── 1. Embed all tokens in a single batch call ──
  const vecs = await embedTokens(tokens);
  if (onProgress) onProgress(n, n);

  // ── 2. Precompute full similarity matrix (flat Float32Array) ──
  const sim = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const s        = cosineSimilarity(vecs[i], vecs[j]);
      sim[i * n + j] = s;
      sim[j * n + i] = s;
    }
  }

  // ── 3. Build top-K neighbour sets for every token ──
  //    Each set contains the indices of that token's K best neighbours
  //    (above threshold).  Used for the mutual-agreement check.
  const topKSets = Array.from({ length: n }, () => new Set());
  for (let i = 0; i < n; i++) {
    const nbrs = [];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const s = sim[i * n + j];
      if (s >= threshold) nbrs.push({ j, s });
    }
    nbrs.sort((a, b) => b.s - a.s);
    for (const { j } of nbrs.slice(0, K)) topKSets[i].add(j);
  }

  // ── 4. Emit mutual edges, tracking per-node degree ──
  //    symmetric=true  → both endpoints must agree (mutual k-NN)
  //    symmetric=false → either endpoint suffices (union k-NN)
  const seen   = new Set();
  const edges  = [];
  const degree = new Int32Array(n); // connection count after mutual pass

  for (let i = 0; i < n; i++) {
    for (const j of topKSets[i]) {
      if (symmetric && !topKSets[j].has(i)) continue; // not mutual — skip
      const key = i < j ? `${i}\x00${j}` : `${j}\x00${i}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const s = sim[i * n + j];
      edges.push({
        source:   tokens[i],
        target:   tokens[j],
        weight:   s,
        distance: baseDistance * (1 - s),
        strength: Math.max(0.05, Math.min(1, s * 2)),
        onesided: false,
      });
      degree[i]++;
      degree[j]++;
    }
  }

  // ── 5. Isolated-node fallback (symmetric mode only) ──
  //    Niche words (proper nouns, slang, rare terms) may end up with
  //    zero mutual connections even though they *do* have a closest
  //    neighbour.  For each such node we allow up to `fallbackK`
  //    one-sided edges, penalised by `onesidedPenalty`, so they stay
  //    anchored without inflating the mutual graph.
  if (symmetric) {
    for (let i = 0; i < n; i++) {
      if (degree[i] > 0) continue; // already connected — don't touch

      // Collect candidates: in i's top-K but not mutual
      const candidates = [];
      for (const j of topKSets[i]) {
        // (if it were mutual, degree[i] would be > 0 already)
        candidates.push({ j, s: sim[i * n + j] });
      }
      candidates.sort((a, b) => b.s - a.s);

      let added = 0;
      for (const { j, s } of candidates) {
        if (added >= fallbackK) break;
        const key = i < j ? `${i}\x00${j}` : `${j}\x00${i}`;
        if (seen.has(key)) continue; // already added (e.g. as j's fallback)
        seen.add(key);
        const w = s * onesidedPenalty;
        edges.push({
          source:   tokens[i],
          target:   tokens[j],
          weight:   w,
          distance: baseDistance * (1 - w),
          strength: Math.max(0.05, Math.min(1, w * 2)),
          onesided: true,  // flagged — physics & render treat it as softer
        });
        degree[i]++;
        added++;
      }
    }
  }

  return edges;
}
