// Load WASM module
const DemucsWasmModule = await import('../build/build-wasm/demucs_wasm.js');
const wasmModule = await DemucsWasmModule.default();

// Constants from C++ implementation
const SAMPLE_RATE = 44100;
const SEGMENT_LEN_SECS = 2.00;
const OVERLAP = 0.25;
const TRANSITION_POWER = 1.0;
const SEGMENT_SAMPLES = Math.floor(SEGMENT_LEN_SECS * SAMPLE_RATE);
const STRIDE_SAMPLES = Math.floor((1.0 - OVERLAP) * SEGMENT_SAMPLES);

// --- Encrypted asset fetch shim (transparent to ORT) ---
// Resolve key endpoint under your app prefix (e.g., /local_realtime_demucs/)
// If the page is /local_realtime_demucs/src_wasm/demo.html, this becomes
// /local_realtime_demucs/get_key_for_session
const APP_BASE = location.pathname.includes('/src_wasm/')
    ? location.pathname.split('/src_wasm/')[0] + '/'
    : (location.pathname.endsWith('/') ? location.pathname : location.pathname.replace(/[^/]+$/, ''));
const MODEL_KEY_ENDPOINT = APP_BASE + 'get_key_for_session';
const ENCRYPTED_SUFFIX = '.enc';
// Large-file caching (use Cache Storage; localStorage is too small for 160MB)
const MODEL_CACHE_VERSION = 'v1';
const MODEL_CACHE_NAME = `demucs-model-cache-${MODEL_CACHE_VERSION}`;

// --- Transport / Mixer state ---
let mixCtx = null;
let mixSources = null;   // [{srcNode, gainNode}]
let mixStartTime = 0;    // audioContext.currentTime when playback started
let mixOffset = 0;       // seconds offset into the track
let mixDuration = 0;     // seconds
let isPlaying = false;
let sourceBuffers = null; // [drums,bass,other,vocals] as AudioBuffers
let drawRAF = null;
let peakData = null;     // {timeline:[min,max][], lanes:{drums:[],...}}
let muteState = { drums: false, bass: false, other: false, vocals: false };
let soloState = { drums: false, bass: false, other: false, vocals: false };

let ortSession = null;
let audioBuffer = null;
// const MODEL_PATH = '../demucs-onnx/htdemucs.onnx';
// const MODEL_PATH = 'adv_inception_v3_Opset16.onnx';
let MODEL_PATH = 'htdemucs.onnx';
let inputFileBaseName = 'track';

let device = null;
let useWebGPU = false; // toggled true when WebGPU EP is active

// --- Google Analytics helper ---
function gaEvent(name, params = {}) {
    try { if (window.gtag) window.gtag('event', name, params); } catch (_) { }
}

// AES-GCM helpers: file is stored as iv(12) || tag(16) || ciphertext
async function hexToBytes(hex) {
    const out = new Uint8Array(hex.length / 2);
    for (let i = 0; i < hex.length; i += 2) out[i / 2] = parseInt(hex.slice(i, i + 2), 16);
    return out;
}
async function decryptAesGcm(ivTagCipherBuf, keyHex) {
    const u8 = new Uint8Array(ivTagCipherBuf);
    const iv = u8.slice(0, 12);
    const tag = u8.slice(12, 28);
    const ct = u8.slice(28);
    const ctPlusTag = new Uint8Array(ct.length + tag.length);
    ctPlusTag.set(ct, 0);
    ctPlusTag.set(tag, ct.length);

    const keyBytes = await hexToBytes(keyHex.trim());
    const cryptoKey = await crypto.subtle.importKey("raw", keyBytes, { name: "AES-GCM" }, false, ["decrypt"]);
    const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, cryptoKey, ctPlusTag.buffer);
    return plain; // ArrayBuffer
}

// Monkey-patch window.fetch so requests for plaintext model assets are served by decrypting the .enc versions.
async function cleanupOldModelCaches() {
    if (!('caches' in window)) return;
    try {
        const keys = await caches.keys();
        await Promise.all(keys
            .filter(k => k.startsWith('demucs-model-cache-') && k !== MODEL_CACHE_NAME)
            .map(k => caches.delete(k)));
    } catch (e) {
        console.warn('Cache cleanup skipped:', e);
    }
}

function installEncryptedAssetFetch() {
    if (window.__encryptedFetchInstalled) return;
    const originalFetch = window.fetch.bind(window);

    window.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        try {
            // Only intercept the model + external data (adjust the path prefix if you host elsewhere)
            const u = new URL(url, location.href);
            const pathname = u.pathname;

            const isModel = pathname.endsWith('/src_wasm/htdemucs.onnx');
            const isExt = pathname.endsWith('/src_wasm/htdemucs.onnx.data');

            if (isModel || isExt) {
                // 0) Try cache first
                let cache = null;
                if ('caches' in window) {
                    try {
                        // Best-effort cleanup once per page load
                        if (!window.__demucsCacheCleaned) {
                            window.__demucsCacheCleaned = true;
                            cleanupOldModelCaches();
                        }
                        cache = await caches.open(MODEL_CACHE_NAME);
                        const cached = await cache.match(url);
                        if (cached) {
                            // console.log('Serving model from cache:', pathname);
                            return cached;
                        }
                    } catch (e) {
                        console.warn('Model cache unavailable, proceeding without cache:', e);
                        cache = null;
                    }
                }

                // 1) fetch short-lived key (server must auth this)
                // normalize asset path for the server: strip app base so it matches "/src_wasm/…"
                const assetForServer = pathname.startsWith(APP_BASE) ? pathname.slice(APP_BASE.length - 1) : pathname;
                const keyResp = await originalFetch(`${MODEL_KEY_ENDPOINT}?asset=${encodeURIComponent(assetForServer)}`, { credentials: 'include' });

                if (!keyResp.ok) throw new Error(`key fetch failed: ${keyResp.status}`);
                const keyHex = (await keyResp.text()).trim();
                if (!/^[0-9a-fA-F]+$/.test(keyHex)) throw new Error('invalid key format');

                // 2) fetch encrypted bytes
                const encUrl = url + ENCRYPTED_SUFFIX; // e.g., .../htdemucs.onnx.enc
                const encResp = await originalFetch(encUrl, { credentials: 'include' });
                if (!encResp.ok) throw new Error(`encrypted fetch failed: ${encResp.status}`);
                const encBuf = await encResp.arrayBuffer();

                // 3) decrypt
                const plainBuf = await decryptAesGcm(encBuf, keyHex);

                // 4) return a Response so ORT can consume it as if it was the plaintext file
                //    content-type octet-stream is fine for ONNX / external data
                const resp = new Response(plainBuf, {
                    status: 200,
                    headers: { 'Content-Type': 'application/octet-stream', 'Cross-Origin-Resource-Policy': 'cross-origin' }
                });
                // 5) Store in cache for future loads
                try {
                    if (cache) await cache.put(url, resp.clone());
                } catch (e) {
                    console.warn('Failed to cache model asset:', e);
                }
                return resp;
            }
        } catch (e) {
            console.error('Encrypted fetch shim failed; falling back to normal fetch:', e);
            // fallthrough to originalFetch below
        }
        return originalFetch(input, init);
    };

    window.__encryptedFetchInstalled = true;
}


// --- iOS Safari debug overlay: mirror console to an on-page log when enabled ---
(function setupDebugOverlay() {
    const original = {
        log: console.log.bind(console),
        warn: console.warn.bind(console),
        error: console.error.bind(console),
    };

    const MAX_LINES = 500;

    function formatArg(a) {
        if (a instanceof Error) return a.stack || a.message || String(a);
        const t = typeof a;
        if (t === 'object') {
            try { return JSON.stringify(a); } catch { return String(a); }
        }
        return String(a);
    }

    function getOverlay() {
        let el = document.getElementById('debugOverlay');
        if (!el) {
            // Create lazily if HTML is missing
            el = document.createElement('div');
            el.id = 'debugOverlay';
            el.style.cssText = 'position:fixed;bottom:8px;left:8px;right:8px;height:30vh;background:rgba(0,0,0,.92);color:#d0ffd0;font:12px/1.4 ui-monospace,Menlo,monospace;padding:8px 10px;border:1px solid #333;border-radius:8px;overflow:auto;z-index:9999;display:none;white-space:pre-wrap;word-break:break-word;';
            document.body.appendChild(el);
        }
        return el;
    }

    function append(level, args) {
        if (!window.__SHOW_IOS_DEBUG) return;
        const el = getOverlay();
        el.style.display = 'block';
        const div = document.createElement('div');
        div.className = `line ${level}`;
        const ts = new Date().toISOString().split('T')[1].replace('Z', '');
        div.textContent = `[${ts}] ${args.map(formatArg).join(' ')}`;
        el.appendChild(div);
        // Trim to max lines
        while (el.childNodes.length > MAX_LINES) el.removeChild(el.firstChild);
        el.scrollTop = el.scrollHeight;
    }

    console.log = (...args) => { original.log(...args); append('log', args); };
    console.warn = (...args) => { original.warn(...args); append('warn', args); };
    console.error = (...args) => { original.error(...args); append('error', args); };

    // Global controls
    window.setDebugOverlay = (enabled) => {
        window.__SHOW_IOS_DEBUG = !!enabled;
        const el = getOverlay();
        el.style.display = enabled ? 'block' : 'none';
    };
    window.clearDebugOverlay = () => {
        const el = getOverlay();
        el.innerHTML = '';
    };
    // Off by default
    window.__SHOW_IOS_DEBUG = false;
})();

export async function initializeDemucs() {
    const initBtn = document.getElementById('initBtn');
    const initStatus = document.getElementById('initStatus');
    const errorContainer = document.getElementById('errorContainer');

    try {
        initBtn.disabled = true;
        initStatus.style.display = 'block';
        errorContainer.innerHTML = '';

        console.log("1");
        installEncryptedAssetFetch();

        // Show iOS notification banner if applicable
        try {
            const ua = navigator.userAgent || navigator.vendor || '';
            const isIOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
            if (isIOS) {
                const iosBanner = document.getElementById('iosBanner');
                if (iosBanner) iosBanner.style.display = 'block';
                gaEvent('ios_detected');
            }
        } catch { }

        // If later we want to fall back to wasm if webgpu is unavailable:
        // ort.env.wasm.numThreads = 8;           // for example
        // ort.env.wasm.simd = true;              // enable SIMD if your build supports it
        // ort.env.wasm.proxy = true;             // (optional) enables multi-threading with Web Workers

        const externalDataUrl = 'htdemucs.onnx.data';

        console.log('Loading ONNX model...');
        MODEL_PATH = 'htdemucs.onnx';
        console.log('Model path:', MODEL_PATH);

        console.log("2");

        // Try WebGPU first when available
        if (navigator.gpu) {
            try {
                ortSession = await ort.InferenceSession.create(MODEL_PATH, {
                    executionProviders: ['webgpu'],
                });
                useWebGPU = true;
                console.log('ONNX Runtime session created with WebGPU');
                device = ort.env.webgpu.device;
                console.log('Using webgpu:', device);
            } catch (e) {
                console.warn('WebGPU init failed, falling back to WASM:', e);
            }
        } else {
            console.warn('navigator.gpu not available; falling back to WASM');
        }

        console.log("3");

        // Fallback to WASM EP
        if (!useWebGPU) {
            // Configure WASM runtime for best performance
            try {
                ort.env.wasm.simd = true;
                ort.env.wasm.proxy = true; // enable worker threading when possible
                const hc = (navigator.hardwareConcurrency || 2);
                ort.env.wasm.numThreads = Math.min(4, Math.max(1, hc));
            } catch { }

            ortSession = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ['wasm'],
            });
            console.log('ONNX Runtime session created with WASM (CPU)');
            const banner = document.getElementById('fallbackBanner');
            if (banner) banner.style.display = 'block';
        }

        initStatus.style.display = 'none';
        initBtn.textContent = '✓ Model Loaded';
        initBtn.style.background = '#4caf50';
        document.getElementById('audioFile').disabled = false;

        const orig = document.getElementById('originalPlayer');
        if (audioBuffer) { orig.src = audioBufferToUrl(audioBuffer); }

        console.log('Demucs initialized successfully');
        gaEvent('model_loaded', { method: 'onnxruntime-web', ep: useWebGPU ? 'webgpu' : 'wasm' });

    } catch (error) {
        initStatus.style.display = 'none';
        initBtn.disabled = false;
        showError('Failed to initialize: ' + error.message);
        console.error(error);
        gaEvent('model_error', { message: String(error && error.message || error) });
    }
};

window.handleFileSelect = async function (event) {
    const file = event.target.files[0];
    if (!file) return;
    console.log('Selected file:', file);

    const fileNameSpan = document.getElementById('fileName');
    const separateBtn = document.getElementById('separateBtn');
    const errorContainer = document.getElementById('errorContainer');

    fileNameSpan.textContent = file.name;
    try { inputFileBaseName = baseNameNoExt(file.name); } catch { }
    errorContainer.innerHTML = '';

    try {
        // Decode audio file
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        // iOS Safari: ensure context is resumed and use a compat decoder
        try { await audioContext.resume(); } catch { }
        audioBuffer = await decodeAudioDataCompat(audioContext, arrayBuffer);

        // Resample to 44100 Hz if needed
        if (audioBuffer.sampleRate !== 44100) {
            console.log(`Resampling from ${audioBuffer.sampleRate} Hz to 44100 Hz...`);
            audioBuffer = await resampleAudio(audioBuffer, 44100);
        }

        // Ensure stereo
        if (audioBuffer.numberOfChannels === 1) {
            console.log('Converting mono to stereo...');
            audioBuffer = monoToStereo(audioBuffer);
        }

        separateBtn.disabled = false;
        gaEvent('audio_loaded', {
            source: 'file_input',
            duration_s: Number(audioBuffer.duration.toFixed(3))
        });

        const orig = document.getElementById('originalPlayer');
        const origSection = document.getElementById('originalSection');
        orig.src = audioBufferToUrl(audioBuffer);
        if (origSection) origSection.style.display = 'block';

        console.log(`Audio loaded: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate} Hz`);
    } catch (error) {
        showError('Failed to load audio file: ' + error.message);
        console.error(error);
    }
};

function byteSizeFromDims(dims, bytesPerElement = 4) {
    return dims.reduce((a, b) => a * b, 1) * bytesPerElement;
}

async function readGpuBufferToFloat32(device, srcBuffer, dims) {
    const size = byteSizeFromDims(dims, 4); // float32
    const readback = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const arr = new Float32Array(readback.getMappedRange().slice(0)); // copy out
    readback.unmap();
    readback.destroy();
    return arr;
}

// Safari-compatible decodeAudioData wrapper (supports both promise and callback forms)
async function decodeAudioDataCompat(audioContext, arrayBuffer) {
    return await new Promise((resolve, reject) => {
        try {
            const maybePromise = audioContext.decodeAudioData(arrayBuffer, resolve, reject);
            if (maybePromise && typeof maybePromise.then === 'function') {
                maybePromise.then(resolve, reject);
            }
        } catch (err) {
            // Some Safari builds throw synchronously; retry legacy signature
            try {
                audioContext.decodeAudioData(arrayBuffer, resolve, reject);
            } catch (err2) {
                reject(err2);
            }
        }
    });
}


window.separateAudio = async function () {
    if (!audioBuffer || !ortSession) return;

    const separateBtn = document.getElementById('separateBtn');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const resultsSection = document.getElementById('resultsSection');
    const sourcesGrid = document.getElementById('sourcesGrid');
    const errorContainer = document.getElementById('errorContainer');

    try {
        separateBtn.disabled = true;
        progressContainer.style.display = 'block';
        resultsSection.style.display = 'none';
        errorContainer.innerHTML = '';
        const globalBtn = document.getElementById('globalDownloadBtn');
        if (globalBtn) { globalBtn.disabled = true; globalBtn.style.display = 'none'; }

        const sep_t0 = performance.now();
        gaEvent('separation_start', {
            duration_s: Number(audioBuffer.duration.toFixed(3)),
            sample_rate: 44100,
            segment_seconds: SEGMENT_LEN_SECS,
            overlap: OVERLAP
        });


        // ============================================================================
        // RANDOM SHIFT AUGMENTATION (Time-Invariance)
        // ============================================================================
        // This implements the random shift logic from model_apply.cpp:
        // 1. Add max_shift (0.5s = 22050 samples) padding on both sides (zero-padded)
        // 2. Generate random shift_offset in [0, max_shift)
        // 3. Process the shifted audio (length = original + max_shift - shift_offset)
        // 4. After processing, trim back to original length by removing (max_shift - shift_offset)
        //    samples from the start
        // This improves separation quality by making the model time-invariant.
        // ============================================================================

        // Extract and normalize audio
        const numSamples = audioBuffer.length;
        const leftChannel = audioBuffer.getChannelData(0);
        const rightChannel = audioBuffer.getChannelData(1);

        // Normalize
        // Compute mono, then stats:
        const mono = new Float32Array(numSamples);
        for (let i = 0; i < numSamples; i++) mono[i] = 0.5 * (leftChannel[i] + rightChannel[i]);
        const mean = mono.reduce((a, b) => a + b, 0) / numSamples;
        let varSum = 0;
        for (let i = 0; i < numSamples; i++) { const d = mono[i] - mean; varSum += d * d; }
        const std = Math.sqrt(varSum / (numSamples - 1));

        // Apply random shift for time-invariance (matching C++ model_apply.cpp)
        const MAX_SHIFT_SECS = 0.5;
        const maxShift = Math.floor(MAX_SHIFT_SECS * SAMPLE_RATE); // 22050 samples
        const shiftOffset = Math.floor(Math.random() * maxShift);  // Random offset [0, maxShift)
        console.log(`Random shift offset: ${shiftOffset} samples (${(shiftOffset / SAMPLE_RATE).toFixed(3)}s)`);

        // Create padded and shifted audio (symmetric zero padding)
        const paddedLength = numSamples + 2 * maxShift;
        const paddedMix = new Float32Array(paddedLength * 2);
        // paddedMix is already zero-initialized

        // Normalize and copy audio into paddedMix starting at maxShift
        for (let ch = 0; ch < 2; ch++) {
            const sourceChannel = ch === 0 ? leftChannel : rightChannel;
            for (let i = 0; i < numSamples; i++) {
                paddedMix[ch * paddedLength + maxShift + i] = (sourceChannel[i] - mean) / std;
            }
        }

        // Create shifted view of the audio
        const shiftedLength = numSamples + maxShift - shiftOffset;
        const audioData = new Float32Array(shiftedLength * 2);
        for (let ch = 0; ch < 2; ch++) {
            for (let i = 0; i < shiftedLength; i++) {
                audioData[ch * shiftedLength + i] = paddedMix[ch * paddedLength + shiftOffset + i];
            }
        }

        const nbSources = 4; // drums, bass, other, vocals
        const sourceNames = ['drums', 'bass', 'other', 'vocals'];

        const N_FFT = 4096;
        const HOP = 1024;

        const pad = Math.floor(HOP / 2.0) * 3;  // 1536, matching C++
        const le = Math.ceil(SEGMENT_SAMPLES / HOP);
        const padEnd = pad + le * HOP - SEGMENT_SAMPLES;
        const paddedSegmentSamples = SEGMENT_SAMPLES + pad + padEnd;

        const nbFrames = Math.floor(paddedSegmentSamples / HOP) + 1;  // Match C++ exactly
        const nbFreqFrames = nbFrames - 4;                                     // 173
        console.log(`nbFrames: ${nbFrames}, nbFreqFrames: ${nbFreqFrames}`);


        console.log('Segment processing params:');
        console.log(`  pad: ${pad}, padEnd: ${padEnd}`);
        console.log(`  paddedSegmentSamples: ${paddedSegmentSamples}`);
        console.log(`  nbFrames: ${nbFrames}, nbFreqFrames: ${nbFreqFrames}`);
        console.log(`  le (segments per chunk): ${le}`);


        // --- Input/Output buffers (GPU when WebGPU, otherwise created per-iteration for WASM) ---
        if (useWebGPU) {
            console.log('Using WebGPU device:', device);
        } else {
            console.log('Using WASM backend (CPU)');
        }
        const timeBytes = 1 * 2 * SEGMENT_SAMPLES * 4;                 // float32
        const freqBytes = 1 * 4 * 2048 * nbFreqFrames * 4;             // float32

        let timeBuf, freqBuf, timeTensorGPU, freqTensorGPU;
        let freqOutBuf, timeOutBuf, freqOutTensorGPU, timeOutTensorGPU;
        if (useWebGPU) {
            timeBuf = device.createBuffer({
                usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
                size: timeBytes,
            });
            freqBuf = device.createBuffer({
                usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
                size: freqBytes,
            });

            timeTensorGPU = ort.Tensor.fromGpuBuffer(timeBuf, {
                dataType: 'float32',
                dims: [1, 2, SEGMENT_SAMPLES]
            });
            freqTensorGPU = ort.Tensor.fromGpuBuffer(freqBuf, {
                dataType: 'float32',
                dims: [1, 4, 2048, nbFreqFrames]
            });
        }


        // Output buffers
        const NB_SOURCES = 4;

        const freqOutDims = [1, 4, 4, 2048, nbFreqFrames];
        const timeOutDims = [1, NB_SOURCES, 2, SEGMENT_SAMPLES];                  // rank-3

        const elements = dims => dims.reduce((a, b) => a * b, 1);
        const bytes = n => n * 4;
        if (useWebGPU) {
            // Output buffers (COPY_SRC so we can read back to CPU)
            freqOutBuf = device.createBuffer({
                size: bytes(elements(freqOutDims)),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            timeOutBuf = device.createBuffer({
                size: bytes(elements(timeOutDims)),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            // GPU-backed output tensors with the **correct shapes**
            freqOutTensorGPU = ort.Tensor.fromGpuBuffer(freqOutBuf, {
                dataType: 'float32',
                dims: freqOutDims
            });
            timeOutTensorGPU = ort.Tensor.fromGpuBuffer(timeOutBuf, {
                dataType: 'float32',
                dims: timeOutDims
            });
        }

        // Calculate total chunks for overlap-add (on shifted audio)
        const totalChunks = Math.ceil(shiftedLength / STRIDE_SAMPLES);
        console.log(`Processing ${totalChunks} chunks for ${shiftedLength} samples (shifted from ${numSamples})`);

        // Initialize output accumulator and weights (for shifted length)
        const out = new Float32Array(nbSources * 2 * shiftedLength);
        const sumWeight = new Float32Array(shiftedLength);

        // Create weight vector for overlapping segments (triangular window)
        const weight = new Float32Array(SEGMENT_SAMPLES);
        const half = Math.floor(SEGMENT_SAMPLES / 2);
        for (let i = 0; i < half; i++) weight[i] = i + 1;
        for (let i = half; i < SEGMENT_SAMPLES; i++) weight[i] = SEGMENT_SAMPLES - i;
        const maxw = weight[half - 1];
        for (let i = 0; i < SEGMENT_SAMPLES; i++) weight[i] = Math.pow(weight[i] / maxw, TRANSITION_POWER);

        // Log first and last 100 values of the weight vector for debugging
        // console.log('Weight vector (first 100 values):', weight.slice(0, 100));
        // console.log('Weight vector (last 100 values):', weight.slice(-100));

        // Process each segment
        for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
            // Take time measurement
            console.time(`Chunk ${chunkIdx + 1}/${totalChunks}`);

            const segmentOffset = chunkIdx * STRIDE_SAMPLES;
            const chunkLength = Math.min(SEGMENT_SAMPLES, shiftedLength - segmentOffset);

            // Map processing progress to full width. Use (chunkIdx+1) so the bar advances immediately.
            const pct = ((chunkIdx + 1) / totalChunks) * 100;
            progressText.textContent = `Processing segment ${chunkIdx + 1}/${totalChunks}...`;
            progressBar.style.width = `${Math.min(100, pct).toFixed(2)}%`;

            // Extract chunk
            const segment = new Float32Array(SEGMENT_SAMPLES * 2);
            for (let ch = 0; ch < 2; ch++) {
                for (let i = 0; i < chunkLength; i++) {
                    segment[ch * SEGMENT_SAMPLES + i] = audioData[ch * shiftedLength + segmentOffset + i];
                }
            }

            // Symmetric padding to fit into segment_samples, then add reflect padding
            const symmetricPadding = paddedSegmentSamples - chunkLength;
            const symmetricPaddingStart = Math.floor(symmetricPadding / 2);

            let paddedSegment = new Float32Array(paddedSegmentSamples * 2);
            // Copy chunk into padded segment
            for (let ch = 0; ch < 2; ch++) {
                for (let i = 0; i < chunkLength; i++) {
                    paddedSegment[ch * paddedSegmentSamples + pad + symmetricPaddingStart + i] =
                        segment[ch * SEGMENT_SAMPLES + i];
                }
            }

            // Apply reflect padding
            for (let ch = 0; ch < 2; ch++) {
                const chOffset = ch * paddedSegmentSamples;
                // Left side
                for (let i = 0; i < pad; i++) {
                    paddedSegment[chOffset + pad - 1 - i] = paddedSegment[chOffset + pad + i];
                }
                // Right side
                const lastElem = SEGMENT_SAMPLES + pad - 1;
                for (let i = 0; i < padEnd; i++) {
                    paddedSegment[chOffset + lastElem + i + 1] = paddedSegment[chOffset + lastElem - i];
                }
            }

            console.time("WASM STFT");
            // Compute STFT
            const spec = wasmModule.wasm_stft(paddedSegment, paddedSegmentSamples);
            console.timeEnd("WASM STFT");
            // Prepare inputs
            console.time("WASM prepare inputs");
            const freqInput = wasmModule.wasm_prepare_freq_input(spec, nbFrames);
            const timeInput = wasmModule.wasm_prepare_time_input(paddedSegment, SEGMENT_SAMPLES, pad);
            console.timeEnd("WASM prepare inputs");

            // Run inference via WebGPU or WASM depending on EP
            console.log("Input names:", ortSession.inputNames);
            console.log("Output names:", ortSession.outputNames);

            let freqOutput, timeOutput;
            if (useWebGPU) {
                // Upload to GPU buffers
                device.queue.writeBuffer(freqBuf, 0, freqInput.buffer, freqInput.byteOffset, freqInput.byteLength);
                device.queue.writeBuffer(timeBuf, 0, timeInput.buffer, timeInput.byteOffset, timeInput.byteLength);

                const feeds = {};
                feeds[ortSession.inputNames[0]] = timeTensorGPU;
                feeds[ortSession.inputNames[1]] = freqTensorGPU;
                const fetches = {
                    [ortSession.outputNames[0]]: freqOutTensorGPU, // freq branch output
                    [ortSession.outputNames[1]]: timeOutTensorGPU  // time branch output
                };

                // Print input shapes
                console.log("Time input shape:", timeTensorGPU.dims);
                console.log("Freq input shape:", freqTensorGPU.dims);

                // Time the ONNX inference (WebGPU)
                console.time(`ONNX Inference chunk ${chunkIdx + 1}`);
                await ortSession.run(feeds, fetches);
                console.timeEnd(`ONNX Inference chunk ${chunkIdx + 1}`);

                freqOutput = await readGpuBufferToFloat32(device, freqOutBuf, freqOutDims);
                timeOutput = await readGpuBufferToFloat32(device, timeOutBuf, timeOutDims);
            } else {
                // WASM/CPU path
                const freqTensor = new ort.Tensor('float32', freqInput, [1, 4, 2048, nbFreqFrames]);
                const timeTensor = new ort.Tensor('float32', timeInput, [1, 2, SEGMENT_SAMPLES]);
                const feeds = {};
                feeds[ortSession.inputNames[0]] = timeTensor;
                feeds[ortSession.inputNames[1]] = freqTensor;

                console.time(`ONNX Inference chunk ${chunkIdx + 1}`);
                const results = await ortSession.run(feeds);
                console.timeEnd(`ONNX Inference chunk ${chunkIdx + 1}`);

                const freqT = results[ortSession.outputNames[0]];
                const timeT = results[ortSession.outputNames[1]];
                freqOutput = freqT.data;
                timeOutput = timeT.data;
            }

            // Print output shapes
            console.log("Freq output shape:", freqOutDims);
            console.log("Time output shape:", timeOutDims);

            console.time("WASM unpack");
            // Unpack frequency output
            const freqSpec = wasmModule.wasm_unpack_freq_output(freqOutput, nbSources, nbFrames);
            console.timeEnd("WASM unpack");

            console.time("WASM iSTFT");
            // iSTFT for each source
            const freqWaveforms = new Float32Array(nbSources * 2 * SEGMENT_SAMPLES);
            for (let src = 0; src < nbSources; src++) {
                // freqSpec layout: (nb_sources, 2, nb_bins=2049, nb_frames, 2)
                // We need to extract the spectrogram for this source
                const srcSpecOffset = src * 2 * 2049 * nbFrames * 2;
                const srcSpecSize = 2 * 2049 * nbFrames * 2;
                const srcSpec = freqSpec.subarray(srcSpecOffset, srcSpecOffset + srcSpecSize);

                const srcWave = wasmModule.wasm_istft(srcSpec, paddedSegmentSamples);

                // Extract the unpadded segment from the iSTFT output
                // srcWave layout: (2, paddedSegmentSamples)
                for (let ch = 0; ch < 2; ch++) {
                    for (let i = 0; i < SEGMENT_SAMPLES; i++) {
                        const srcIdx = ch * paddedSegmentSamples + pad + i;
                        const dstIdx = src * 2 * SEGMENT_SAMPLES + ch * SEGMENT_SAMPLES + i;
                        freqWaveforms[dstIdx] = srcWave[srcIdx];
                    }
                }
            }
            console.timeEnd("WASM iSTFT");

            console.time("WASM merge");
            // Merge branches
            const merged = wasmModule.wasm_merge_branches(freqWaveforms, timeOutput, nbSources, SEGMENT_SAMPLES);
            console.timeEnd("WASM merge");

            // Extract chunk output with center trimming
            const chunkOut = new Float32Array(nbSources * 2 * chunkLength);
            for (let src = 0; src < nbSources; src++) {
                for (let ch = 0; ch < 2; ch++) {
                    for (let i = 0; i < chunkLength; i++) {
                        const kidx = Math.min(i + symmetricPaddingStart, SEGMENT_SAMPLES - 1);
                        const srcIdx = src * 2 * SEGMENT_SAMPLES + ch * SEGMENT_SAMPLES + kidx;
                        const dstIdx = src * 2 * chunkLength + ch * chunkLength + i;
                        chunkOut[dstIdx] = merged[srcIdx];
                    }
                }
            }

            // Accumulate weighted chunk output
            for (let src = 0; src < nbSources; src++) {
                for (let ch = 0; ch < 2; ch++) {
                    for (let i = 0; i < chunkLength; i++) {
                        const outIdx = src * 2 * shiftedLength + ch * shiftedLength + segmentOffset + i;
                        const chunkIdx = src * 2 * chunkLength + ch * chunkLength + i;
                        out[outIdx] += chunkOut[chunkIdx] * weight[i];
                    }
                }
            }

            // Accumulate weights
            for (let i = 0; i < chunkLength; i++) {
                sumWeight[segmentOffset + i] += weight[i];
            }

            console.timeEnd(`Chunk ${chunkIdx + 1}/${totalChunks}`);
        }

        // Normalize by sum_weight and denormalize (keep progress at current level)
        progressText.textContent = 'Normalizing outputs...';

        for (let src = 0; src < nbSources; src++) {
            for (let ch = 0; ch < 2; ch++) {
                for (let i = 0; i < shiftedLength; i++) {
                    const idx = src * 2 * shiftedLength + ch * shiftedLength + i;
                    out[idx] = (out[idx] / sumWeight[i]) * std + mean;
                }
            }
        }

        // Trim the output to remove the shift padding
        progressText.textContent = 'Trimming output...';

        const trimStart = maxShift - shiftOffset;
        const trimmedOut = new Float32Array(nbSources * 2 * numSamples);

        for (let src = 0; src < nbSources; src++) {
            for (let ch = 0; ch < 2; ch++) {
                for (let i = 0; i < numSamples; i++) {
                    const srcIdx = src * 2 * shiftedLength + ch * shiftedLength + trimStart + i;
                    const dstIdx = src * 2 * numSamples + ch * numSamples + i;
                    trimmedOut[dstIdx] = out[srcIdx];
                }
            }
        }

        // Create AudioBuffers
        progressText.textContent = 'Creating audio outputs...';

        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const sources = [];

        for (let src = 0; src < nbSources; src++) {
            const buffer = audioContext.createBuffer(2, numSamples, 44100);
            const left = buffer.getChannelData(0);
            const right = buffer.getChannelData(1);

            for (let i = 0; i < numSamples; i++) {
                left[i] = trimmedOut[src * 2 * numSamples + i];
                right[i] = trimmedOut[src * 2 * numSamples + numSamples + i];
            }
            sources.push(buffer);
        }

        // Display results
        progressBar.style.width = '100%';
        progressText.textContent = 'Complete!';

        // Store buffers for mixer (order must match labels above)
        sourceBuffers = [sources[0], sources[1], sources[2], sources[3]];
        mixDuration = audioBuffer.duration;

        resultsSection.style.display = 'block';
        setupMixerUI();
        // ensure canvases have layout widths before drawing peaks
        requestAnimationFrame(() => computeAndRenderWaveforms());

        progressContainer.style.display = 'none';
        separateBtn.disabled = false;
        if (globalBtn) { globalBtn.disabled = false; globalBtn.style.display = 'inline-block'; }

        gaEvent('separation_complete', {
            wall_ms: Math.round(performance.now() - sep_t0)
        });

        console.log('Separation complete!');
    } catch (error) {
        progressContainer.style.display = 'none';
        separateBtn.disabled = false;
        showError('Separation failed: ' + error.message);
        console.error(error);
        gaEvent('separation_error', { message: String(error && error.message || error) });
    }
};

function setupMixGraph(startAtSec = 0) {
    if (!mixCtx) mixCtx = new (window.AudioContext || window.webkitAudioContext)();
    // Stop previous
    stopMixGraph(false);

    mixSources = [];
    const names = ['drums', 'bass', 'other', 'vocals'];
    for (let i = 0; i < 4; i++) {
        const src = mixCtx.createBufferSource();
        src.buffer = sourceBuffers[i];
        const gain = mixCtx.createGain();
        const muted = effectiveMuted(names[i]);
        gain.gain.value = muted ? 0 : 1;

        src.connect(gain).connect(mixCtx.destination);
        mixSources.push({ srcNode: src, gainNode: gain });
    }
    mixStartTime = mixCtx.currentTime;
    mixOffset = startAtSec;
    for (let i = 0; i < 4; i++) {
        const dur = sourceBuffers[i].duration;
        const len = Math.max(0, dur - startAtSec);
        // start(when, offset, duration)
        mixSources[i].srcNode.start(0, startAtSec, len);
    }
}

function stopMixGraph(resetOffset = true) {
    if (!mixSources) return;
    try { mixSources.forEach(s => { try { s.srcNode.stop(); } catch { } }); } catch { }
    mixSources = null;
    if (resetOffset) { mixOffset = 0; }
}

function setupMixerUI() {
    const playBtn = document.getElementById('playPauseBtn');
    const timeline = document.getElementById('timeline');
    const names = ['drums', 'bass', 'other', 'vocals'];

    playBtn.disabled = false;
    document.getElementById('durTime').textContent = fmtTime(mixDuration);

    playBtn.onclick = () => {
        if (!isPlaying) {
            setupMixGraph(mixOffset);
            isPlaying = true;
            playBtn.textContent = '⏸ Pause';
            loopDraw();
            gaEvent('transport', { action: 'play', at_s: Number(mixOffset.toFixed(3)) });
        } else {
            // pause
            const now = performance.now();
            mixOffset = getPlayheadSec();
            stopMixGraph(false);
            isPlaying = false;
            playBtn.textContent = '▶ Play';
            if (drawRAF) cancelAnimationFrame(drawRAF);
            drawPlayheads(mixOffset);
            gaEvent('transport', { action: 'pause', at_s: Number(mixOffset.toFixed(3)) });
        }
    };

    // Seek via timeline and lane canvases
    const seekTargets = [timeline, ...names.map(n => document.getElementById(`wf-${n}`))];
    seekTargets.forEach(canvas => {
        canvas.onmousedown = (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = Math.min(Math.max(0, e.clientX - rect.left), rect.width);
            const ratio = x / rect.width;
            const sec = ratio * mixDuration;
            seekTo(sec);
        };
    });

    // Mute / Solo toggles and Download buttons
    names.forEach((n, i) => {
        const muteBtn = document.getElementById(`mute-${n}`);
        const soloBtn = document.getElementById(`solo-${n}`);
        const dlBtn = document.getElementById(`download-${n}`);
        if (muteBtn) {
            muteBtn.onclick = () => {
                muteState[n] = !muteState[n];
                updateToggleUI(n);
                updateGainsFromState();
            };
        }
        if (soloBtn) {
            soloBtn.onclick = () => {
                const willActivate = !soloState[n];
                // Make solo exclusive: clear all, then optionally activate this one
                soloState = { drums: false, bass: false, other: false, vocals: false };
                if (willActivate) {
                    soloState[n] = true;
                }
                // Update all toggle UIs since multiple buttons change
                names.forEach(updateToggleUI);
                updateGainsFromState();
            };
        }
        if (dlBtn) {
            dlBtn.disabled = false; // results exist if we're here
            dlBtn.onclick = () => downloadStemByIndex(i, n);
        }
        updateToggleUI(n);
    });

    // Enable and wire global ZIP button
    const globalBtn = document.getElementById('globalDownloadBtn');
    if (globalBtn) {
        globalBtn.disabled = false;
        globalBtn.onclick = downloadAllStemsZip;
    }

}

function anySoloActive() {
    return soloState.drums || soloState.bass || soloState.other || soloState.vocals;
}
function effectiveMuted(name) {
    if (anySoloActive()) return !soloState[name];
    return muteState[name];
}
function updateToggleUI(name) {
    const m = document.getElementById(`mute-${name}`);
    const s = document.getElementById(`solo-${name}`);
    if (m) m.classList.toggle('active', muteState[name]);
    if (s) s.classList.toggle('active', soloState[name]);
}
function updateGainsFromState() {
    if (!mixSources) return;
    const names = ['drums', 'bass', 'other', 'vocals'];
    for (let i = 0; i < names.length; i++) {
        const muted = effectiveMuted(names[i]);
        mixSources[i].gainNode.gain.value = muted ? 0 : 1;
    }
}


function seekTo(sec) {
    mixOffset = Math.max(0, Math.min(mixDuration, sec));
    if (isPlaying) {
        setupMixGraph(mixOffset);
    }
    drawPlayheads(mixOffset);
}

function getPlayheadSec() {
    if (!isPlaying || !mixCtx) return mixOffset;
    const elapsed = mixCtx.currentTime - mixStartTime;
    return Math.min(mixDuration, mixOffset + elapsed);
}

function fmtTime(t) {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// -------- Waveforms --------
function computePeaksForBuffer(buffer, pixels) {
    const chL = buffer.getChannelData(0);
    const chR = buffer.getChannelData(1);
    const block = Math.max(1, Math.floor(chL.length / pixels));
    const out = new Array(pixels);
    for (let i = 0; i < pixels; i++) {
        const start = i * block;
        const end = Math.min(chL.length, start + block);
        let min = 0, max = 0;
        for (let j = start; j < end; j++) {
            const v = 0.5 * (chL[j] + chR[j]); // mono view
            if (v < min) min = v;
            if (v > max) max = v;
        }
        out[i] = [min, max];
    }
    return out;
}

function computeAndRenderWaveforms() {
    const width = document.getElementById('timeline').clientWidth || 800;
    const timelinePx = width;
    const lanePx = document.getElementById('wf-drums').clientWidth || width;

    peakData = {
        timeline: computePeaksForBuffer(audioBuffer, timelinePx),
        lanes: {
            drums: computePeaksForBuffer(sourceBuffers[0], lanePx),
            bass: computePeaksForBuffer(sourceBuffers[1], lanePx),
            other: computePeaksForBuffer(sourceBuffers[2], lanePx),
            vocals: computePeaksForBuffer(sourceBuffers[3], lanePx),
        }
    };
    renderStaticWaveforms();
    drawPlayheads(mixOffset || 0);
}

function renderPeaksToCanvas(canvas, peaks, color = '#667eea') {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth * dpr;
    const h = canvas.clientHeight * dpr;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = color;
    const mid = h / 2;
    for (let x = 0; x < w; x++) {
        const i = Math.floor((x / w) * peaks.length);
        const [mn, mx] = peaks[i] || [0, 0];
        const y1 = mid - Math.abs(mn) * mid;
        const y2 = mid + Math.abs(mx) * mid;
        ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
}

function renderStaticWaveforms() {
    renderPeaksToCanvas(document.getElementById('timeline'), peakData.timeline, '#90caf9');
    renderPeaksToCanvas(document.getElementById('wf-drums'), peakData.lanes.drums, '#b39ddb');
    renderPeaksToCanvas(document.getElementById('wf-bass'), peakData.lanes.bass, '#a5d6a7');
    renderPeaksToCanvas(document.getElementById('wf-other'), peakData.lanes.other, '#ffcc80');
    renderPeaksToCanvas(document.getElementById('wf-vocals'), peakData.lanes.vocals, '#ef9a9a');
}

function drawPlayheadOn(canvas, sec, color = '#333') {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth * dpr;
    const h = canvas.clientHeight * dpr;
    const ctx = canvas.getContext('2d');
    // draw overlay line
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * dpr;
    const x = Math.min(w, Math.max(0, (sec / mixDuration) * w));
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
    ctx.stroke();
    ctx.restore();
}

function drawPlayheads(sec) {
    // redraw static waves then overlay playhead to avoid trails
    renderStaticWaveforms();
    drawPlayheadOn(document.getElementById('timeline'), sec);
    ['drums', 'bass', 'other', 'vocals'].forEach(n => {
        drawPlayheadOn(document.getElementById(`wf-${n}`), sec);
    });
    document.getElementById('curTime').textContent = fmtTime(sec);
}

function loopDraw() {
    const sec = getPlayheadSec();
    drawPlayheads(sec);
    if (isPlaying && sec < mixDuration) {
        drawRAF = requestAnimationFrame(loopDraw);
    } else if (sec >= mixDuration) {
        // auto-stop at end
        isPlaying = false;
        stopMixGraph();
        document.getElementById('playPauseBtn').textContent = '▶ Play';
        drawPlayheads(mixDuration);
    }
}

window.addEventListener('resize', () => {
    if (sourceBuffers && peakData) computeAndRenderWaveforms();
});



function showError(message) {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.innerHTML = `<div class="error">${message}</div>`;
}

function audioBufferToUrl(buffer) {
    const wav = audioBufferToWav(buffer);
    const blob = new Blob([wav], { type: 'audio/wav' });
    return URL.createObjectURL(blob);
}

function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const data = new Float32Array(buffer.length * numChannels);
    for (let ch = 0; ch < numChannels; ch++) {
        const channelData = buffer.getChannelData(ch);
        for (let i = 0; i < buffer.length; i++) {
            data[i * numChannels + ch] = channelData[i];
        }
    }

    const dataLength = data.length * bytesPerSample;
    const arrayBuffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(arrayBuffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);

    // PCM samples
    let offset = 44;
    for (let i = 0; i < data.length; i++) {
        const sample = Math.max(-1, Math.min(1, data[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
    }

    return arrayBuffer;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// ---------- Downloads helpers ----------
function baseNameNoExt(name) {
    try {
        const n = (name || 'track').split('/').pop().split('\\').pop();
        const dot = n.lastIndexOf('.');
        return (dot > 0 ? n.slice(0, dot) : n).replace(/[^a-zA-Z0-9_\-]+/g, '_');
    } catch {
        return 'track';
    }
}

function downloadBlob(filename, blob) {
    const a = document.createElement('a');
    const url = URL.createObjectURL(blob);
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }, 0);
}

function downloadStemByIndex(index, name) {
    try {
        if (!sourceBuffers || !sourceBuffers[index]) return;
        const wavAB = audioBufferToWav(sourceBuffers[index]);
        const blob = new Blob([wavAB], { type: 'audio/wav' });
        const fname = `${inputFileBaseName || 'track'}_${name}.wav`;
        downloadBlob(fname, blob);
        gaEvent('download_stem', { stem: name });
    } catch (e) {
        console.error('Failed to download stem', name, e);
    }
}

async function downloadAllStemsZip() {
    try {
        if (!sourceBuffers) return;
        if (typeof JSZip === 'undefined') {
            alert('ZIP download unavailable (JSZip not loaded).');
            return;
        }
        const zip = new JSZip();
        const names = ['drums', 'bass', 'other', 'vocals'];
        for (let i = 0; i < names.length; i++) {
            if (!sourceBuffers[i]) continue;
            const wavAB = audioBufferToWav(sourceBuffers[i]);
            zip.file(`${inputFileBaseName || 'track'}_${names[i]}.wav`, wavAB);
        }
        const blob = await zip.generateAsync({ type: 'blob' });
        downloadBlob(`${inputFileBaseName || 'track'}_stems.zip`, blob);
        gaEvent('download_all_zip');
    } catch (e) {
        console.error('Failed to create ZIP', e);
    }
}

async function resampleAudio(audioBuffer, targetSampleRate) {
    const length = Math.max(1, Math.round(audioBuffer.duration * targetSampleRate));
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const offlineContext = new OfflineCtx(
        audioBuffer.numberOfChannels,
        length,
        targetSampleRate
    );

    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start(0);

    return await offlineContext.startRendering();
}

function monoToStereo(audioBuffer) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const stereoBuffer = audioContext.createBuffer(2, audioBuffer.length, audioBuffer.sampleRate);
    const monoData = audioBuffer.getChannelData(0);
    stereoBuffer.getChannelData(0).set(monoData);
    stereoBuffer.getChannelData(1).set(monoData);
    return stereoBuffer;
}

export async function loadAudioFile(url) {
    const fileNameSpan = document.getElementById('fileName');
    const separateBtn = document.getElementById('separateBtn');
    const errorContainer = document.getElementById('errorContainer');

    try {
        fileNameSpan.textContent = 'Loading ' + url + '...';
        try { inputFileBaseName = baseNameNoExt(url.split('/').pop().split('?')[0] || 'track'); } catch { }
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        try { await audioContext.resume(); } catch { }
        audioBuffer = await decodeAudioDataCompat(audioContext, arrayBuffer);

        // Resample to 44100 Hz if needed
        if (audioBuffer.sampleRate !== 44100) {
            console.log(`Resampling from ${audioBuffer.sampleRate} Hz to 44100 Hz...`);
            audioBuffer = await resampleAudio(audioBuffer, 44100);
        }

        // Ensure stereo
        if (audioBuffer.numberOfChannels === 1) {
            console.log('Converting mono to stereo...');
            audioBuffer = monoToStereo(audioBuffer);
        }

        separateBtn.disabled = false;
        fileNameSpan.textContent = url;

        const orig = document.getElementById('originalPlayer');
        const origSection = document.getElementById('originalSection');
        orig.src = audioBufferToUrl(audioBuffer);
        if (origSection) origSection.style.display = 'block';

        console.log(`Audio loaded: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate} Hz`);
    } catch (error) {
        fileNameSpan.textContent = 'Failed to load ' + url;
        showError('Failed to load audio file: ' + error.message);
        console.error(error);
    }
}
