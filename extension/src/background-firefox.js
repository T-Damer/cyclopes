const MAX_INFERENCE_CONCURRENCY = 1;
const MAX_CACHE_ENTRIES = 256;
const ICON_SIZES = [16, 32, 48, 128];
const inFlight = new Map();
const queue = [];
const scores = new Map();
let sessionPromise;
let active = 0;
let filterEnabled = false;
let loadingFrame = 0;
let loadingTimer;

const api = globalThis.chrome || globalThis.browser;
if (!api) throw new Error("Extension API is unavailable.");
const action = api.action || api.browserAction;

function toHashString(arrayBuffer) {
  return Array.from(new Uint8Array(arrayBuffer), (value) => value.toString(16).padStart(2, "0")).join("");
}

function iconPaths(name) {
  return Object.fromEntries(ICON_SIZES.map((size) => [size, `icons/${name}-${size}.png`]));
}

function publishState(warmingUp) {
  return api.runtime.sendMessage({ target: "popup", type: "state", enabled: filterEnabled, warmingUp }).catch(() => {});
}

function stopLoadingAnimation() {
  clearInterval(loadingTimer);
  loadingTimer = undefined;
}

function showState(enabled) {
  filterEnabled = enabled;
  const warmingUp = enabled && (active > 0 || queue.length > 0);
  if (warmingUp && !loadingTimer) {
    loadingFrame = 0;
    loadingTimer = setInterval(() => {
      loadingFrame = (loadingFrame + 1) % 16;
      action?.setIcon({ path: iconPaths(`loading-${loadingFrame}`) });
    }, 125);
  } else if (!warmingUp) {
    stopLoadingAnimation();
  }
  action?.setIcon({ path: iconPaths(warmingUp ? `loading-${loadingFrame}` : enabled ? "on" : "off") });
  action?.setTitle({ title: warmingUp ? "Cyclopes — Warming up" : `Cyclopes — Filter ${enabled ? "ON" : "OFF"}` });
  publishState(warmingUp);
}

function showError() {
  stopLoadingAnimation();
  action?.setIcon({ path: iconPaths("off") });
  action?.setTitle({ title: "Cyclopes — Inference failed" });
  publishState(false);
}

function normalizeScoreOutput(output) {
  if (output?.data?.length !== 1) throw new Error(`Expected one AI logit, received ${output?.data?.length ?? 0}.`);
  return 1 / (1 + Math.exp(-Number(output.data[0])));
}

function sourceRegion(width, height) {
  return { x: 0, y: 0, width, height };
}

async function inferBlob(blob) {
  const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  try {
    if (!globalThis.ort) throw new Error("Local ONNX Runtime is unavailable.");
    if (!sessionPromise) {
      if (globalThis.ort?.env?.wasm) {
        globalThis.ort.env.wasm.wasmPaths = api.runtime.getURL("ort/");
        globalThis.ort.env.wasm.numThreads = 1;
      }
      sessionPromise = globalThis.ort.InferenceSession.create(api.runtime.getURL("models/cyclopes.onnx"), {
        executionProviders: ["webgl"],
      }).catch(() => globalThis.ort.InferenceSession.create(api.runtime.getURL("models/cyclopes.onnx")));
    }
    const session = await sessionPromise;

    const SIZE = 384;
    const MEAN = [0.48145466, 0.45782754, 0.40821073];
    const STD = [0.26862954, 0.26130258, 0.27577711];
    const canvas = new OffscreenCanvas(SIZE, SIZE);
    const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
    if (!context) throw new Error("2D canvas is unavailable.");
    context.fillStyle = "rgb(128 128 128)";
    context.fillRect(0, 0, SIZE, SIZE);
    const source = sourceRegion(bitmap.width, bitmap.height);
    const scale = Math.min(SIZE / source.width, SIZE / source.height);
    const width = Math.max(1, Math.round(source.width * scale));
    const height = Math.max(1, Math.round(source.height * scale));
    const left = (SIZE - width) / 2;
    const top = (SIZE - height) / 2;
    context.fillStyle = "white";
    context.fillRect(left, top, width, height);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, source.x, source.y, source.width, source.height, left, top, width, height);

    const rgba = context.getImageData(0, 0, SIZE, SIZE).data;
    const data = new Float32Array(3 * SIZE * SIZE);
    const plane = SIZE * SIZE;
    for (let index = 0; index < plane; index += 1) {
      const offset = index * 4;
      data[index] = (rgba[offset] / 255 - MEAN[0]) / STD[0];
      data[plane + index] = (rgba[offset + 1] / 255 - MEAN[1]) / STD[1];
      data[2 * plane + index] = (rgba[offset + 2] / 255 - MEAN[2]) / STD[2];
    }
    const input = new globalThis.ort.Tensor("float32", data, [1, 3, SIZE, SIZE]);
    const outputs = await session.run({ [session.inputNames[0]]: input });
    return normalizeScoreOutput(outputs[session.outputNames[0]]);
  } finally {
    bitmap.close();
  }
}

async function imageBlob(url) {
  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) throw new Error(`Image request failed (${response.status}).`);
  const blob = await response.blob();
  if (!blob.type.startsWith("image/")) throw new Error("Displayed resource is not an image.");
  return blob;
}

async function digest(blob) {
  const bytes = await blob.arrayBuffer();
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return toHashString(hash);
}

async function inferUrl(url) {
  const blob = await imageBlob(url);
  const key = await digest(blob);
  if (scores.has(key)) {
    const score = scores.get(key);
    scores.delete(key);
    scores.set(key, score);
    return score;
  }
  const score = await inferBlob(blob);
  scores.set(key, score);
  while (scores.size > MAX_CACHE_ENTRIES) scores.delete(scores.keys().next().value);
  return score;
}

async function thumbnailUrl(url) {
  const bitmap = await createImageBitmap(await imageBlob(url), { imageOrientation: "from-image" });
  try {
    const size = 128;
    const canvas = new OffscreenCanvas(size, size);
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("2D canvas is unavailable.");
    context.fillStyle = "#e2e8f0";
    context.fillRect(0, 0, size, size);
    const scale = Math.min(size / bitmap.width, size / bitmap.height);
    const width = Math.round(bitmap.width * scale);
    const height = Math.round(bitmap.height * scale);
    context.drawImage(bitmap, (size - width) / 2, (size - height) / 2, width, height);
    const thumbnail = await canvas.convertToBlob({ type: "image/webp", quality: 0.72 });
    const bytes = new Uint8Array(await thumbnail.arrayBuffer());
    let binary = "";
    for (let index = 0; index < bytes.length; index += 0x8000) {
      binary += String.fromCharCode(...bytes.slice(index, index + 0x8000));
    }
    return `data:image/webp;base64,${btoa(binary)}`;
  } finally {
    bitmap.close();
  }
}

function publishInitialState() {
  api.storage.local.get({ enabled: false }).then(({ enabled }) => showState(enabled));
}

api.storage.onChanged.addListener((changes) => {
  if (changes.enabled) showState(changes.enabled.newValue);
});

api.runtime.onInstalled.addListener(async () => {
  await api.contextMenus.removeAll();
  api.contextMenus.create({ id: "cyclopes-report-ai", title: "Report as AI", contexts: ["image"] });
  api.contextMenus.create({ id: "cyclopes-report-real", title: "Report as Non-AI", contexts: ["image"] });
  const tabs = await api.tabs.query({ url: ["http://*/*", "https://*/*"] });
  await Promise.allSettled(tabs.map(({ id }) => {
    if (id === undefined) return undefined;
    return new Promise((resolve, reject) => {
      if (typeof api.tabs.executeScript === "function") {
        api.tabs.executeScript(id, { file: "content.js" }, () => {
          if (api.runtime.lastError) reject(api.runtime.lastError);
          else resolve(undefined);
        });
        return;
      }
      if (api.scripting?.executeScript) {
        api.scripting.executeScript({
          target: { tabId: id },
          files: ["content.js"],
        }).then(() => resolve(undefined), reject);
        return;
      }
      reject(new Error("Cannot inject content script: no script execution API available."));
    });
  }));
});

function schedule(url) {
  return new Promise((resolve, reject) => {
    queue.push({ reject, resolve, url });
    drain();
  });
}

function drain() {
  while (active < MAX_INFERENCE_CONCURRENCY && queue.length) {
    const job = queue.shift();
    active += 1;
    showState(filterEnabled);
    (async () => {
      try {
        job.resolve(await inferUrl(job.url));
      } catch (error) {
        job.reject(error);
      } finally {
        active -= 1;
        showState(filterEnabled);
        drain();
      }
    })();
  }
}

function scoreImage(url) {
  if (inFlight.has(url)) return inFlight.get(url);
  const task = schedule(url)
    .then((score) => {
      if (typeof score !== "number") throw new Error("Inference returned no score.");
      return score;
    })
    .catch((error) => {
      if (filterEnabled) showError();
      throw error;
    })
    .finally(() => inFlight.delete(url));
  inFlight.set(url, task);
  return task;
}

async function saveFeedback(report) {
  const result = await thumbnailUrl(report.source);
  if (typeof result !== "string") throw new Error("Thumbnail creation failed.");
  const { feedbackReports = [] } = await api.storage.local.get({ feedbackReports: [] });
  await api.storage.local.set({
    feedbackReports: [{ ...report, thumbnail: result }, ...feedbackReports.filter(({ source }) => source !== report.source)].slice(0, 500),
  });
}

api.contextMenus.onClicked.addListener((info, tab) => {
  const label = info.menuItemId === "cyclopes-report-ai" ? "ai"
    : info.menuItemId === "cyclopes-report-real" ? "real"
    : undefined;
  if (!label || !info.srcUrl) return;
  let site = "";
  try { site = new URL(tab?.url ?? "").hostname; } catch {}
  saveFeedback({
    source: info.srcUrl,
    page: tab?.url ?? "",
    site,
    label,
    createdAt: new Date().toISOString(),
  }).catch((error) => console.warn(`Cyclopes: ${error instanceof Error ? error.message : String(error)}`));
});

api.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "background") return;
  if (message.type === "status") {
    sendResponse({ enabled: filterEnabled, warmingUp: filterEnabled && (active > 0 || queue.length > 0) });
    return;
  }
  const task = message.type === "score" && typeof message.url === "string" ? scoreImage(message.url) : undefined;
  if (!task) return;
  task.then(
    (score) => sendResponse({ score }),
    (error) => sendResponse({ error: error instanceof Error ? error.message : String(error) })
  );
  return true;
});

publishInitialState();
