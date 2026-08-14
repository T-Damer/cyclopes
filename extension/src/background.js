const MAX_INFERENCE_CONCURRENCY = 1;
const inFlight = new Map();
const queue = [];
let active = 0;
let offscreenReady;
let filterEnabled = false;
let ready = false;

function showState(enabled) {
  filterEnabled = enabled;
  const warmingUp = enabled && !ready;
  chrome.action.setBadgeText({ text: warmingUp ? "…" : enabled ? "ON" : "OFF" });
  chrome.action.setBadgeBackgroundColor({ color: warmingUp ? "#d97706" : enabled ? "#2563eb" : "#6b7280" });
  chrome.action.setTitle({ title: warmingUp ? "Cyclopes — Warming up" : `Cyclopes — Filter ${enabled ? "ON" : "OFF"}` });
}

function showError() {
  chrome.action.setBadgeText({ text: "ERR" });
  chrome.action.setBadgeBackgroundColor({ color: "#dc2626" });
  chrome.action.setTitle({ title: "Cyclopes — Inference failed" });
}

async function toggleFilter() {
  const { enabled } = await chrome.storage.local.get({ enabled: false });
  await chrome.storage.local.set({ enabled: !enabled });
  showState(!enabled);
}

chrome.action.onClicked.addListener(toggleFilter);
chrome.storage.local.get({ enabled: false }).then(({ enabled }) => showState(enabled));

async function ensureOffscreenDocument() {
  if (await chrome.offscreen.hasDocument()) return;
  offscreenReady ??= chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["BLOBS"],
    justification: "Decode displayed image bytes and run packaged ONNX inference."
  }).finally(() => { offscreenReady = undefined; });
  await offscreenReady;
}

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
    (async () => {
      try {
        await ensureOffscreenDocument();
        job.resolve(await chrome.runtime.sendMessage({ target: "offscreen", type: "infer", url: job.url }));
      } catch (error) {
        job.reject(error);
      } finally {
        active -= 1;
        drain();
      }
    })();
  }
}

function scoreImage(url) {
  if (inFlight.has(url)) return inFlight.get(url);
  const task = schedule(url)
    .then((result) => {
      if (typeof result?.score !== "number") throw new Error(result?.error || "Inference returned no score.");
      ready = true;
      showState(filterEnabled);
      return result.score;
    })
    .catch((error) => {
      if (filterEnabled && !ready) showError();
      throw error;
    })
    .finally(() => inFlight.delete(url));
  inFlight.set(url, task);
  return task;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "background" || message.type !== "score" || typeof message.url !== "string") return;
  scoreImage(message.url).then(
    (score) => sendResponse({ score }),
    (error) => sendResponse({ error: error instanceof Error ? error.message : String(error) })
  );
  return true;
});
