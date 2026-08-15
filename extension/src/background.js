const MAX_INFERENCE_CONCURRENCY = 1;
const ICON_SIZES = [16, 32, 48, 128];
const inFlight = new Map();
const queue = [];
let active = 0;
let offscreenReady;
let filterEnabled = false;
let loadingFrame = 0;
let loadingTimer;

void chrome.action.setBadgeText({ text: "" });

function stopLoadingAnimation() {
  clearInterval(loadingTimer);
  loadingTimer = undefined;
}

function iconPaths(name) {
  return Object.fromEntries(ICON_SIZES.map((size) => [size, `icons/${name}-${size}.png`]));
}

function publishState(warmingUp) {
  void chrome.runtime.sendMessage({ target: "popup", type: "state", enabled: filterEnabled, warmingUp }).catch(() => {});
}

function showState(enabled) {
  filterEnabled = enabled;
  const warmingUp = enabled && (active > 0 || queue.length > 0);
  if (warmingUp && !loadingTimer) {
    loadingFrame = 0;
    loadingTimer = setInterval(() => {
      loadingFrame = (loadingFrame + 1) % 16;
      void chrome.action.setIcon({ path: iconPaths(`loading-${loadingFrame}`) });
    }, 125);
  } else if (!warmingUp) stopLoadingAnimation();
  void chrome.action.setIcon({ path: iconPaths(warmingUp ? `loading-${loadingFrame}` : enabled ? "on" : "off") });
  chrome.action.setTitle({ title: warmingUp ? "Cyclopes — Warming up" : `Cyclopes — Filter ${enabled ? "ON" : "OFF"}` });
  publishState(warmingUp);
}

function showError() {
  stopLoadingAnimation();
  void chrome.action.setIcon({ path: iconPaths("off") });
  chrome.action.setTitle({ title: "Cyclopes — Inference failed" });
  publishState(false);
}

chrome.storage.local.get({ enabled: false }).then(({ enabled }) => showState(enabled));
chrome.storage.onChanged.addListener((changes) => {
  if (changes.enabled) showState(changes.enabled.newValue);
});

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.contextMenus.removeAll();
  chrome.contextMenus.create({ id: "cyclopes-report-ai", title: "Report as AI", contexts: ["image"] });
  chrome.contextMenus.create({ id: "cyclopes-report-real", title: "Report as Non-AI", contexts: ["image"] });
  const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
  await Promise.allSettled(tabs.map(({ id }) => id === undefined ? undefined : chrome.scripting.executeScript({
    target: { tabId: id },
    files: ["content.js"],
  })));
});

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
    showState(filterEnabled);
    (async () => {
      try {
        await ensureOffscreenDocument();
        job.resolve(await chrome.runtime.sendMessage({ target: "offscreen", type: "infer", url: job.url }));
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
    .then((result) => {
      if (typeof result?.score !== "number") throw new Error(result?.error || "Inference returned no score.");
      return result.score;
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
  await ensureOffscreenDocument();
  const result = await chrome.runtime.sendMessage({ target: "offscreen", type: "thumbnail", url: report.source });
  if (typeof result?.thumbnail !== "string") throw new Error(result?.error || "Thumbnail creation failed.");
  const { feedbackReports = [] } = await chrome.storage.local.get({ feedbackReports: [] });
  await chrome.storage.local.set({
    feedbackReports: [{ ...report, thumbnail: result.thumbnail }, ...feedbackReports.filter(({ source }) => source !== report.source)].slice(0, 500),
  });
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
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

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
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
