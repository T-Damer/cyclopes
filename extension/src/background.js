const MAX_INFERENCE_CONCURRENCY = 2;
const MAX_CACHE_ENTRIES = 256;
const cachedScores = new Map();
const inFlight = new Map();
const queue = [];
let active = 0;
let offscreenReady;

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
  if (cachedScores.has(url)) {
    const score = cachedScores.get(url);
    cachedScores.delete(url);
    cachedScores.set(url, score);
    return Promise.resolve(score);
  }
  if (inFlight.has(url)) return inFlight.get(url);
  const task = schedule(url)
    .then((result) => {
      if (typeof result?.score !== "number") throw new Error(result?.error || "Inference returned no score.");
      cachedScores.set(url, result.score);
      while (cachedScores.size > MAX_CACHE_ENTRIES) cachedScores.delete(cachedScores.keys().next().value);
      return result.score;
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
