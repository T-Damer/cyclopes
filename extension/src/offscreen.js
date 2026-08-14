import { inferBlob } from "./inference.js";

const MAX_CACHE_ENTRIES = 256;
const scores = new Map();

async function digest(blob) {
  const bytes = await blob.arrayBuffer();
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function inferUrl(url) {
  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) throw new Error(`Image request failed (${response.status}).`);
  const blob = await response.blob();
  if (!blob.type.startsWith("image/")) throw new Error("Displayed resource is not an image.");
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

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "offscreen" || message.type !== "infer") return;
  inferUrl(message.url).then(
    (score) => sendResponse({ score }),
    (error) => sendResponse({ error: error instanceof Error ? error.message : String(error) })
  );
  return true;
});
