import { inferBlob } from "./inference.js";

const MAX_CACHE_ENTRIES = 256;
const scores = new Map();

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
  return Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, "0")).join("");
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
      binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
    }
    return `data:image/webp;base64,${btoa(binary)}`;
  } finally {
    bitmap.close();
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "offscreen") return;
  const task = message.type === "infer" ? inferUrl(message.url)
    : message.type === "thumbnail" ? thumbnailUrl(message.url)
    : undefined;
  if (!task) return;
  task.then(
    (result) => sendResponse(message.type === "infer" ? { score: result } : { thumbnail: result }),
    (error) => sendResponse({ error: error instanceof Error ? error.message : String(error) })
  );
  return true;
});
