import { AI_THRESHOLD, isEligibleImage } from "./shared.js";

let states = new WeakMap();
let enabled = false;
const style = document.createElement("style");
style.textContent = ".cyclopes-ai{filter:blur(20px)!important}";
document.documentElement.append(style);

async function analyze(image) {
  if (!enabled || !isEligibleImage(image)) return;
  const source = image.currentSrc;
  if (states.get(image) === source) return;
  image.classList.remove("cyclopes-ai");
  states.set(image, source);
  try {
    const result = await chrome.runtime.sendMessage({ target: "background", type: "score", url: source });
    if (typeof result?.score !== "number") throw new Error(result?.error || "Inference returned no score.");
    if (states.get(image) !== source) return;
    image.dataset.cyclopesScore = result.score.toFixed(4);
    image.classList.toggle("cyclopes-ai", result.score >= AI_THRESHOLD);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    image.dataset.cyclopesError = message;
    console.warn(`Cyclopes: ${message}`);
  }
}

function watch(image) {
  if (image.complete) analyze(image);
  else image.addEventListener("load", () => analyze(image), { once: true });
}

function scan(root = document) {
  if (root instanceof HTMLImageElement) watch(root);
  root.querySelectorAll?.("img").forEach(watch);
}

function setEnabled(value) {
  enabled = value;
  document.documentElement.dataset.cyclopesEnabled = String(enabled);
  if (!enabled) {
    states = new WeakMap();
    document.querySelectorAll(".cyclopes-ai").forEach((image) => image.classList.remove("cyclopes-ai"));
  }
  if (enabled) scan();
}

chrome.storage.local.get({ enabled: false }).then(({ enabled: value }) => setEnabled(value));
chrome.storage.onChanged.addListener((changes) => {
  if (changes.enabled) setEnabled(changes.enabled.newValue);
});
new MutationObserver((records) => {
  for (const record of records) {
    if (record.type === "attributes") watch(record.target);
    else record.addedNodes.forEach((node) => node.nodeType === Node.ELEMENT_NODE && scan(node));
  }
}).observe(document.documentElement, { attributes: true, attributeFilter: ["src", "srcset"], childList: true, subtree: true });
