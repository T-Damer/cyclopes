import { AI_THRESHOLD, isEligibleImage, isMostlyOccluded } from "./shared.js";

let states = new WeakMap();
let enabled = false;
const badges = new Map();
const anchors = new Map();
const visible = new Set();
let nextAnchor = 0;
const loadingFrames = ["|", "/", "-", "\\"];

const style = document.createElement("style");
style.textContent = `
  .cyclopes-badge{position:absolute;top:anchor(top);left:anchor(right);transform:translate(calc(-100% - 3px),3px);padding:3px 6px;border-radius:5px;color:#fff;background:#475569;font:600 11px/1.2 system-ui,sans-serif;pointer-events:none;box-shadow:0 1px 4px #0008}
  .cyclopes-badge[data-verdict="ai"]{background:#dc2626}
  .cyclopes-badge[data-verdict="real"]{background:#2563eb}
`;
document.documentElement.append(style);

function updateBadge(image) {
  const badge = badges.get(image);
  if (!badge) return;
  const rect = image.getBoundingClientRect();
  const shown = isEligibleImage(image) && rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
  badge.hidden = !shown;
}

function removeBadge(image) {
  badges.get(image)?.remove();
  badges.delete(image);
  const previous = anchors.get(image);
  if (previous !== undefined) image.style.setProperty("anchor-name", previous);
  anchors.delete(image);
}

let refreshing = false;
function refresh() {
  if (refreshing) return;
  refreshing = true;
  requestAnimationFrame(() => {
    refreshing = false;
    badges.forEach((_badge, image) => {
      if (!image.isConnected) {
        removeBadge(image);
      } else updateBadge(image);
    });
    visible.forEach(analyze);
  });
}

function badgeFor(image) {
  let badge = badges.get(image);
  if (badge) return badge;
  const anchor = `--cyclopes-${nextAnchor += 1}`;
  anchors.set(image, image.style.getPropertyValue("anchor-name"));
  image.style.setProperty("anchor-name", anchor);
  badge = document.createElement("span");
  badge.className = "cyclopes-badge";
  badge.style.setProperty("position-anchor", anchor);
  badge.style.zIndex = getComputedStyle(image).zIndex;
  badge.textContent = "Cyclopes |";
  image.insertAdjacentElement("afterend", badge);
  badges.set(image, badge);
  updateBadge(image);
  return badge;
}

let loadingFrame = 0;
setInterval(() => {
  loadingFrame = (loadingFrame + 1) % loadingFrames.length;
  badges.forEach((badge) => {
    if (badge.dataset.verdict === "loading") badge.textContent = `Cyclopes ${loadingFrames[loadingFrame]}`;
  });
}, 160);

async function analyze(image) {
  if (!enabled || !visible.has(image) || !isEligibleImage(image)) return;
  const source = image.currentSrc;
  if (states.get(image) === source) return;
  if (isMostlyOccluded(image, innerWidth, innerHeight)) return;
  delete image.dataset.cyclopesError;
  states.set(image, source);
  const badge = badgeFor(image);
  badge.dataset.verdict = "loading";
  badge.textContent = "Cyclopes |";
  try {
    const result = await chrome.runtime.sendMessage({ target: "background", type: "score", url: source });
    if (typeof result?.score !== "number") throw new Error(result?.error || "Inference returned no score.");
    if (!enabled || states.get(image) !== source || !isEligibleImage(image) || isMostlyOccluded(image, innerWidth, innerHeight)) {
      states.delete(image);
      removeBadge(image);
      return;
    }
    image.dataset.cyclopesScore = result.score.toFixed(4);
    const ai = result.score >= AI_THRESHOLD;
    badge.dataset.verdict = ai ? "ai" : "real";
    badge.textContent = `AI ${(result.score * 100).toFixed(0)}%`;
    updateBadge(image);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    image.dataset.cyclopesError = message;
    badge.textContent = "Cyclopes ERR";
    console.warn(`Cyclopes: ${message}`);
  }
}

const viewport = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      visible.add(entry.target);
      analyze(entry.target);
    } else visible.delete(entry.target);
  }
  refresh();
}, { rootMargin: "200px" });

function watch(image) {
  viewport.observe(image);
  if (image.complete && visible.has(image)) analyze(image);
  else image.addEventListener("load", () => analyze(image), { once: true });
}

function scan(root = document) {
  if (root instanceof HTMLImageElement) watch(root);
  root.querySelectorAll?.("img").forEach(watch);
}

function clear() {
  states = new WeakMap();
  Array.from(badges.keys()).forEach(removeBadge);
}

function setEnabled(value) {
  enabled = value;
  document.documentElement.dataset.cyclopesEnabled = String(enabled);
  if (enabled) scan();
  else clear();
}

chrome.storage.local.get({ enabled: false }).then(({ enabled: value }) => setEnabled(value));
chrome.storage.onChanged.addListener((changes) => {
  if (changes.enabled) setEnabled(changes.enabled.newValue);
});
new MutationObserver((records) => {
  for (const record of records) {
    if (record.target instanceof Element && record.target.closest(".cyclopes-badge")) continue;
    if (record.type === "attributes") {
      const image = record.target instanceof HTMLImageElement ? record.target : record.target.parentElement?.querySelector("img");
      if (image) watch(image);
    }
    else record.addedNodes.forEach((node) => node.nodeType === Node.ELEMENT_NODE && scan(node));
  }
  refresh();
}).observe(document.documentElement, { attributes: true, attributeFilter: ["src", "srcset"], childList: true, subtree: true });
addEventListener("scroll", refresh, { passive: true });
addEventListener("resize", refresh, { passive: true });
