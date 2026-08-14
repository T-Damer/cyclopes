import { AI_THRESHOLD, isEligibleImage } from "./shared.js";

let states = new WeakMap();
let enabled = false;
const badges = new Map();
const visible = new Set();

const style = document.createElement("style");
style.textContent = `
  .cyclopes-badge{position:fixed;z-index:2147483647;padding:3px 6px;border-radius:5px;color:#fff;background:#475569;font:600 11px/1.2 system-ui,sans-serif;pointer-events:none;box-shadow:0 1px 4px #0008}
  .cyclopes-badge[data-verdict="ai"]{background:#dc2626}
  .cyclopes-badge[data-verdict="real"]{background:#2563eb}
`;
document.documentElement.append(style);

function positionBadge(image) {
  const badge = badges.get(image);
  if (!badge) return;
  const rect = image.getBoundingClientRect();
  const shown = isEligibleImage(image) && rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
  badge.hidden = !shown;
  if (!shown) return;
  badge.style.top = `${Math.max(3, rect.top + 3)}px`;
  badge.style.left = `${Math.max(3, Math.min(innerWidth - badge.offsetWidth - 3, rect.right - badge.offsetWidth - 3))}px`;
}

let positioning = false;
function positionBadges() {
  if (positioning) return;
  positioning = true;
  requestAnimationFrame(() => {
    positioning = false;
    badges.forEach((badge, image) => {
      if (!image.isConnected) {
        badge.remove();
        badges.delete(image);
      } else positionBadge(image);
    });
  });
}

function badgeFor(image) {
  let badge = badges.get(image);
  if (badge) return badge;
  badge = document.createElement("span");
  badge.className = "cyclopes-badge";
  badge.textContent = "Cyclopes …";
  document.documentElement.append(badge);
  badges.set(image, badge);
  positionBadge(image);
  return badge;
}

async function analyze(image) {
  if (!enabled || !visible.has(image) || !isEligibleImage(image)) return;
  const source = image.currentSrc;
  if (states.get(image) === source) return;
  delete image.dataset.cyclopesError;
  states.set(image, source);
  const badge = badgeFor(image);
  badge.dataset.verdict = "loading";
  badge.textContent = "Cyclopes …";
  positionBadge(image);
  try {
    const result = await chrome.runtime.sendMessage({ target: "background", type: "score", url: source });
    if (typeof result?.score !== "number") throw new Error(result?.error || "Inference returned no score.");
    if (!enabled || states.get(image) !== source || !isEligibleImage(image)) return;
    image.dataset.cyclopesScore = result.score.toFixed(4);
    const ai = result.score >= AI_THRESHOLD;
    badge.dataset.verdict = ai ? "ai" : "real";
    badge.textContent = `AI ${(result.score * 100).toFixed(0)}%`;
    positionBadge(image);
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
  positionBadges();
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
  badges.forEach((badge) => badge.remove());
  badges.clear();
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
    if (record.type === "attributes") {
      const image = record.target instanceof HTMLImageElement ? record.target : record.target.parentElement?.querySelector("img");
      if (image) watch(image);
    }
    else record.addedNodes.forEach((node) => node.nodeType === Node.ELEMENT_NODE && scan(node));
  }
}).observe(document.documentElement, { attributes: true, attributeFilter: ["src", "srcset"], childList: true, subtree: true });
addEventListener("scroll", positionBadges, { passive: true });
addEventListener("resize", positionBadges, { passive: true });
