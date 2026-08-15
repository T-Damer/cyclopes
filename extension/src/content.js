import {
  BADGE_PLACEMENTS,
  DEFAULT_IMAGE_SETTINGS,
  MIN_RENDERED_AREA,
  badgeObstructionScore,
  badgePlacementOffset,
  badgePlacementRect,
  cssBackgroundUrl,
  isEligibleImageWithSettings,
  isMostlyOccluded,
  normalizeImageSettings,
  roundedCornerInset,
} from "./shared.js";

let states = new WeakMap();
let enabled = false;
let globallyEnabled = false;
let excludedSites = [];
let imageSettings = DEFAULT_IMAGE_SETTINGS;
const badges = new Map();
const anchors = new Map();
const placementTimers = new WeakMap();
const visible = new Set();
const backgroundImages = new WeakMap();
const originalFilters = new WeakMap();
let nextAnchor = 0;
const loadingFrames = ["𓁺", "𓁻", "𓁿", "𓂀"];

document.querySelectorAll(".cyclopes-badge").forEach((badge) => badge.remove());

const style = document.createElement("style");
style.textContent = `
  .cyclopes-badge{position:absolute;white-space:nowrap;padding:3px 6px;border-radius:10px;color:#fff;background:#475569;font:600 11px/1.2 system-ui,sans-serif;pointer-events:none;text-shadow:0 1px 1px #000;box-shadow:0 0 0 1px #fff,0 0 0 2px #000,0 2px 5px #000a}
  .cyclopes-badge[data-verdict="ai"]{background:#dc2626}
  .cyclopes-badge[data-verdict="real"]{background:#2563eb}
  .cyclopes-badge{opacity:0; transition: opacity 140ms ease, transform 140ms ease, left 140ms ease, top 140ms ease; will-change: opacity, transform, left, top}
  .cyclopes-badge[data-visible="1"]{opacity:1}
  .cyclopes-badge[data-loading="1"]{opacity:0.6}
`;
document.documentElement.append(style);

function backgroundSource(element) {
  return cssBackgroundUrl(getComputedStyle(element).backgroundImage);
}

function sourceFor(target) {
  return target instanceof HTMLImageElement ? target.currentSrc : backgroundImages.get(target)?.source;
}

function dimensionsFor(target) {
  if (target instanceof HTMLImageElement) return { width: target.naturalWidth, height: target.naturalHeight };
  const { width = 0, height = 0 } = backgroundImages.get(target) ?? {};
  return { width, height };
}

function eligibilityCandidate(target) {
  if (target instanceof HTMLImageElement) return target;
  const source = sourceFor(target);
  const natural = dimensionsFor(target);
  const rect = target.getBoundingClientRect();
  return {
    currentSrc: source,
    naturalWidth: natural.width,
    naturalHeight: natural.height,
    width: rect.width,
    height: rect.height,
    closest: () => null,
    ownerDocument: target.ownerDocument,
  };
}

function isEligibleTarget(target) {
  return (target instanceof HTMLImageElement || imageSettings.cssBackgrounds) &&
    isEligibleImageWithSettings(eligibilityCandidate(target), imageSettings);
}

function radiusPixels(value, width, height) {
  const parts = value.split(/\s+/);
  const values = parts.flatMap((part, index) => {
    const number = Number.parseFloat(part);
    if (!Number.isFinite(number)) return [0];
    if (!part.endsWith("%")) return [number];
    if (parts.length === 1) return [number * width / 100, number * height / 100];
    return [number * (index ? height : width) / 100];
  });
  return Math.max(...values, 0);
}

function cornerInsets(image, imageRect) {
  const insets = { topLeft: 0, topRight: 0, bottomRight: 0, bottomLeft: 0 };
  // ponytail: the image and its immediate mask cover current layouts; walk ancestors if nested masks appear.
  for (const element of [image, image.parentElement].filter(Boolean)) {
    const rect = element.getBoundingClientRect();
    const computed = getComputedStyle(element);
    const near = (a, b) => Math.abs(a - b) <= 2;
    const inset = (property, borders) => roundedCornerInset(radiusPixels(computed[property], rect.width, rect.height))
      + Math.max(...borders.map((name) => Number.parseFloat(computed[name]) || 0));
    if (near(rect.left, imageRect.left) && near(rect.top, imageRect.top))
      insets.topLeft = Math.max(insets.topLeft, inset("borderTopLeftRadius", ["borderTopWidth", "borderLeftWidth"]));
    if (near(rect.right, imageRect.right) && near(rect.top, imageRect.top))
      insets.topRight = Math.max(insets.topRight, inset("borderTopRightRadius", ["borderTopWidth", "borderRightWidth"]));
    if (near(rect.right, imageRect.right) && near(rect.bottom, imageRect.bottom))
      insets.bottomRight = Math.max(insets.bottomRight, inset("borderBottomRightRadius", ["borderBottomWidth", "borderRightWidth"]));
    if (near(rect.left, imageRect.left) && near(rect.bottom, imageRect.bottom))
      insets.bottomLeft = Math.max(insets.bottomLeft, inset("borderBottomLeftRadius", ["borderBottomWidth", "borderLeftWidth"]));
  }
  return insets;
}

function placementInset(placement, insets) {
  if (placement.ax === 0 && placement.ay === 0) return insets.topLeft;
  if (placement.ax === 1 && placement.ay === 0) return insets.topRight;
  if (placement.ax === 1 && placement.ay === 1) return insets.bottomRight;
  if (placement.ax === 0 && placement.ay === 1) return insets.bottomLeft;
  return 0;
}

function isBadgeShown(image, rect = image.getBoundingClientRect()) {
  return isEligibleTarget(image) && rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
}

function placeBadge(image) {
  const badge = badges.get(image);
  if (!badge) return;
  const rect = image.getBoundingClientRect();
  const shown = isBadgeShown(image, rect);
  badge.dataset.visible = shown ? "1" : "0";
  delete badge.dataset.loading;
  if (!shown) return;
  const width = badge.offsetWidth;
  const height = badge.offsetHeight;
  const insets = cornerInsets(image, rect);
  let best;
  let bestInset = 0;
  let bestScore = Infinity;
  const placements = imageSettings.smartPositioning ? BADGE_PLACEMENTS : BADGE_PLACEMENTS.slice(0, 1);
  for (const placement of placements) {
    const inset = placementInset(placement, insets);
    const area = badgePlacementRect(rect, width, height, placement, inset);
    if (area.left < rect.left || area.top < rect.top || area.right > rect.right || area.bottom > rect.bottom) continue;
    const score = badgeObstructionScore(image, area);
    if (score < bestScore) {
      best = placement;
      bestInset = inset;
      bestScore = score;
      if (score === 0) break;
    }
  }
  if (!best || bestScore > 0) {
    best = BADGE_PLACEMENTS[0];
    bestInset = placementInset(best, insets);
  }
  badge.dataset.position = best.name;
  badge.style.left = `anchor(${best.x})`;
  badge.style.top = `anchor(${best.y})`;
  if (best.rotate) {
    const inset = height / 2 + 3;
    const direction = best.ax === 1 ? -1 : 1;
    badge.style.transform = `translate(calc(-50% + ${direction * inset}px),-50%) rotate(${best.rotate}deg)`;
  } else {
    const offset = badgePlacementOffset(best, bestInset);
    badge.style.transform = `translate(${best.tx * 100}%,${best.ty * 100}%) translate(${best.dx + offset.x}px,${best.dy + offset.y}px)`;
  }
}

function updateBadge(image) {
  const badge = badges.get(image);
  if (!badge) return;
  if (badge.dataset.position) {
    badge.dataset.visible = isBadgeShown(image) ? "1" : "0";
    delete badge.dataset.loading;
    return;
  }
  const shouldShow = isBadgeShown(image);
  badge.dataset.visible = shouldShow ? "1" : "0";
  badge.dataset.loading = shouldShow ? "1" : "0";
  clearTimeout(placementTimers.get(image));
  placementTimers.set(image, setTimeout(() => {
    placementTimers.delete(image);
    placeBadge(image);
  }, 120));
}

function resetBadgePosition(image, immediate = false) {
  const badge = badges.get(image);
  if (!badge) return;
  delete badge.dataset.position;
  clearTimeout(placementTimers.get(image));
  placementTimers.delete(image);
  if (immediate) placeBadge(image);
  else updateBadge(image);
}

function removeBadge(image) {
  clearTimeout(placementTimers.get(image));
  placementTimers.delete(image);
  applyImageBlur(image, 0);
  badges.get(image)?.remove();
  badges.delete(image);
  const previous = anchors.get(image);
  if (previous !== undefined) image.style.setProperty("anchor-name", previous);
  anchors.delete(image);
}

function applyImageBlur(image, blurPx) {
  if (!(image instanceof Element)) return;
  const amount = Number(blurPx);
  if (!Number.isFinite(amount) || amount <= 0) {
    const previous = originalFilters.get(image);
    if (previous === undefined) return;
    image.style.filter = previous;
    originalFilters.delete(image);
    return;
  }
  if (!originalFilters.has(image)) {
    originalFilters.set(image, image.style.filter);
  }
  const previous = originalFilters.get(image);
  const base = previous || "";
  image.style.filter = `${base ? `${base} ` : ""}blur(${amount}px)`;
}

function applyBlurForImage(image) {
  const badge = badges.get(image);
  const isAi = badge?.dataset?.verdict === "ai" && imageSettings.blurAiImages;
  applyImageBlur(image, isAi ? imageSettings.blurLevel : 0);
}

function applyBlurForAll() {
  badges.forEach((_, image) => applyBlurForImage(image));
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
  badge.textContent = loadingFrames[0];
  image.insertAdjacentElement("afterend", badge);
  badges.set(image, badge);
  updateBadge(image);
  return badge;
}

let loadingFrame = 0;
setInterval(() => {
  loadingFrame = (loadingFrame + 1) % loadingFrames.length;
  badges.forEach((badge) => {
    if (badge.dataset.verdict === "loading") badge.textContent = loadingFrames[loadingFrame];
  });
}, 150);

async function analyze(image) {
  if (!enabled || document.hidden || !visible.has(image) || !isEligibleTarget(image)) return;
  const source = sourceFor(image);
  if (states.get(image) === source) return;
  if (isMostlyOccluded(image, innerWidth, innerHeight)) return;
  delete image.dataset.cyclopesError;
  states.set(image, source);
  const badge = badgeFor(image);
  badge.dataset.verdict = "loading";
  badge.textContent = loadingFrames[0];
  try {
    const result = await chrome.runtime.sendMessage({ target: "background", type: "score", url: source });
    if (typeof result?.score !== "number") throw new Error(result?.error || "Inference returned no score.");
    if (!enabled || document.hidden || states.get(image) !== source || !isEligibleTarget(image) || isMostlyOccluded(image, innerWidth, innerHeight)) {
      states.delete(image);
      applyImageBlur(image, 0);
      removeBadge(image);
      return;
    }
    image.dataset.cyclopesScore = result.score.toFixed(4);
    const ai = result.score >= imageSettings.threshold;
    badge.dataset.verdict = ai ? "ai" : "real";
    badge.textContent = `AI ${(result.score * 100).toFixed(0)}%`;
    applyBlurForImage(image);
    updateBadge(image);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message.includes("Extension context invalidated")) {
      enabled = false;
      clear();
      return;
    }
    if (document.hidden || !enabled) {
      states.delete(image);
      removeBadge(image);
      return;
    }
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
  if (!(image instanceof HTMLImageElement) || image.complete) analyze(image);
  else image.addEventListener("load", () => analyze(image), { once: true });
}

function forgetBackground(element) {
  if (!backgroundImages.has(element)) return;
  backgroundImages.delete(element);
  states.delete(element);
  visible.delete(element);
  viewport.unobserve(element);
  removeBadge(element);
}

function watchBackground(element) {
  if (element instanceof HTMLImageElement || element.closest?.(".cyclopes-badge")) return;
  const rect = element.getBoundingClientRect();
  if (rect.width * rect.height < MIN_RENDERED_AREA) {
    forgetBackground(element);
    return;
  }
  const source = backgroundSource(element);
  const previous = backgroundImages.get(element);
  if (!source) {
    forgetBackground(element);
    return;
  }
  if (previous?.source === source) {
    watch(element);
    return;
  }
  states.delete(element);
  backgroundImages.set(element, { source, width: 0, height: 0 });
  watch(element);
  const probe = new Image();
  probe.addEventListener("load", () => {
    if (backgroundImages.get(element)?.source !== source) return;
    backgroundImages.set(element, { source, width: probe.naturalWidth, height: probe.naturalHeight });
    analyze(element);
  }, { once: true });
  probe.src = source;
}

function scanBackgrounds(root = document) {
  if (!imageSettings.cssBackgrounds) return;
  if (root instanceof Element) watchBackground(root);
  root.querySelectorAll?.("*").forEach(watchBackground);
}

function scan(root = document) {
  if (root instanceof HTMLImageElement) watch(root);
  root.querySelectorAll?.("img").forEach(watch);
  scanBackgrounds(root);
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

function applySiteSetting() {
  setEnabled(globallyEnabled && !excludedSites.includes(location.hostname));
}

chrome.storage.local.get({ enabled: false, excludedSites: [], ...DEFAULT_IMAGE_SETTINGS }).then((values) => {
  imageSettings = normalizeImageSettings(values);
  globallyEnabled = values.enabled;
  excludedSites = values.excludedSites;
  applySiteSetting();
});
chrome.storage.onChanged.addListener((changes) => {
  if (changes.enabled) globallyEnabled = changes.enabled.newValue;
  if (changes.excludedSites) excludedSites = changes.excludedSites.newValue ?? [];
  if (changes.enabled || changes.excludedSites) applySiteSetting();
  if (
    changes.minSourceSize ||
    changes.maxAspectRatio ||
    changes.threshold ||
    changes.smartPositioning ||
    changes.cssBackgrounds ||
    changes.blurAiImages ||
    changes.blurLevel
  ) {
    imageSettings = normalizeImageSettings({
      minSourceSize: changes.minSourceSize?.newValue ?? imageSettings.minSourceSize,
      maxAspectRatio: changes.maxAspectRatio?.newValue ?? imageSettings.maxAspectRatio,
      threshold: changes.threshold?.newValue ?? imageSettings.threshold,
      smartPositioning: changes.smartPositioning?.newValue ?? imageSettings.smartPositioning,
      cssBackgrounds: changes.cssBackgrounds?.newValue ?? imageSettings.cssBackgrounds,
      blurAiImages: changes.blurAiImages?.newValue ?? imageSettings.blurAiImages,
      blurLevel: changes.blurLevel?.newValue ?? imageSettings.blurLevel,
      theme: imageSettings.theme,
    });
    if (changes.smartPositioning) badges.forEach((_badge, image) => resetBadgePosition(image));
    if (changes.cssBackgrounds) {
      clear();
      scan();
    }
    if (changes.blurAiImages || changes.blurLevel) {
      applyBlurForAll();
    }
    refresh();
  }
});
new MutationObserver((records) => {
  for (const record of records) {
    if (record.target instanceof Element && record.target.closest(".cyclopes-badge")) continue;
    if (record.type === "attributes") {
      const image = record.target instanceof HTMLImageElement ? record.target : record.target.parentElement?.querySelector("img");
      if (image) {
        resetBadgePosition(image);
        watch(image);
      }
      if (imageSettings.cssBackgrounds && record.target instanceof Element) watchBackground(record.target);
    }
    else record.addedNodes.forEach((node) => node.nodeType === Node.ELEMENT_NODE && scan(node));
  }
  refresh();
}).observe(document.documentElement, { attributes: true, attributeFilter: ["class", "src", "srcset", "style"], childList: true, subtree: true });
addEventListener("scroll", refresh, { passive: true });
addEventListener("resize", () => {
  scanBackgrounds();
  refresh();
}, { passive: true });
addEventListener("animationstart", (event) => {
  if (event.target instanceof HTMLImageElement) resetBadgePosition(event.target, true);
  else event.target.querySelectorAll?.("img").forEach((image) => resetBadgePosition(image, true));
}, true);
for (const type of ["animationend", "transitionend"]) {
  addEventListener(type, (event) => {
    if (event.target instanceof HTMLImageElement) resetBadgePosition(event.target);
    else event.target.querySelectorAll?.("img").forEach((image) => resetBadgePosition(image));
  }, true);
}
addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
