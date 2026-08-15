export const AI_THRESHOLD = 0.65;
export const DEFAULT_IMAGE_SETTINGS = Object.freeze({
  minSourceSize: 256,
  maxAspectRatio: 4,
  threshold: AI_THRESHOLD,
  smartPositioning: true,
  cssBackgrounds: true,
  theme: "system",
});
export const MIN_RENDERED_AREA = 96 * 96;

export const BADGE_PLACEMENTS = [
  { name: "top-right", x: "right", y: "top", ax: 1, ay: 0, tx: -1, ty: 0, dx: -3, dy: 3 },
  { name: "top-left", x: "left", y: "top", ax: 0, ay: 0, tx: 0, ty: 0, dx: 3, dy: 3 },
  { name: "bottom-right", x: "right", y: "bottom", ax: 1, ay: 1, tx: -1, ty: -1, dx: -3, dy: -3 },
  { name: "bottom-left", x: "left", y: "bottom", ax: 0, ay: 1, tx: 0, ty: -1, dx: 3, dy: -3 },
  { name: "top-center", x: "center", y: "top", ax: 0.5, ay: 0, tx: -0.5, ty: 0, dx: 0, dy: 3 },
  { name: "bottom-center", x: "center", y: "bottom", ax: 0.5, ay: 1, tx: -0.5, ty: -1, dx: 0, dy: -3 },
  { name: "right-center", x: "right", y: "center", ax: 1, ay: 0.5, rotate: 90 },
  { name: "left-center", x: "left", y: "center", ax: 0, ay: 0.5, rotate: -90 },
  { name: "top-right-edge", x: "right", y: "top", ax: 1, ay: 0, tx: -2, ty: 0, dx: -6, dy: 3 },
  { name: "top-left-edge", x: "left", y: "top", ax: 0, ay: 0, tx: 1, ty: 0, dx: 6, dy: 3 },
  { name: "bottom-right-edge", x: "right", y: "bottom", ax: 1, ay: 1, tx: -2, ty: -1, dx: -6, dy: -3 },
  { name: "bottom-left-edge", x: "left", y: "bottom", ax: 0, ay: 1, tx: 1, ty: -1, dx: 6, dy: -3 },
  { name: "right-top-edge", x: "right", y: "top", ax: 1, ay: 0, tx: -1, ty: 1, dx: -3, dy: 6 },
  { name: "left-top-edge", x: "left", y: "top", ax: 0, ay: 0, tx: 0, ty: 1, dx: 3, dy: 6 },
  { name: "right-bottom-edge", x: "right", y: "bottom", ax: 1, ay: 1, tx: -1, ty: -2, dx: -3, dy: -6 },
  { name: "left-bottom-edge", x: "left", y: "bottom", ax: 0, ay: 1, tx: 0, ty: -2, dx: 3, dy: -6 },
  { name: "top-right-diagonal", x: "right", y: "top", ax: 1, ay: 0, tx: -2, ty: 1, dx: -6, dy: 6 },
  { name: "top-left-diagonal", x: "left", y: "top", ax: 0, ay: 0, tx: 1, ty: 1, dx: 6, dy: 6 },
  { name: "bottom-right-diagonal", x: "right", y: "bottom", ax: 1, ay: 1, tx: -2, ty: -2, dx: -6, dy: -6 },
  { name: "bottom-left-diagonal", x: "left", y: "bottom", ax: 0, ay: 1, tx: 1, ty: -2, dx: 6, dy: -6 },
];

export function roundedCornerInset(radius) {
  return Math.ceil(Math.max(0, radius) * (1 - Math.SQRT1_2));
}

export function cssBackgroundUrl(value) {
  const match = value?.match(/^url\((?:"([^"]+)"|'([^']+)'|([^)]*))\)$/);
  return match ? (match[1] || match[2] || match[3]).trim() : "";
}

export function badgePlacementOffset(placement, cornerInset = 0) {
  const atCorner = (placement.ax === 0 || placement.ax === 1) && (placement.ay === 0 || placement.ay === 1);
  return {
    x: atCorner ? (placement.ax ? -cornerInset : cornerInset) : 0,
    y: atCorner ? (placement.ay ? -cornerInset : cornerInset) : 0,
  };
}

export function badgePlacementRect(imageRect, badgeWidth, badgeHeight, placement, cornerInset = 0) {
  if (placement.rotate) {
    const left = placement.ax === 1 ? imageRect.right - badgeHeight - 3 : imageRect.left + 3;
    const top = imageRect.top + (imageRect.height - badgeWidth) / 2;
    return { left, top, right: left + badgeHeight, bottom: top + badgeWidth, width: badgeHeight, height: badgeWidth };
  }
  const offset = badgePlacementOffset(placement, cornerInset);
  const left = imageRect.left + imageRect.width * placement.ax + badgeWidth * placement.tx + placement.dx + offset.x;
  const top = imageRect.top + imageRect.height * placement.ay + badgeHeight * placement.ty + placement.dy + offset.y;
  return { left, top, right: left + badgeWidth, bottom: top + badgeHeight, width: badgeWidth, height: badgeHeight };
}

function belongsToImage(image, element) {
  if (element === image || element?.contains?.(image)) return true;
  const mediaWrapper = image.parentElement?.parentElement;
  return Boolean(
    mediaWrapper &&
    element?.parentElement === mediaWrapper &&
    !element.firstElementChild &&
    !element.textContent?.trim()
  );
}

function elementBelowBadge(document, x, y) {
  return document.elementsFromPoint?.(x, y)
    .find((element) => !element.classList?.contains("cyclopes-badge")) ?? document.elementFromPoint(x, y);
}

export function badgeObstructionScore(image, rect) {
  const points = [
    [rect.left + 1, rect.top + 1],
    [rect.right - 1, rect.top + 1],
    [rect.left + 1, rect.bottom - 1],
    [rect.right - 1, rect.bottom - 1],
    [rect.left + rect.width / 2, rect.top + rect.height / 2],
  ];
  return points.reduce((blocked, [x, y]) => {
    const top = elementBelowBadge(image.ownerDocument, x, y);
    return blocked + Number(!belongsToImage(image, top));
  }, 0);
}

export function isVideoPoster(image) {
  if (image.closest?.("video, [data-testid='videoPlayer'], [data-video-player]")) return true;
  return Array.from(image.ownerDocument?.querySelectorAll?.("video[poster]") ?? [])
    .some((video) => video.poster === image.currentSrc);
}

export function isMostlyOccluded(image, viewportWidth, viewportHeight) {
  const rect = image.getBoundingClientRect();
  let visible = 0;
  for (let row = 0; row < 5; row += 1) {
    for (let column = 0; column < 5; column += 1) {
      const x = rect.left + rect.width * (column + 0.5) / 5;
      const y = rect.top + rect.height * (row + 0.5) / 5;
      if (x < 0 || y < 0 || x >= viewportWidth || y >= viewportHeight) continue;
      const top = elementBelowBadge(image.ownerDocument, x, y);
      if (belongsToImage(image, top)) visible += 1;
    }
  }
  return visible < 4;
}

export function isEligibleImage(image) {
  return isEligibleImageWithSettings(image, DEFAULT_IMAGE_SETTINGS);
}

export function normalizeImageSettings(settings = {}) {
  const minSourceSize = Math.min(1024, Math.max(64, Number(settings.minSourceSize) || DEFAULT_IMAGE_SETTINGS.minSourceSize));
  const maxAspectRatio = Math.min(10, Math.max(1, Number(settings.maxAspectRatio) || DEFAULT_IMAGE_SETTINGS.maxAspectRatio));
  const threshold = Math.min(0.95, Math.max(0.5, Number(settings.threshold) || DEFAULT_IMAGE_SETTINGS.threshold));
  const smartPositioning = settings.smartPositioning !== false;
  const cssBackgrounds = settings.cssBackgrounds !== false;
  const theme = ["system", "light", "dark"].includes(settings.theme) ? settings.theme : DEFAULT_IMAGE_SETTINGS.theme;
  return { minSourceSize, maxAspectRatio, threshold, smartPositioning, cssBackgrounds, theme };
}

export function isEligibleImageWithSettings(image, settings) {
  const { minSourceSize, maxAspectRatio } = normalizeImageSettings(settings);
  const sourceArea = image.naturalWidth * image.naturalHeight;
  const renderedArea = image.width * image.height;
  const aspectRatio = Math.max(image.naturalWidth, image.naturalHeight) / Math.max(1, Math.min(image.naturalWidth, image.naturalHeight));
  return Boolean(
    image.currentSrc &&
      sourceArea >= minSourceSize ** 2 &&
      renderedArea >= MIN_RENDERED_AREA &&
      aspectRatio <= maxAspectRatio &&
      !isVideoPoster(image)
  );
}
