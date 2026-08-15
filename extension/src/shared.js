export const AI_THRESHOLD = 0.65;

export const BADGE_PLACEMENTS = [
  { name: "top-right", x: "right", y: "top", ax: 1, ay: 0, tx: -1, ty: 0, dx: -3, dy: 3 },
  { name: "top-left", x: "left", y: "top", ax: 0, ay: 0, tx: 0, ty: 0, dx: 3, dy: 3 },
  { name: "bottom-right", x: "right", y: "bottom", ax: 1, ay: 1, tx: -1, ty: -1, dx: -3, dy: -3 },
  { name: "bottom-left", x: "left", y: "bottom", ax: 0, ay: 1, tx: 0, ty: -1, dx: 3, dy: -3 },
  { name: "top-center", x: "center", y: "top", ax: 0.5, ay: 0, tx: -0.5, ty: 0, dx: 0, dy: 3 },
  { name: "bottom-center", x: "center", y: "bottom", ax: 0.5, ay: 1, tx: -0.5, ty: -1, dx: 0, dy: -3 },
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

export function badgePlacementRect(imageRect, badgeWidth, badgeHeight, placement) {
  const left = imageRect.left + imageRect.width * placement.ax + badgeWidth * placement.tx + placement.dx;
  const top = imageRect.top + imageRect.height * placement.ay + badgeHeight * placement.ty + placement.dy;
  return { left, top, right: left + badgeWidth, bottom: top + badgeHeight, width: badgeWidth, height: badgeHeight };
}

function belongsToImage(image, element) {
  return element === image || element?.contains?.(image) || element?.classList?.contains("cyclopes-badge");
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
    const top = image.ownerDocument.elementFromPoint(x, y);
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
      const top = image.ownerDocument.elementFromPoint(x, y);
      if (belongsToImage(image, top)) visible += 1;
    }
  }
  return visible < 4;
}

export function isEligibleImage(image) {
  return Boolean(
    image.currentSrc &&
      image.naturalWidth >= 256 &&
      image.naturalHeight >= 256 &&
      image.width >= 256 &&
      image.height >= 256 &&
      !isVideoPoster(image)
  );
}
