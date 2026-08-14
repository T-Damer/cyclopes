export const AI_THRESHOLD = 0.65;

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
      if (top === image || top?.classList?.contains("cyclopes-badge")) visible += 1;
    }
  }
  return visible < 4;
}

export function isEligibleImage(image) {
  return Boolean(
    image.currentSrc &&
      image.naturalWidth >= 64 &&
      image.naturalHeight >= 64 &&
      image.width >= 64 &&
      image.height >= 64 &&
      !isVideoPoster(image)
  );
}
