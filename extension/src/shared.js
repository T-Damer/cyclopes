export const AI_THRESHOLD = 0.65;

export function isVideoPoster(image) {
  if (image.closest?.("video, [data-testid='videoPlayer'], [data-video-player]")) return true;
  return Array.from(image.ownerDocument?.querySelectorAll?.("video[poster]") ?? [])
    .some((video) => video.poster === image.currentSrc);
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
