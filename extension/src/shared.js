export const AI_THRESHOLD = 0.65;

export function isEligibleImage(image) {
  return Boolean(
    image.currentSrc &&
      image.naturalWidth >= 64 &&
      image.naturalHeight >= 64 &&
      image.width >= 64 &&
      image.height >= 64
  );
}
