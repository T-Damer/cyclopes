import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const shared = await import("../extension/src/shared.js");
const inference = await import("../extension/src/inference.js");

test("the filter boundary is exactly 65 percent", () => {
  assert.equal(shared.AI_THRESHOLD, 0.65);
  assert.ok(Math.abs(inference.outputToScore({ data: [Math.log(0.65 / 0.35)] }) - 0.65) < 1e-12);
});

test("images smaller than 256 pixels are ignored even when their source is large", () => {
  const image = { currentSrc: "image.jpg", naturalWidth: 512, naturalHeight: 512, width: 32, height: 32 };
  assert.equal(shared.isEligibleImage(image), false);
  assert.equal(shared.isEligibleImage({ ...image, width: 255, height: 256 }), false);
  assert.equal(shared.isEligibleImage({ ...image, naturalWidth: 255, width: 256, height: 256 }), false);
  assert.equal(shared.isEligibleImage({ ...image, width: 256, height: 256 }), true);
});

test("video poster images are ignored", () => {
  const image = {
    currentSrc: "https://example.test/poster.jpg",
    naturalWidth: 512,
    naturalHeight: 512,
    width: 512,
    height: 512,
    closest: () => null,
    ownerDocument: { querySelectorAll: () => [{ poster: "https://example.test/poster.jpg" }] },
  };
  assert.equal(shared.isEligibleImage(image), false);
  assert.equal(shared.isEligibleImage({ ...image, currentSrc: "https://example.test/photo.jpg" }), true);
});

test("images with more than 85 percent sampled occlusion are ignored", () => {
  const image = {
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
    ownerDocument: {},
  };
  let samples = 0;
  image.ownerDocument.elementFromPoint = () => samples++ < 3 ? image : {};
  assert.equal(shared.isMostlyOccluded(image, 100, 100), true);
  samples = 0;
  image.ownerDocument.elementFromPoint = () => samples++ < 4 ? image : {};
  assert.equal(shared.isMostlyOccluded(image, 100, 100), false);
  image.ownerDocument.elementFromPoint = () => ({ contains: (element) => element === image });
  assert.equal(shared.isMostlyOccluded(image, 100, 100), false);
});

test("badge placement checks its four corners and center", () => {
  const placement = shared.BADGE_PLACEMENTS[0];
  const rect = shared.badgePlacementRect({ left: 0, top: 0, width: 200, height: 100 }, 50, 20, placement);
  assert.deepEqual(rect, { left: 147, top: 3, right: 197, bottom: 23, width: 50, height: 20 });
  const image = { ownerDocument: { elementFromPoint: (x) => x < 150 ? {} : image } };
  assert.equal(shared.badgeObstructionScore(image, rect), 2);
  image.ownerDocument.elementFromPoint = () => ({ contains: (element) => element === image });
  assert.equal(shared.badgeObstructionScore(image, rect), 0);
  const side = shared.BADGE_PLACEMENTS.find(({ name }) => name === "right-center");
  assert.deepEqual(
    shared.badgePlacementRect({ left: 0, top: 0, right: 200, width: 200, height: 100 }, 50, 20, side),
    { left: 177, top: 25, right: 197, bottom: 75, width: 20, height: 50 },
  );
  const badge = { classList: { contains: (name) => name === "cyclopes-badge" } };
  image.ownerDocument.elementsFromPoint = () => [badge, {}];
  assert.equal(shared.badgeObstructionScore(image, rect), 5);
});

test("preprocessing preserves the complete displayed image", () => {
  assert.deepEqual(inference.sourceRegion(200, 400), { x: 0, y: 0, width: 200, height: 400 });
  assert.deepEqual(inference.sourceRegion(400, 200), { x: 0, y: 0, width: 400, height: 200 });
  assert.deepEqual(inference.sourceRegion(120, 100), { x: 0, y: 0, width: 120, height: 100 });
});

test("the built MV3 package is local and has its inference document", () => {
  const manifest = JSON.parse(readFileSync("dist/manifest.json", "utf8"));
  const background = readFileSync("dist/background.js", "utf8");
  const offscreen = readFileSync("dist/offscreen.js", "utf8");
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.background.service_worker, "background.js");
  assert.equal(manifest.action.default_popup, undefined);
  assert.match(background, /action\.onClicked/);
  assert.match(background, /MAX_INFERENCE_CONCURRENCY = 1/);
  assert.match(background, /Warming up/);
  assert.match(background, /loadingFrames/);
  assert.match(background, /\\u\{1307A\}/);
  assert.match(background, /setInterval/);
  assert.doesNotMatch(background, /text: "…"/);
  assert.match(background, /text: "ERR"/);
  const content = readFileSync("dist/content.js", "utf8");
  assert.match(content, /cyclopesScore/);
  assert.match(content, /IntersectionObserver/);
  assert.match(content, /document\.hidden/);
  assert.match(content, /visibilitychange/);
  assert.match(content, /loadingFrames/);
  assert.match(content, /\\u\{1307A\}/);
  assert.match(content, /150/);
  assert.match(content, /white-space:nowrap/);
  assert.match(content, /border-radius:10px/);
  assert.match(content, /anchor-name/);
  assert.match(content, /rotate\(/);
  assert.match(content, /AI.*%/);
  assert.doesNotMatch(content, /blur\(/);
  assert.doesNotMatch(content, /2147483647/);
  assert.equal(existsSync("dist/popup.html"), false);
  assert.ok(existsSync("dist/offscreen.html"));
  assert.ok(existsSync("dist/offscreen.js"));
  assert.ok(existsSync("dist/models/cyclopes.onnx"));
  assert.equal(existsSync("dist/models/cyclopes-fastvit.onnx"), false);
  assert.equal(existsSync("dist/models/cyclopes-sentry.onnx"), false);
  const model = readFileSync("dist/models/cyclopes.onnx");
  const metadata = JSON.parse(readFileSync("dist/models/cyclopes.json", "utf8"));
  assert.equal(model.byteLength, metadata.size_bytes);
  assert.equal(createHash("sha256").update(model).digest("hex"), metadata.sha256);
  assert.equal(metadata.threshold, shared.AI_THRESHOLD);
  assert.ok(existsSync("dist/ort/ort-wasm-simd-threaded.jsep.wasm"));
  assert.ok(existsSync("dist/ort/ort-wasm-simd-threaded.wasm"));
  assert.match(offscreen, /fetch\(url/);
  assert.doesNotMatch(offscreen, /Promise\.all\(\[createSession/);
  assert.doesNotMatch(offscreen, /https?:\/\//);
});
