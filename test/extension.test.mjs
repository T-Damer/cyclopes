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

test("single CSS background URLs are accepted without accepting layered backgrounds", () => {
  assert.equal(shared.cssBackgroundUrl('url("https://example.test/image.jpg")'), "https://example.test/image.jpg");
  assert.equal(shared.cssBackgroundUrl("url('/image.webp')"), "/image.webp");
  assert.equal(shared.cssBackgroundUrl("linear-gradient(red, blue), url(image.jpg)"), "");
  assert.equal(shared.normalizeImageSettings({ cssBackgrounds: true }).cssBackgrounds, true);
  assert.equal(shared.normalizeImageSettings({ cssBackgrounds: false }).cssBackgrounds, false);
  assert.equal(shared.normalizeImageSettings({}).cssBackgrounds, true);
});

test("image eligibility uses source area, rendered area, and aspect ratio", () => {
  const image = { currentSrc: "image.jpg", naturalWidth: 512, naturalHeight: 512, width: 32, height: 32 };
  assert.equal(shared.isEligibleImage(image), false);
  assert.equal(shared.isEligibleImage({ ...image, naturalWidth: 255, naturalHeight: 255, width: 255, height: 255 }), false);
  assert.equal(shared.isEligibleImage({ ...image, naturalWidth: 348, naturalHeight: 195, width: 348, height: 195 }), true);
  assert.equal(shared.isEligibleImage({ ...image, naturalWidth: 267, naturalHeight: 356, width: 267, height: 356 }), true);
  assert.equal(shared.isEligibleImage({ ...image, naturalWidth: 1024, naturalHeight: 288, width: 348, height: 98 }), true);
  assert.equal(shared.isEligibleImage({ ...image, naturalWidth: 4096, naturalHeight: 128, width: 512, height: 16 }), false);
  assert.equal(shared.isEligibleImageWithSettings({ ...image, width: 128, height: 128 }, { minSourceSize: 512, maxAspectRatio: 4 }), true);
  assert.equal(shared.isEligibleImageWithSettings({ ...image, width: 128, height: 128 }, { minSourceSize: 513, maxAspectRatio: 4 }), false);
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
  const mediaWrapper = {};
  image.parentElement = { parentElement: mediaWrapper };
  image.ownerDocument.elementFromPoint = () => ({ parentElement: mediaWrapper, firstElementChild: null, textContent: "" });
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

test("rounded image corners inset corner badges", () => {
  assert.equal(shared.roundedCornerInset(0), 0);
  assert.equal(shared.roundedCornerInset(48), 15);
  assert.deepEqual(shared.badgePlacementOffset(shared.BADGE_PLACEMENTS[1], 15), { x: 15, y: 15 });
  assert.deepEqual(
    shared.badgePlacementRect(
      { left: 0, top: 0, width: 200, height: 100 },
      50,
      20,
      shared.BADGE_PLACEMENTS.find(({ name }) => name === "top-left"),
      15,
    ),
    { left: 18, top: 18, right: 68, bottom: 38, width: 50, height: 20 },
  );
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
  assert.ok(manifest.permissions.includes("scripting"));
  assert.ok(manifest.permissions.includes("activeTab"));
  assert.ok(manifest.permissions.includes("contextMenus"));
  assert.equal(manifest.background.service_worker, "background.js");
  assert.equal(manifest.action.default_popup, "options.html");
  assert.equal(manifest.options_ui.page, "options.html");
  assert.doesNotMatch(background, /action\.onClicked/);
  assert.match(background, /MAX_INFERENCE_CONCURRENCY = 1/);
  assert.match(background, /active > 0 \|\| queue\.length > 0/);
  assert.doesNotMatch(background, /enabled && !ready/);
  assert.match(background, /Warming up/);
  assert.match(background, /loading-/);
  assert.match(background, /% 16/);
  assert.match(background, /}, 125\)/);
  assert.match(background, /feedbackReports/);
  assert.match(background, /cyclopes-report-ai/);
  assert.match(background, /cyclopes-report-real/);
  assert.match(background, /contextMenus\.onClicked/);
  assert.match(background, /runtime\.onInstalled/);
  assert.match(background, /scripting\.executeScript/);
  assert.match(background, /setInterval/);
  assert.doesNotMatch(background, /setBadgeBackgroundColor/);
  const content = readFileSync("dist/content.js", "utf8");
  assert.match(content, /cyclopesScore/);
  assert.match(content, /IntersectionObserver/);
  assert.match(content, /document\.hidden/);
  assert.match(content, /visibilitychange/);
  assert.match(content, /loadingFrames/);
  assert.match(content, /\\u\{1307A\}/);
  assert.match(content, /150/);
  assert.match(content, /Extension context invalidated/);
  assert.match(content, /white-space:nowrap/);
  assert.match(content, /0 0 0 1px #fff,0 0 0 2px #000/);
  assert.match(content, /border-radius:10px/);
  assert.match(content, /anchor-name/);
  assert.match(content, /rotate\(/);
  assert.match(content, /setTimeout\(\(\) => \{/);
  assert.match(content, /}, 120\)/);
  assert.match(content, /badge\.dataset\.position/);
  assert.match(content, /animationstart/);
  assert.match(content, /transitionend/);
  assert.match(content, /AI.*%/);
  assert.doesNotMatch(content, /type: "report"|Click to report this image/);
  assert.match(content, /excludedSites/);
  assert.match(content, /cssBackgrounds/);
  assert.doesNotMatch(content, /blur\(/);
  assert.doesNotMatch(content, /2147483647/);
  assert.equal(existsSync("dist/popup.html"), false);
  assert.ok(existsSync("dist/offscreen.html"));
  assert.ok(existsSync("dist/offscreen.js"));
  assert.ok(existsSync("dist/options.html"));
  assert.ok(existsSync("dist/options.js"));
  assert.match(readFileSync("dist/options.css", "utf8"), /\[hidden\]\{display:none!important\}/);
  assert.match(readFileSync("dist/options.js", "utf8"), /openSections/);
  assert.match(readFileSync("dist/options.js", "utf8"), /Loading local model/);
  assert.match(readFileSync("dist/options.css", "utf8"), /state-spin/);
  assert.match(readFileSync("dist/options.css", "utf8"), /width:330px;height:560px/);
  assert.match(readFileSync("dist/options.html", "utf8"), /Smart badge position/);
  assert.match(readFileSync("dist/options.html", "utf8"), /AI threshold/);
  assert.match(readFileSync("dist/options.html", "utf8"), /CSS background images/);
  assert.match(readFileSync("dist/options.html", "utf8"), /Manage excluded sites/);
  assert.match(readFileSync("dist/options.html", "utf8"), /<details id="detection-section">/);
  assert.match(readFileSync("dist/options.html", "utf8"), /reports-tooltip/);
  assert.ok(existsSync("dist/icons/off-128.png"));
  assert.ok(existsSync("dist/icons/on-128.png"));
  assert.ok(existsSync("dist/icons/loading-3-32.png"));
  assert.ok(existsSync("dist/icons/loading-15-32.png"));
  assert.match(readFileSync("dist/icons/loading-0.svg", "utf8"), /<path/);
  assert.doesNotMatch(readFileSync("dist/icons/loading-0.svg", "utf8"), /<text/);
  assert.equal(existsSync("dist/logo.webp"), false);
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
  assert.match(offscreen, /convertToBlob/);
  assert.doesNotMatch(offscreen, /Promise\.all\(\[createSession/);
  assert.doesNotMatch(offscreen, /https?:\/\//);
});
