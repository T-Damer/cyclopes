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

test("strong aspect ratios use one centered square view", () => {
  assert.deepEqual(inference.sourceRegion(200, 400), { x: 0, y: 100, width: 200, height: 200 });
  assert.deepEqual(inference.sourceRegion(400, 200), { x: 100, y: 0, width: 200, height: 200 });
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
  assert.match(background, /text: "ERR"/);
  assert.match(readFileSync("dist/content.js", "utf8"), /cyclopesScore/);
  assert.match(readFileSync("dist/content.js", "utf8"), /IntersectionObserver/);
  assert.match(readFileSync("dist/content.js", "utf8"), /Cyclopes \\u2026/);
  assert.match(readFileSync("dist/content.js", "utf8"), /AI.*%/);
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
