import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const shared = await import("../extension/src/shared.js");
const inference = await import("../extension/src/inference.js");

test("the filter boundary is exactly 65 percent", () => {
  assert.equal(shared.AI_THRESHOLD, 0.65);
  assert.ok(Math.abs(inference.calibratedBlend(0.6794350376527292, 0.6794350376527292) - 0.65) < 1e-12);
});

test("the built MV3 package is local and has its inference document", () => {
  const manifest = JSON.parse(readFileSync("dist/manifest.json", "utf8"));
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.background.service_worker, "background.js");
  assert.ok(existsSync("dist/offscreen.html"));
  assert.ok(existsSync("dist/offscreen.js"));
  assert.ok(existsSync("dist/ort"));
  assert.match(readFileSync("dist/offscreen.js", "utf8"), /fetch\(url/);
  assert.doesNotMatch(readFileSync("dist/offscreen.js", "utf8"), /https?:\/\//);
});
