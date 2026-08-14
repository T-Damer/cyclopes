import { build } from "esbuild";
import { cp, mkdir, readdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "dist");
const extension = join(root, "extension");

await rm(dist, { force: true, recursive: true });
await mkdir(dist, { recursive: true });

for (const name of ["background", "content", "offscreen"]) {
  await build({
    bundle: true,
    entryPoints: [join(extension, "src", `${name}.js`)],
    format: "iife",
    outfile: join(dist, `${name}.js`),
    platform: "browser",
    target: "chrome121"
  });
}

await cp(join(extension, "manifest.json"), join(dist, "manifest.json"));
await cp(join(extension, "offscreen.html"), join(dist, "offscreen.html"));
await cp(join(extension, "models"), join(dist, "models"), { recursive: true });
await cp(join(extension, "icons"), join(dist, "icons"), { recursive: true });

const ortSource = join(root, "node_modules", "onnxruntime-web", "dist");
const ortDestination = join(dist, "ort");
await mkdir(ortDestination, { recursive: true });
await cp(join(ortSource, "ort.js"), join(ortDestination, "ort.js"));
for (const name of await readdir(ortSource)) {
  if (/^ort-wasm-simd-threaded(?:\.jsep)?\.(mjs|wasm)$/.test(name)) {
    await cp(join(ortSource, name), join(ortDestination, name));
  }
}
