import { build } from "esbuild";
import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "dist-firefox");
const extension = join(root, "extension");

await rm(dist, { force: true, recursive: true });
await mkdir(dist, { recursive: true });

for (const name of ["background", "content", "options"]) {
  await build({
    bundle: true,
    entryPoints: [join(extension, "src", `${name === "background" ? "background-firefox" : name}.js`)],
    format: "iife",
    outfile: join(dist, `${name}.js`),
    platform: "browser",
    target: "firefox121"
  });
}

await cp(join(extension, "manifest-firefox.json"), join(dist, "manifest.json"));
await cp(join(extension, "background-firefox.html"), join(dist, "background.html"));
await cp(join(extension, "options.html"), join(dist, "options.html"));
await cp(join(extension, "options.css"), join(dist, "options.css"));

await mkdir(join(dist, "models"), { recursive: true });
for (const name of ["cyclopes.onnx", "cyclopes.json"]) {
  await cp(join(extension, "models", name), join(dist, "models", name));
}

await cp(join(extension, "icons"), join(dist, "icons"), { recursive: true });

const ortSource = join(root, "node_modules", "onnxruntime-web", "dist");
const ortDestination = join(dist, "ort");
await mkdir(ortDestination, { recursive: true });
await cp(join(ortSource, "ort.webgl.js"), join(ortDestination, "ort.js"));
