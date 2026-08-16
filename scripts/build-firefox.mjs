import { build } from "esbuild";
import { access, cp, mkdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "dist-firefox");
const extension = join(root, "extension");
const isCI = process.env.CI === "true";

async function copyModelArtifact(source, target) {
  try {
    await access(source);
    await cp(source, target);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") {
      if (isCI) {
        console.warn(`Warning: missing model artifact ${source}. Build will continue without it for CI packaging.`);
        return false;
      }
      throw new Error(`Missing required model artifact: ${source}. Add extension/models/cyclopes.onnx before building locally.`);
    }
    throw error;
  }
}

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
  const source = join(extension, "models", name);
  const copied = await copyModelArtifact(source, join(dist, "models", name));
  if (!copied && name === "cyclopes.onnx") {
    console.warn("Local inference will be unavailable until cyclopes.onnx is added.");
  }
}

await cp(join(extension, "icons"), join(dist, "icons"), { recursive: true });

const ortSource = join(root, "node_modules", "onnxruntime-web", "dist");
const ortDestination = join(dist, "ort");
await mkdir(ortDestination, { recursive: true });
await cp(join(ortSource, "ort.webgl.js"), join(ortDestination, "ort.js"));
