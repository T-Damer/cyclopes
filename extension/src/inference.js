const SIZE = 384;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];
let sessionPromise;

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export function outputToScore(output) {
  if (output.data.length !== 1) throw new Error(`Expected one AI logit, received ${output.data.length}.`);
  return sigmoid(Number(output.data[0]));
}

export function sourceRegion(width, height) {
  return { x: 0, y: 0, width, height };
}

function runtime() {
  if (!globalThis.ort) throw new Error("Local ONNX Runtime is unavailable.");
  return globalThis.ort;
}

async function createSession() {
  const ort = runtime();
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("ort/");
  ort.env.wasm.numThreads = 1;
  const url = chrome.runtime.getURL("models/cyclopes.onnx");
  try {
    return await ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
  } catch (_webgpuError) {
    return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  }
}

export function pixels(bitmap) {
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) throw new Error("2D canvas is unavailable.");

  context.fillStyle = "rgb(128 128 128)";
  context.fillRect(0, 0, SIZE, SIZE);
  const source = sourceRegion(bitmap.width, bitmap.height);
  const scale = Math.min(SIZE / source.width, SIZE / source.height);
  const width = Math.max(1, Math.round(source.width * scale));
  const height = Math.max(1, Math.round(source.height * scale));
  const left = (SIZE - width) / 2;
  const top = (SIZE - height) / 2;
  context.fillStyle = "white";
  context.fillRect(left, top, width, height);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, source.x, source.y, source.width, source.height, left, top, width, height);

  const rgba = context.getImageData(0, 0, SIZE, SIZE).data;
  const data = new Float32Array(3 * SIZE * SIZE);
  const plane = SIZE * SIZE;
  for (let index = 0; index < plane; index += 1) {
    const offset = index * 4;
    data[index] = (rgba[offset] / 255 - MEAN[0]) / STD[0];
    data[plane + index] = (rgba[offset + 1] / 255 - MEAN[1]) / STD[1];
    data[2 * plane + index] = (rgba[offset + 2] / 255 - MEAN[2]) / STD[2];
  }
  return data;
}

export async function inferBlob(blob) {
  const ort = runtime();
  const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  try {
    sessionPromise ??= createSession().catch((error) => {
      sessionPromise = undefined;
      throw error;
    });
    const session = await sessionPromise;
    const input = new ort.Tensor("float32", pixels(bitmap), [1, 3, SIZE, SIZE]);
    const outputs = await session.run({ [session.inputNames[0]]: input });
    return outputToScore(outputs[session.outputNames[0]]);
  } finally {
    bitmap.close();
  }
}
