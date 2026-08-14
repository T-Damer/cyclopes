const SIZE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const SENTRY_SIZE = 224;
const SENTRY_MEAN = [123.675 / 255, 116.28 / 255, 103.53 / 255];
const SENTRY_STD = [58.395 / 255, 57.12 / 255, 57.375 / 255];
const FASTVIT_WEIGHT = 0.68;
const OPERATING_POINT = 0.6794350376527292;
let sessionsPromise;

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function softmaxAi(values) {
  const max = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - max));
  return exponentials[1] / (exponentials[0] + exponentials[1]);
}

export function outputToScore(output) {
  const values = Array.from(output.data);
  if (values.length === 1) return sigmoid(values[0]);
  if (values.length === 2) return softmaxAi(values);
  throw new Error(`Expected one AI logit or two class logits, received ${values.length}.`);
}

function runtime() {
  const ort = globalThis.ort;
  if (!ort) throw new Error("Local ONNX Runtime is unavailable.");
  return ort;
}

async function createSession(model) {
  const ort = runtime();
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("ort/");
  ort.env.wasm.numThreads = 1;
  const url = chrome.runtime.getURL(`models/${model}`);
  try {
    return await ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
  } catch (_webgpuError) {
    return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  }
}

function pixels(bitmap, size, mean, std, crop = false) {
  const canvas = new OffscreenCanvas(size, size);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) throw new Error("2D canvas is unavailable.");
  if (crop) {
    const edge = Math.min(bitmap.width, bitmap.height) * (SENTRY_SIZE / 256);
    context.drawImage(bitmap, (bitmap.width - edge) / 2, (bitmap.height - edge) / 2, edge, edge, 0, 0, size, size);
  } else context.drawImage(bitmap, 0, 0, size, size);
  const rgba = context.getImageData(0, 0, size, size).data;
  const data = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let index = 0; index < plane; index += 1) {
    const offset = index * 4;
    data[index] = (rgba[offset] / 255 - mean[0]) / std[0];
    data[plane + index] = (rgba[offset + 1] / 255 - mean[1]) / std[1];
    data[2 * plane + index] = (rgba[offset + 2] / 255 - mean[2]) / std[2];
  }
  return data;
}

export function calibratedBlend(fastvit, sentry) {
  const blend = Math.min(1 - 1e-6, Math.max(1e-6, FASTVIT_WEIGHT * fastvit + (1 - FASTVIT_WEIGHT) * sentry));
  const shift = Math.log(0.65 / 0.35) - Math.log(OPERATING_POINT / (1 - OPERATING_POINT));
  return sigmoid(Math.log(blend / (1 - blend)) + shift);
}

export async function inferBlob(blob) {
  const ort = runtime();
  const bitmap = await createImageBitmap(blob);
  try {
    sessionsPromise ??= Promise.all([createSession("cyclopes-fastvit.onnx"), createSession("cyclopes-sentry.onnx")]);
    const [fastvit, sentry] = await sessionsPromise;
    const fastvitInput = new ort.Tensor("float32", pixels(bitmap, SIZE, MEAN, STD), [1, 3, SIZE, SIZE]);
    const sentryInput = new ort.Tensor("float32", pixels(bitmap, SENTRY_SIZE, SENTRY_MEAN, SENTRY_STD, true), [1, 3, SENTRY_SIZE, SENTRY_SIZE]);
    const fastvitOutput = await fastvit.run({ [fastvit.inputNames[0]]: fastvitInput });
    const sentryOutput = await sentry.run({ [sentry.inputNames[0]]: sentryInput });
    const fastvitScore = sigmoid(Number(fastvitOutput[fastvit.outputNames[0]].data[0]));
    const sentryLogits = Array.from(sentryOutput[sentry.outputNames[0]].data);
    return calibratedBlend(fastvitScore, 1 - softmaxAi(sentryLogits));
  } finally {
    bitmap.close();
  }
}
