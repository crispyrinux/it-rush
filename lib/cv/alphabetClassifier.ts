/**
 * Lightweight browser runner for the exported TensorFlow.js BISINDO alphabet MLP.
 *
 * Parses `model.json`, `group1-shard1of1.bin`, and `labels.json` directly using
 * typed Float32Arrays so inference takes <0.2ms per frame without external runtime bloat.
 */

export interface TfjsWeightSpec {
  name: string;
  shape: number[];
  dtype: string;
}

interface TfjsLayerConfig {
  class_name: string;
  config: {
    name: string;
    units?: number;
    activation?: string;
    use_bias?: boolean;
    epsilon?: number;
  };
}

interface TfjsModelJson {
  modelTopology: {
    model_config: {
      config: {
        layers: TfjsLayerConfig[];
      };
    };
  };
  weightsManifest: Array<{
    paths: string[];
    weights: TfjsWeightSpec[];
  }>;
}

interface DenseLayerRuntime {
  kind: "Dense";
  inDim: number;
  outDim: number;
  activation: string;
  kernel: Float32Array;
  bias: Float32Array | null;
}

interface BatchNormLayerRuntime {
  kind: "BatchNormalization";
  epsilon: number;
  gamma: Float32Array;
  beta: Float32Array;
  mean: Float32Array;
  variance: Float32Array;
}

interface ActivationLayerRuntime {
  kind: "Activation";
  activation: string;
}

type RuntimeLayer = DenseLayerRuntime | BatchNormLayerRuntime | ActivationLayerRuntime;

export interface AlphabetClassifier {
  labels: string[];
  inputDim: number;
  predictProbabilities: (features: Float32Array) => Float32Array;
}

function buildWeightMap(
  buffer: ArrayBuffer,
  specs: TfjsWeightSpec[],
): Map<string, Float32Array> {
  const weights = new Map<string, Float32Array>();
  let byteOffset = 0;

  for (const spec of specs) {
    const count = spec.shape.reduce((acc, dim) => acc * dim, 1);
    const byteLength = count * Float32Array.BYTES_PER_ELEMENT;
    const view = new Float32Array(buffer.slice(byteOffset, byteOffset + byteLength));
    weights.set(spec.name, view);
    byteOffset += byteLength;
  }

  return weights;
}

function buildRuntimeLayers(
  layers: TfjsLayerConfig[],
  weightMap: Map<string, Float32Array>,
): RuntimeLayer[] {
  const runtime: RuntimeLayer[] = [];

  for (const layer of layers) {
    const name = layer.config.name;
    if (layer.class_name === "Dense") {
      const kernel = weightMap.get(`${name}/kernel`);
      if (!kernel) {
        throw new Error(`Missing kernel weights for Dense layer "${name}"`);
      }
      const outDim = layer.config.units ?? 0;
      if (outDim <= 0 || kernel.length % outDim !== 0) {
        throw new Error(`Invalid Dense weight shape for layer "${name}"`);
      }
      const bias = layer.config.use_bias ? (weightMap.get(`${name}/bias`) ?? null) : null;
      runtime.push({
        kind: "Dense",
        inDim: kernel.length / outDim,
        outDim,
        activation: layer.config.activation ?? "linear",
        kernel,
        bias,
      });
    } else if (layer.class_name === "BatchNormalization") {
      const gamma = weightMap.get(`${name}/gamma`);
      const beta = weightMap.get(`${name}/beta`);
      const mean = weightMap.get(`${name}/moving_mean`);
      const variance = weightMap.get(`${name}/moving_variance`);
      if (!gamma || !beta || !mean || !variance) {
        throw new Error(`Missing BatchNormalization weights for "${name}"`);
      }
      runtime.push({
        kind: "BatchNormalization",
        epsilon: layer.config.epsilon ?? 1e-3,
        gamma,
        beta,
        mean,
        variance,
      });
    } else if (layer.class_name === "Activation") {
      runtime.push({
        kind: "Activation",
        activation: layer.config.activation ?? "linear",
      });
    } else if (layer.class_name === "ReLU") {
      runtime.push({
        kind: "Activation",
        activation: "relu",
      });
    }
  }

  return runtime;
}

function applySoftmaxInPlace(values: Float32Array): void {
  let maxVal = -Infinity;
  for (let j = 0; j < values.length; j++) {
    if (values[j] > maxVal) maxVal = values[j];
  }
  let sumExp = 0;
  for (let j = 0; j < values.length; j++) {
    const expVal = Math.exp(values[j] - maxVal);
    values[j] = expVal;
    sumExp += expVal;
  }
  const denom = sumExp > 0 ? sumExp : 1;
  for (let j = 0; j < values.length; j++) {
    values[j] /= denom;
  }
}

function applyActivation(input: Float32Array, activation: string): Float32Array {
  if (activation === "linear") {
    return input;
  }
  const out = new Float32Array(input.length);
  if (activation === "relu") {
    for (let i = 0; i < input.length; i++) {
      out[i] = input[i] > 0 ? input[i] : 0;
    }
    return out;
  }
  if (activation === "tanh") {
    for (let i = 0; i < input.length; i++) {
      out[i] = Math.tanh(input[i]);
    }
    return out;
  }
  if (activation === "sigmoid") {
    for (let i = 0; i < input.length; i++) {
      out[i] = 1 / (1 + Math.exp(-input[i]));
    }
    return out;
  }
  if (activation === "softmax") {
    out.set(input);
    applySoftmaxInPlace(out);
    return out;
  }
  out.set(input);
  return out;
}

function applyDense(input: Float32Array, layer: DenseLayerRuntime): Float32Array {
  const { inDim, outDim, kernel, bias, activation } = layer;
  if (input.length !== inDim) {
    throw new Error(`Expected input dim ${inDim}, got ${input.length}`);
  }

  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    out[j] = bias ? bias[j] : 0;
  }

  for (let i = 0; i < inDim; i++) {
    const x = input[i];
    if (x === 0) continue;
    const rowOffset = i * outDim;
    for (let j = 0; j < outDim; j++) {
      out[j] += x * kernel[rowOffset + j];
    }
  }

  if (activation === "relu") {
    for (let j = 0; j < outDim; j++) {
      if (out[j] < 0) out[j] = 0;
    }
  } else if (activation === "softmax") {
    applySoftmaxInPlace(out);
  }

  return out;
}

function applyBatchNorm(input: Float32Array, layer: BatchNormLayerRuntime): Float32Array {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const invStd = 1 / Math.sqrt(layer.variance[i] + layer.epsilon);
    out[i] = layer.gamma[i] * ((input[i] - layer.mean[i]) * invStd) + layer.beta[i];
  }
  return out;
}

let cachedPromise: Promise<AlphabetClassifier> | null = null;

const MODEL_VERSION = "v2";
const DEFAULT_SHARD_PATH = "group1-shard1of1.bin";

/**
 * Load and cache the BISINDO TF.js classifier from `/models/bisindo/`.
 */
export function loadAlphabetClassifier(basePath = "/models/bisindo"): Promise<AlphabetClassifier> {
  if (!cachedPromise || basePath !== "/models/bisindo") {
    const promise = (async (): Promise<AlphabetClassifier> => {
      const [modelRes, labelsRes, prefetchedShardRes] = await Promise.all([
        fetch(`${basePath}/model.json?v=${MODEL_VERSION}`),
        fetch(`${basePath}/labels.json?v=${MODEL_VERSION}`),
        fetch(`${basePath}/${DEFAULT_SHARD_PATH}?v=${MODEL_VERSION}`),
      ]);
      if (!modelRes.ok) {
        throw new Error(`Failed to fetch model.json (${modelRes.status})`);
      }
      if (!labelsRes.ok) {
        throw new Error(`Failed to fetch labels.json (${labelsRes.status})`);
      }

      const [modelJson, labels] = (await Promise.all([
        modelRes.json(),
        labelsRes.json(),
      ])) as [TfjsModelJson, string[]];
      const manifest = modelJson.weightsManifest[0];
      if (!manifest || manifest.paths.length === 0) {
        throw new Error("Invalid TF.js weightsManifest in model.json");
      }

      const shardPath = manifest.paths[0];
      let shardRes = prefetchedShardRes;
      if (shardPath !== DEFAULT_SHARD_PATH || !prefetchedShardRes.ok) {
        shardRes = await fetch(`${basePath}/${shardPath}?v=${MODEL_VERSION}`);
      }
      if (!shardRes.ok) {
        throw new Error(`Failed to fetch weights shard (${shardRes.status})`);
      }
      const buffer = await shardRes.arrayBuffer();
      const weightMap = buildWeightMap(buffer, manifest.weights);
      const runtimeLayers = buildRuntimeLayers(
        modelJson.modelTopology.model_config.config.layers,
        weightMap,
      );

      const firstDense = runtimeLayers.find((l): l is DenseLayerRuntime => l.kind === "Dense");
      const inputDim = firstDense ? firstDense.inDim : 86;

      return {
        labels,
        inputDim,
        predictProbabilities(features: Float32Array): Float32Array {
          let x = features;
          for (const layer of runtimeLayers) {
            if (layer.kind === "Dense") {
              x = applyDense(x, layer);
            } else if (layer.kind === "BatchNormalization") {
              x = applyBatchNorm(x, layer);
            } else {
              x = applyActivation(x, layer.activation);
            }
          }
          return x;
        },
      };
    })();

    if (basePath === "/models/bisindo") {
      cachedPromise = promise.catch((err) => {
        cachedPromise = null;
        throw err;
      });
    }
    return promise;
  }

  return cachedPromise;
}
