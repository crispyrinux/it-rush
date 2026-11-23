/**
 * Browser inference runner for the WL-BISINDO 8-word dynamic Conformer classifier.
 *
 * Feature extraction and Conformer math match `src/bisindo/word_level.py` and
 * `public/models/bisindo/bisindo_wl_model.tflite` bit-for-bit within float32 tolerance.
 */

import { WORD_ITEMS, WORD_MODEL_VERSION } from "@/lib/bisindo";
import type { NormalizedLandmark2D } from "./normalizeLandmarks";

export const WL_SEQUENCE_LEN = 30;
export const WL_RAW_FEATURE_DIM = 100;
export const WL_TEMPORAL_FEATURE_DIM = 300;
const NUM_POSE_ANCHORS = 7;

export interface WordPredictionCandidate {
  word: string;
  label: string;
  confidence: number;
}

interface TensorSpec {
  name: string;
  shape: number[];
  dtype: "float32" | "int32";
  byteOffset: number;
  byteLength: number;
}

interface WlModelManifest {
  format: string;
  tensors: Record<string, TensorSpec>;
}

interface WlLabelsJson {
  labels: string[];
}

export interface WlVariantSpec {
  id: string;
  label: string;
  basePath: string;
  format: "legacy-tflite-conformer" | "e2e-conformer";
  featureMode: "standard300" | "geo350";
}

export interface WlVariantsRegistry {
  defaultVariant: string;
  variants: WlVariantSpec[];
}

export interface WordClassifier {
  labels: string[];
  predictFromFeatures: (frames: ReadonlyArray<Float32Array>) => WordPredictionCandidate[];
  predictProbabilities: (inputSeq30x300: Float32Array) => Float32Array;
}

const WORD_DISPLAY_NAMES: Record<string, string> = Object.fromEntries(
  WORD_ITEMS.map((item) => [item.id, item.title]),
);

function toTitleCase(text: string): string {
  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

export function formatWordLabel(word: string): string {
  return WORD_DISPLAY_NAMES[word] ?? toTitleCase(word.replaceAll("_", " "));
}

/**
 * Determine anatomical right vs left hand from palm triangle + thumb/pinky geometry.
 */
export function anatomicalIsRight(landmarks: ReadonlyArray<NormalizedLandmark2D>): boolean {
  const w = landmarks[0];
  const i5 = landmarks[5];
  const p17 = landmarks[17];
  const t4 = landmarks[4];
  const p20 = landmarks[20];
  const crossZ = (i5.x - w.x) * (p17.y - w.y) - (i5.y - w.y) * (p17.x - w.x);
  const thumbVsPinky = t4.x - p20.x;
  return (crossZ >= 0 && thumbVsPinky <= 0) || (crossZ < 0 && thumbVsPinky > 0);
}

/**
 * Build a 100-float nose-centered frame feature vector (NaN for missing body parts).
 */
export function extractWlFrameFeatures(
  hands: ReadonlyArray<ReadonlyArray<NormalizedLandmark2D>>,
  poseAnchors: ReadonlyArray<NormalizedLandmark2D> | null,
): Float32Array {
  const out = new Float32Array(WL_RAW_FEATURE_DIM);
  out.fill(Number.NaN);

  let noseX = 0.5;
  let noseY = 0.5;

  if (poseAnchors && poseAnchors.length >= NUM_POSE_ANCHORS) {
    noseX = poseAnchors[0].x;
    noseY = poseAnchors[0].y;
    for (let i = 0; i < NUM_POSE_ANCHORS; i++) {
      out[84 + i * 2] = poseAnchors[i].x - noseX;
      out[84 + i * 2 + 1] = poseAnchors[i].y - noseY;
    }
  }

  let hasRight = false;
  let hasLeft = false;
  for (const hand of hands) {
    if (hand.length < 21) continue;
    const isRight = anatomicalIsRight(hand);
    const base = isRight ? 0 : 42;
    if (isRight) {
      hasRight = true;
    } else {
      hasLeft = true;
    }
    for (let j = 0; j < 21; j++) {
      out[base + j * 2] = hand[j].x - noseX;
      out[base + j * 2 + 1] = hand[j].y - noseY;
    }
  }

  out[98] = hasRight ? 1.0 : Number.NaN;
  out[99] = hasLeft ? 1.0 : Number.NaN;
  return out;
}

/**
 * Uniformly resample a variable-length frame sequence into `(30, 100)`.
 */
export function resampleSequence(
  frames: ReadonlyArray<Float32Array>,
  targetLen = WL_SEQUENCE_LEN,
): Float32Array {
  const resampled = new Float32Array(targetLen * WL_RAW_FEATURE_DIM);
  if (frames.length === 0) return resampled;

  for (let t = 0; t < targetLen; t++) {
    const srcIdx =
      frames.length === 1
        ? 0
        : t === targetLen - 1
          ? frames.length - 1
          : Math.floor((t * (frames.length - 1)) / (targetLen - 1));
    resampled.set(frames[srcIdx], t * WL_RAW_FEATURE_DIM);
  }
  return resampled;
}

/**
 * Normalize sequence globally over valid entries and replace NaNs with zero.
 */
export function normalizeSequenceStd(seq: Float32Array): Float32Array {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < seq.length; i++) {
    const v = seq[i];
    if (Number.isFinite(v)) {
      sum += v;
      count++;
    }
  }
  const out = new Float32Array(seq.length);
  if (count === 0) return out;

  const mean = sum / count;
  let sqSum = 0;
  for (let i = 0; i < seq.length; i++) {
    const v = seq[i];
    if (Number.isFinite(v)) {
      const d = v - mean;
      sqSum += d * d;
    }
  }
  const stdRaw = Math.sqrt(sqSum / count);
  const std = !Number.isFinite(stdRaw) || stdRaw < 1e-6 ? 1.0 : stdRaw;

  for (let i = 0; i < seq.length; i++) {
    const v = seq[i];
    out[i] = Number.isFinite(v) ? (v - mean) / std : 0;
  }
  return out;
}

/**
 * Convert raw 100-dim frame vectors into flattened `(30, 300)` `[pos, vel, acc]` model input.
 */
export function prepareWlModelInput(
  frames: ReadonlyArray<Float32Array>,
  sequenceLen = WL_SEQUENCE_LEN,
): Float32Array {
  const norm = normalizeSequenceStd(resampleSequence(frames, sequenceLen));
  const out = new Float32Array(sequenceLen * WL_TEMPORAL_FEATURE_DIM);

  for (let t = 0; t < sequenceLen; t++) {
    const srcOffset = t * WL_RAW_FEATURE_DIM;
    const dstOffset = t * WL_TEMPORAL_FEATURE_DIM;
    for (let f = 0; f < WL_RAW_FEATURE_DIM; f++) {
      const pos = norm[srcOffset + f];
      out[dstOffset + f] = pos;
      if (t >= 1) {
        out[dstOffset + WL_RAW_FEATURE_DIM + f] = pos - norm[(t - 1) * WL_RAW_FEATURE_DIM + f];
      }
      if (t >= 2) {
        out[dstOffset + WL_RAW_FEATURE_DIM * 2 + f] = pos - norm[(t - 2) * WL_RAW_FEATURE_DIM + f];
      }
    }
  }

  return out;
}

/**
 * Append 25 scale-robust geometric features (distances, openness, limb vectors) + velocities -> `(30, 350)`.
 */
export function extractGeometricFeatures(
  seq300: Float32Array,
  sequenceLen = WL_SEQUENCE_LEN,
): Float32Array {
  const GEO_DIM = 25;
  const OUT_DIM = WL_TEMPORAL_FEATURE_DIM + GEO_DIM * 2; // 350
  const geo = new Float32Array(sequenceLen * GEO_DIM);

  const dist2d = (
    arr: Float32Array,
    aOff: number,
    bOff: number,
  ): number => {
    const dx = arr[aOff] - arr[bOff];
    const dy = arr[aOff + 1] - arr[bOff + 1];
    return Math.hypot(dx, dy);
  };

  const writeUnit = (
    arr: Float32Array,
    aOff: number,
    bOff: number,
    scale: number,
    out: Float32Array,
    outIdx: number,
  ): void => {
    const vx = arr[bOff] - arr[aOff];
    const vy = arr[bOff + 1] - arr[aOff + 1];
    const norm = Math.max(Math.hypot(vx, vy), 1e-4);
    out[outIdx] = (vx / norm) * scale;
    out[outIdx + 1] = (vy / norm) * scale;
  };

  for (let t = 0; t < sequenceLen; t++) {
    const s = t * WL_TEMPORAL_FEATURE_DIM;
    const g = t * GEO_DIM;
    const hasR = seq300[s + 98];
    const hasL = seq300[s + 99];
    const hasBoth = hasR * hasL;

    const rWrist = s + 0 * 2;
    const rThumb = s + 4 * 2;
    const rIdx = s + 8 * 2;
    const rPalm = s + 9 * 2;
    const rMid = s + 12 * 2;
    const rPinky = s + 20 * 2;

    const lWrist = s + 42 + 0 * 2;
    const lThumb = s + 42 + 4 * 2;
    const lIdx = s + 42 + 8 * 2;
    const lPalm = s + 42 + 9 * 2;
    const lMid = s + 42 + 12 * 2;
    const lPinky = s + 42 + 20 * 2;

    const nose = s + 84 + 0 * 2;
    const lSh = s + 84 + 1 * 2;
    const rSh = s + 84 + 2 * 2;
    const lEl = s + 84 + 5 * 2;
    const rEl = s + 84 + 6 * 2;

    geo[g + 0] = dist2d(seq300, rWrist, nose) * hasR;
    geo[g + 1] = dist2d(seq300, lWrist, nose) * hasL;
    geo[g + 2] = dist2d(seq300, rIdx, nose) * hasR;
    geo[g + 3] = dist2d(seq300, lIdx, nose) * hasL;
    geo[g + 4] = dist2d(seq300, rWrist, rSh) * hasR;
    geo[g + 5] = dist2d(seq300, lWrist, lSh) * hasL;
    geo[g + 6] = dist2d(seq300, rWrist, lSh) * hasR;
    geo[g + 7] = dist2d(seq300, lWrist, rSh) * hasL;

    geo[g + 8] = dist2d(seq300, rWrist, lWrist) * hasBoth;
    geo[g + 9] = dist2d(seq300, rIdx, lIdx) * hasBoth;
    geo[g + 10] = dist2d(seq300, rPalm, lPalm) * hasBoth;

    geo[g + 11] = dist2d(seq300, rThumb, rIdx) * hasR;
    geo[g + 12] = dist2d(seq300, rThumb, rPinky) * hasR;
    geo[g + 13] = dist2d(seq300, rWrist, rMid) * hasR;
    geo[g + 14] = dist2d(seq300, lThumb, lIdx) * hasL;
    geo[g + 15] = dist2d(seq300, lThumb, lPinky) * hasL;
    geo[g + 16] = dist2d(seq300, lWrist, lMid) * hasL;

    writeUnit(seq300, rEl, rWrist, hasR, geo, g + 17);
    writeUnit(seq300, lEl, lWrist, hasL, geo, g + 19);
    writeUnit(seq300, rWrist, rPalm, hasR, geo, g + 21);
    writeUnit(seq300, lWrist, lPalm, hasL, geo, g + 23);
  }

  const out = new Float32Array(sequenceLen * OUT_DIM);
  for (let t = 0; t < sequenceLen; t++) {
    const src300 = t * WL_TEMPORAL_FEATURE_DIM;
    const dst = t * OUT_DIM;
    out.set(seq300.subarray(src300, src300 + WL_TEMPORAL_FEATURE_DIM), dst);
    const gOff = t * GEO_DIM;
    for (let f = 0; f < GEO_DIM; f++) {
      const val = geo[gOff + f];
      out[dst + WL_TEMPORAL_FEATURE_DIM + f] = val;
      if (t >= 1) {
        out[dst + WL_TEMPORAL_FEATURE_DIM + GEO_DIM + f] = val - geo[(t - 1) * GEO_DIM + f];
      }
    }
  }
  return out;
}

function swishInPlace(arr: Float32Array): Float32Array {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    arr[i] = v / (1 + Math.exp(-v));
  }
  return arr;
}

function linearProjection(
  input: Float32Array,
  rows: number,
  inDim: number,
  weight: Float32Array,
  outDim: number,
  bias: Float32Array | null,
): Float32Array {
  const out = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r++) {
    const xBase = r * inDim;
    const yBase = r * outDim;
    for (let o = 0; o < outDim; o++) {
      let acc = bias ? bias[o] : 0;
      const wBase = o * inDim;
      for (let i = 0; i < inDim; i++) {
        acc += input[xBase + i] * weight[wBase + i];
      }
      out[yBase + o] = acc;
    }
  }
  return out;
}

function runConvBlock(
  xInProj: Float32Array,
  xRes: Float32Array,
  getF32: (idx: number) => Float32Array,
  ids: {
    fcInW: number;
    fcInB: number;
    dwW: number;
    dwB: number;
    ecaW: number;
    ecaB: number;
    fcOutW: number;
    fcOutB: number;
  },
): Float32Array {
  const T = WL_SEQUENCE_LEN;
  const C = 128;
  const D = 64;
  const h = swishInPlace(
    linearProjection(xInProj, T, D, getF32(ids.fcInW), C, getF32(ids.fcInB)),
  );

  // Depthwise causal conv1d: kernel size 11 stacked left-padded by 10.
  // TFLite depthwise weight shape is [1, 1, 11, 128].
  const dwK = getF32(ids.dwW);
  const dwB = getF32(ids.dwB);
  const hDw = new Float32Array(T * C);
  const channelMeans = new Float32Array(C);

  for (let t = 0; t < T; t++) {
    for (let c = 0; c < C; c++) {
      let acc = dwB[c];
      for (let k = 0; k < 11; k++) {
        const srcT = t + k - 10;
        if (srcT >= 0) {
          acc += h[srcT * C + c] * dwK[k * C + c];
        }
      }
      hDw[t * C + c] = acc;
      channelMeans[c] += acc;
    }
  }
  for (let c = 0; c < C; c++) {
    channelMeans[c] /= T;
  }

  // Efficient Channel Attention (1D conv across 128 channels, kernel size 5, pad 2).
  const ecaK = getF32(ids.ecaW);
  const ecaBias = getF32(ids.ecaB)[0];
  const ecaScale = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let acc = ecaBias;
    for (let k = 0; k < 5; k++) {
      const idx = c + k - 2;
      if (idx >= 0 && idx < C) {
        acc += channelMeans[idx] * ecaK[k];
      }
    }
    ecaScale[c] = 1 / (1 + Math.exp(-acc));
  }

  for (let t = 0; t < T; t++) {
    for (let c = 0; c < C; c++) {
      hDw[t * C + c] *= ecaScale[c];
    }
  }

  const proj = linearProjection(hDw, T, C, getF32(ids.fcOutW), D, getF32(ids.fcOutB));
  for (let i = 0; i < proj.length; i++) {
    proj[i] += xRes[i];
  }
  return proj;
}

function runMhsaBlock(
  x: Float32Array,
  getF32: (idx: number) => Float32Array,
  ids: {
    scale: number;
    shift: number;
    qW: number;
    qB: number;
    kW: number;
    kB: number;
    vW: number;
    vB: number;
    oW: number;
    oB: number;
    qScale: number;
  },
): Float32Array {
  const T = WL_SEQUENCE_LEN;
  const D = 64;
  const H = 2;
  const HD = 32;
  const bnScale = getF32(ids.scale);
  const bnShift = getF32(ids.shift);
  const scaleScalar = getF32(ids.qScale)[0];

  const norm = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      norm[t * D + d] = x[t * D + d] * bnScale[d] + bnShift[d];
    }
  }

  const Q = linearProjection(norm, T, D, getF32(ids.qW), D, getF32(ids.qB));
  const K = linearProjection(norm, T, D, getF32(ids.kW), D, getF32(ids.kB));
  const V = linearProjection(norm, T, D, getF32(ids.vW), D, getF32(ids.vB));

  for (let i = 0; i < K.length; i++) {
    K[i] *= scaleScalar;
  }

  const ctx = new Float32Array(T * D);
  const rawScores = new Float32Array(T * T);

  // Per head attention: note TFLite op 49 transposes [0, 1, 3, 2] before softmax.
  for (let h = 0; h < H; h++) {
    const hOffset = h * HD;
    for (let r = 0; r < T; r++) {
      for (let c = 0; c < T; c++) {
        let dot = 0;
        for (let d = 0; d < HD; d++) {
          dot += Q[r * D + hOffset + d] * K[c * D + hOffset + d];
        }
        // Store transposed `[c, r]` matching TFLite op 49 `[0, 1, 3, 2]`
        rawScores[c * T + r] = dot;
      }
    }

    for (let row = 0; row < T; row++) {
      let maxVal = -Infinity;
      const base = row * T;
      for (let col = 0; col < T; col++) {
        if (rawScores[base + col] > maxVal) maxVal = rawScores[base + col];
      }
      let sumExp = 0;
      for (let col = 0; col < T; col++) {
        const e = Math.exp(rawScores[base + col] - maxVal);
        rawScores[base + col] = e;
        sumExp += e;
      }
      for (let col = 0; col < T; col++) {
        rawScores[base + col] /= sumExp;
      }
    }

    for (let row = 0; row < T; row++) {
      for (let d = 0; d < HD; d++) {
        let acc = 0;
        for (let col = 0; col < T; col++) {
          acc += rawScores[row * T + col] * V[col * D + hOffset + d];
        }
        ctx[row * D + hOffset + d] = acc;
      }
    }
  }

  const proj = linearProjection(ctx, T, D, getF32(ids.oW), D, getF32(ids.oB));
  for (let i = 0; i < proj.length; i++) {
    proj[i] += x[i];
  }
  return proj;
}

function runFfnBlock(
  x: Float32Array,
  getF32: (idx: number) => Float32Array,
  ids: { w1: number; b1: number; w2: number; b2: number },
): Float32Array {
  const T = WL_SEQUENCE_LEN;
  const h = swishInPlace(linearProjection(x, T, 64, getF32(ids.w1), 128, getF32(ids.b1)));
  const out = linearProjection(h, T, 128, getF32(ids.w2), 64, getF32(ids.b2));
  for (let i = 0; i < out.length; i++) {
    out[i] += x[i];
  }
  return out;
}

interface E2eTensorSpec {
  shape: number[];
  dtype: string;
  byteOffset: number;
  byteLength: number;
}

interface E2eDenseRef {
  w: string;
  b?: string;
}

interface E2eLnRef {
  gamma: string;
  beta: string;
}

interface E2eBnRef {
  scale: string;
  shift: string;
}

interface E2eBlockSpec {
  ln1: E2eLnRef;
  convIn: E2eDenseRef;
  dw: { w: string; b: string };
  convBn: E2eBnRef;
  convOut: E2eDenseRef;
  lnAtt: E2eLnRef;
  mha: {
    qW: string;
    qB: string;
    kW: string;
    kB: string;
    vW: string;
    vB: string;
    oW: string;
    oB: string;
  };
  lnFf: E2eLnRef;
  ff1: E2eDenseRef;
  ff2: E2eDenseRef;
}

interface E2eManifest {
  format: string;
  featureMode: "standard300" | "geo350";
  seqLen: number;
  inDim: number;
  dim: number;
  numHeads: number;
  numClasses: number;
  lnEps: number;
  stem: { proj: E2eDenseRef; bn: E2eBnRef };
  blocks: E2eBlockSpec[];
  head: { lnOut: E2eLnRef; dense: E2eDenseRef };
  tensors: Record<string, E2eTensorSpec>;
}

function layerNorm(
  x: Float32Array,
  rows: number,
  dim: number,
  gamma: Float32Array,
  beta: Float32Array,
  eps: number,
): Float32Array {
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r++) {
    const base = r * dim;
    let sum = 0;
    for (let d = 0; d < dim; d++) sum += x[base + d];
    const mean = sum / dim;
    let sq = 0;
    for (let d = 0; d < dim; d++) {
      const diff = x[base + d] - mean;
      sq += diff * diff;
    }
    const invStd = 1 / Math.sqrt(sq / dim + eps);
    for (let d = 0; d < dim; d++) {
      out[base + d] = ((x[base + d] - mean) * invStd) * gamma[d] + beta[d];
    }
  }
  return out;
}

function depthwiseConv1dSame(
  x: Float32Array,
  rows: number,
  channels: number,
  kernel: Float32Array,
  bias: Float32Array,
  kernelSize = 7,
): Float32Array {
  const half = Math.floor(kernelSize / 2);
  const out = new Float32Array(rows * channels);
  for (let t = 0; t < rows; t++) {
    for (let c = 0; c < channels; c++) {
      let acc = bias[c];
      for (let k = 0; k < kernelSize; k++) {
        const srcT = t + k - half;
        if (srcT >= 0 && srcT < rows) {
          acc += x[srcT * channels + c] * kernel[k * channels + c];
        }
      }
      out[t * channels + c] = acc;
    }
  }
  return out;
}

function runE2eConformerForward(
  manifest: E2eManifest,
  getT: (name: string) => Float32Array,
  inputSeq: Float32Array,
): Float32Array {
  const T = manifest.seqLen;
  const D = manifest.dim;
  const H = manifest.numHeads;
  const HD = D / H;
  const eps = manifest.lnEps || 1e-3;

  const dense = (inp: Float32Array, rows: number, inDim: number, outDim: number, ref: E2eDenseRef) =>
    linearProjection(inp, rows, inDim, getT(ref.w), outDim, ref.b ? getT(ref.b) : null);

  const x = dense(inputSeq, T, manifest.inDim, D, manifest.stem.proj);
  const sScale = getT(manifest.stem.bn.scale);
  const sShift = getT(manifest.stem.bn.shift);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      const v = x[t * D + d] * sScale[d] + sShift[d];
      x[t * D + d] = v / (1 + Math.exp(-v));
    }
  }

  for (const blk of manifest.blocks) {
    const hLn = layerNorm(x, T, D, getT(blk.ln1.gamma), getT(blk.ln1.beta), eps);
    const hIn = swishInPlace(dense(hLn, T, D, D * 2, blk.convIn));
    const hDw = depthwiseConv1dSame(hIn, T, D * 2, getT(blk.dw.w), getT(blk.dw.b), 7);
    const cScale = getT(blk.convBn.scale);
    const cShift = getT(blk.convBn.shift);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < D * 2; c++) {
        const v = hDw[t * (D * 2) + c] * cScale[c] + cShift[c];
        hDw[t * (D * 2) + c] = v / (1 + Math.exp(-v));
      }
    }
    const hConvOut = dense(hDw, T, D * 2, D, blk.convOut);
    for (let i = 0; i < x.length; i++) x[i] += hConvOut[i];

    const hAttLn = layerNorm(x, T, D, getT(blk.lnAtt.gamma), getT(blk.lnAtt.beta), eps);
    const Q = linearProjection(hAttLn, T, D, getT(blk.mha.qW), D, getT(blk.mha.qB));
    const K = linearProjection(hAttLn, T, D, getT(blk.mha.kW), D, getT(blk.mha.kB));
    const V = linearProjection(hAttLn, T, D, getT(blk.mha.vW), D, getT(blk.mha.vB));

    const attScale = 1 / Math.sqrt(HD);
    const ctx = new Float32Array(T * D);
    const scores = new Float32Array(T * T);
    for (let h = 0; h < H; h++) {
      const hOff = h * HD;
      for (let r = 0; r < T; r++) {
        let maxS = -Infinity;
        for (let c = 0; c < T; c++) {
          let dot = 0;
          for (let d = 0; d < HD; d++) {
            dot += Q[r * D + hOff + d] * K[c * D + hOff + d];
          }
          const s = dot * attScale;
          scores[r * T + c] = s;
          if (s > maxS) maxS = s;
        }
        let sumE = 0;
        for (let c = 0; c < T; c++) {
          const e = Math.exp(scores[r * T + c] - maxS);
          scores[r * T + c] = e;
          sumE += e;
        }
        for (let c = 0; c < T; c++) scores[r * T + c] /= sumE;
      }
      for (let r = 0; r < T; r++) {
        for (let d = 0; d < HD; d++) {
          let acc = 0;
          for (let c = 0; c < T; c++) acc += scores[r * T + c] * V[c * D + hOff + d];
          ctx[r * D + hOff + d] = acc;
        }
      }
    }
    const hAttOut = linearProjection(ctx, T, D, getT(blk.mha.oW), D, getT(blk.mha.oB));
    for (let i = 0; i < x.length; i++) x[i] += hAttOut[i];

    const hFfLn = layerNorm(x, T, D, getT(blk.lnFf.gamma), getT(blk.lnFf.beta), eps);
    const hFf1 = swishInPlace(dense(hFfLn, T, D, D * 2, blk.ff1));
    const hFf2 = dense(hFf1, T, D * 2, D, blk.ff2);
    for (let i = 0; i < x.length; i++) x[i] += hFf2[i];
  }

  const xNorm = layerNorm(x, T, D, getT(manifest.head.lnOut.gamma), getT(manifest.head.lnOut.beta), eps);
  const emb = new Float32Array(D);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) emb[d] += xNorm[t * D + d];
  }
  for (let d = 0; d < D; d++) emb[d] /= T;

  const logits = dense(emb, 1, D, manifest.numClasses, manifest.head.dense);
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
  let sumExp = 0;
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    probs[i] = e;
    sumExp += e;
  }
  const denom = sumExp > 0 ? sumExp : 1;
  for (let i = 0; i < probs.length; i++) probs[i] /= denom;
  return probs;
}

const cachedWordPromises = new Map<string, Promise<WordClassifier>>();
let cachedVariantsPromise: Promise<WlVariantsRegistry> | null = null;

const WL_MODEL_VERSION = encodeURIComponent(WORD_MODEL_VERSION);

export const DEFAULT_WL_VARIANT: WlVariantSpec = {
  id: "frozen-head",
  label: "Model A: Frozen Conformer + Head (110 Klip)",
  basePath: "/models/bisindo",
  format: "legacy-tflite-conformer",
  featureMode: "standard300",
};

/**
 * Load available A/B/C model variants for live testing in `/bisindo`.
 */
export function loadWordVariants(): Promise<WlVariantsRegistry> {
  if (!cachedVariantsPromise) {
    cachedVariantsPromise = fetch(`/models/bisindo/wl_variants.json?v=${WL_MODEL_VERSION}`)
      .then(async (res) => {
        if (!res.ok) {
          return { defaultVariant: DEFAULT_WL_VARIANT.id, variants: [DEFAULT_WL_VARIANT] };
        }
        const data = (await res.json()) as WlVariantsRegistry;
        if (!Array.isArray(data.variants) || data.variants.length === 0) {
          return { defaultVariant: DEFAULT_WL_VARIANT.id, variants: [DEFAULT_WL_VARIANT] };
        }
        return data;
      })
      .catch(() => ({ defaultVariant: DEFAULT_WL_VARIANT.id, variants: [DEFAULT_WL_VARIANT] }));
  }
  return cachedVariantsPromise;
}

/**
 * Load and cache a WL-BISINDO classifier (supports both legacy TFLite-exported and E2E Conformer variants).
 */
export function loadWordClassifier(
  basePath = "/models/bisindo",
  options?: {
    format?: "legacy-tflite-conformer" | "e2e-conformer";
    featureMode?: "standard300" | "geo350";
  },
): Promise<WordClassifier> {
  const format = options?.format ?? "legacy-tflite-conformer";
  const featureMode = options?.featureMode ?? "standard300";
  const cacheKey = `${basePath}|${format}|${featureMode}`;

  const existing = cachedWordPromises.get(cacheKey);
  if (existing) return existing;

  const promise = (async (): Promise<WordClassifier> => {
    if (format === "e2e-conformer") {
      const [manifestRes, labelsRes, weightsRes] = await Promise.all([
        fetch(`${basePath}/wl_e2e_model.json?v=${WL_MODEL_VERSION}`),
        fetch(`${basePath}/bisindo_wl_labels.json?v=${WL_MODEL_VERSION}`),
        fetch(`${basePath}/wl_e2e_weights.bin?v=${WL_MODEL_VERSION}`),
      ]);
      if (!manifestRes.ok) throw new Error(`Failed to fetch wl_e2e_model.json (${manifestRes.status})`);
      if (!labelsRes.ok) throw new Error(`Failed to fetch bisindo_wl_labels.json (${labelsRes.status})`);
      if (!weightsRes.ok) throw new Error(`Failed to fetch wl_e2e_weights.bin (${weightsRes.status})`);

      const manifest = (await manifestRes.json()) as E2eManifest;
      const labelsJson = (await labelsRes.json()) as WlLabelsJson;
      const buffer = await weightsRes.arrayBuffer();

      const tensorMap = new Map<string, Float32Array>();
      for (const [name, spec] of Object.entries(manifest.tensors)) {
        const slice = buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength);
        tensorMap.set(name, new Float32Array(slice));
      }
      const getT = (name: string): Float32Array => {
        const arr = tensorMap.get(name);
        if (!arr) throw new Error(`Missing E2E tensor ${name}`);
        return arr;
      };
      const labels = labelsJson.labels;
      const predictProbabilities = (inputSeq30x300: Float32Array): Float32Array => {
        const modelInput =
          manifest.featureMode === "geo350" || featureMode === "geo350"
            ? extractGeometricFeatures(inputSeq30x300, WL_SEQUENCE_LEN)
            : inputSeq30x300;
        return runE2eConformerForward(manifest, getT, modelInput);
      };
      return {
        labels,
        predictFromFeatures(frames: ReadonlyArray<Float32Array>): WordPredictionCandidate[] {
          const input300 = prepareWlModelInput(frames, WL_SEQUENCE_LEN);
          const probs = predictProbabilities(input300);
          return labels
            .map((word, idx) => ({
              word,
              label: formatWordLabel(word),
              confidence: probs[idx] ?? 0,
            }))
            .sort((a, b) => b.confidence - a.confidence);
        },
        predictProbabilities,
      };
    }

    const [manifestRes, labelsRes, weightsRes] = await Promise.all([
      fetch(`${basePath}/wl_model.json?v=${WL_MODEL_VERSION}`),
      fetch(`${basePath}/bisindo_wl_labels.json?v=${WL_MODEL_VERSION}`),
      fetch(`${basePath}/wl_weights.bin?v=${WL_MODEL_VERSION}`),
    ]);

    if (!manifestRes.ok) throw new Error(`Failed to fetch wl_model.json (${manifestRes.status})`);
    if (!labelsRes.ok) throw new Error(`Failed to fetch bisindo_wl_labels.json (${labelsRes.status})`);
    if (!weightsRes.ok) throw new Error(`Failed to fetch wl_weights.bin (${weightsRes.status})`);

    const manifest = (await manifestRes.json()) as WlModelManifest;
    const labelsJson = (await labelsRes.json()) as WlLabelsJson;
    const buffer = await weightsRes.arrayBuffer();

    const f32Map = new Map<number, Float32Array>();
    for (const [idStr, spec] of Object.entries(manifest.tensors)) {
      if (spec.dtype === "float32") {
        const slice = buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength);
        f32Map.set(Number(idStr), new Float32Array(slice));
      }
    }

    const getF32 = (idx: number): Float32Array => {
      const arr = f32Map.get(idx);
      if (!arr) throw new Error(`Missing WL tensor ${idx}`);
      return arr;
    };

    const labels = labelsJson.labels;

    const predictProbabilities = (inputSeq30x300: Float32Array): Float32Array => {
      const x0 = linearProjection(inputSeq30x300, WL_SEQUENCE_LEN, 300, getF32(78), 64, null);
      const stemScale = getF32(76);
      const stemShift = getF32(77);
      const xInit = new Float32Array(x0.length);
      for (let t = 0; t < WL_SEQUENCE_LEN; t++) {
        for (let d = 0; d < 64; d++) {
          xInit[t * 64 + d] = x0[t * 64 + d] * stemScale[d] + stemShift[d];
        }
      }

      // Conformer Block 1
      let x = runConvBlock(x0, xInit, getF32, {
        fcInW: 14, fcInB: 15, dwW: 8, dwB: 4, ecaW: 38, ecaB: 39, fcOutW: 33, fcOutB: 57,
      });
      x = runConvBlock(x, x, getF32, {
        fcInW: 32, fcInB: 56, dwW: 7, dwB: 3, ecaW: 36, ecaB: 39, fcOutW: 31, fcOutB: 55,
      });
      x = runMhsaBlock(x, getF32, {
        scale: 72, shift: 73, qW: 30, qB: 43, kW: 29, kB: 42, vW: 28, vB: 41, oW: 27, oB: 44, qScale: 66,
      });
      x = runFfnBlock(x, getF32, { w1: 11, b1: 12, w2: 26, b2: 54 });

      // Conformer Block 2
      x = runConvBlock(x, x, getF32, {
        fcInW: 25, fcInB: 53, dwW: 6, dwB: 2, ecaW: 35, ecaB: 39, fcOutW: 24, fcOutB: 52,
      });
      x = runConvBlock(x, x, getF32, {
        fcInW: 23, fcInB: 51, dwW: 5, dwB: 1, ecaW: 34, ecaB: 39, fcOutW: 22, fcOutB: 50,
      });
      x = runMhsaBlock(x, getF32, {
        scale: 74, shift: 75, qW: 21, qB: 47, kW: 20, kB: 46, vW: 19, vB: 45, oW: 18, oB: 48, qScale: 66,
      });
      x = runFfnBlock(x, getF32, { w1: 9, b1: 10, w2: 17, b2: 58 });

      // Global average pooling + classifier head
      const projHead = swishInPlace(
        linearProjection(x, WL_SEQUENCE_LEN, 64, getF32(16), 128, getF32(49)),
      );
      const emb = new Float32Array(128);
      for (let t = 0; t < WL_SEQUENCE_LEN; t++) {
        for (let c = 0; c < 128; c++) {
          emb[c] += projHead[t * 128 + c];
        }
      }
      for (let c = 0; c < 128; c++) {
        emb[c] /= WL_SEQUENCE_LEN;
      }

      const logits = linearProjection(emb, 1, 128, getF32(40), labels.length, getF32(59));
      let maxLogit = -Infinity;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
      }
      let sumExp = 0;
      const probs = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) {
        const e = Math.exp(logits[i] - maxLogit);
        probs[i] = e;
        sumExp += e;
      }
      const denom = sumExp > 0 ? sumExp : 1;
      for (let i = 0; i < probs.length; i++) {
        probs[i] /= denom;
      }
      return probs;
    };

    return {
      labels,
      predictFromFeatures(frames: ReadonlyArray<Float32Array>): WordPredictionCandidate[] {
        const input = prepareWlModelInput(frames, WL_SEQUENCE_LEN);
        const probs = predictProbabilities(input);
        return labels
          .map((word, idx) => ({
            word,
            label: formatWordLabel(word),
            confidence: probs[idx] ?? 0,
          }))
          .sort((a, b) => b.confidence - a.confidence);
      },
      predictProbabilities,
    };
  })().catch((err) => {
    cachedWordPromises.delete(cacheKey);
    throw err;
  });

  cachedWordPromises.set(cacheKey, promise);
  return promise;
}
