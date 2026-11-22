/**
 * BISINDO 86-dim Bohacek hand landmark normalization & hold-time validator.
 *
 * Mathematically identical to `src/bisindo/features.py`:
 *   [0..41]   Leftmost hand on screen: 21 x (x, y) Bohacek-normalized to [0, 1]
 *   [42..83]  Rightmost hand on screen: 21 x (x, y) Bohacek-normalized, or 0.0
 *   [84]      dx = mean_x(right) - mean_x(left) (>= 0 for two hands, 0.0 for single)
 *   [85]      dy = mean_y(right) - mean_y(left) (0.0 for single)
 */

export interface NormalizedLandmark2D {
  x: number;
  y: number;
  z?: number;
}

export const NUM_LANDMARKS = 21;
export const NUM_HAND_FEATURES = NUM_LANDMARKS * 2; // 42
export const FEATURE_DIM = NUM_HAND_FEATURES * 2 + 2; // 86
export const INTER_HAND_KEY_INDICES: ReadonlyArray<number> = [0, 4, 8, 12, 16, 20, 9];
export const FEATURE_DIM_V2 =
  NUM_HAND_FEATURES * 4 + INTER_HAND_KEY_INDICES.length * INTER_HAND_KEY_INDICES.length + 5; // 222
const EPS = 1e-6;

/**
 * Apply Bohacek bounding-box normalization to 21 hand landmarks.
 * Returns a 42-element Float32Array with values in [0, 1].
 */
export function normalizeHandBohacek(landmarks: ReadonlyArray<NormalizedLandmark2D>): Float32Array {
  if (landmarks.length < NUM_LANDMARKS) {
    throw new Error(`Expected at least ${NUM_LANDMARKS} landmarks, got ${landmarks.length}`);
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const { x, y } = landmarks[i];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const spanX = Math.max(Math.abs(maxX - minX), EPS);
  const spanY = Math.max(Math.abs(maxY - minY), EPS);

  const norm = new Float32Array(NUM_HAND_FEATURES);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    norm[i * 2] = (landmarks[i].x - minX) / spanX;
    norm[i * 2 + 1] = (landmarks[i].y - minY) / spanY;
  }
  return norm;
}

function meanXY(landmarks: ReadonlyArray<NormalizedLandmark2D>): { cx: number; cy: number } {
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    sumX += landmarks[i].x;
    sumY += landmarks[i].y;
  }
  return { cx: sumX / NUM_LANDMARKS, cy: sumY / NUM_LANDMARKS };
}

/**
 * Build an 86-dim feature vector from two detected hands (sorted left-to-right by mean x).
 */
export function buildTwoHandFeatures(
  handA: ReadonlyArray<NormalizedLandmark2D>,
  handB: ReadonlyArray<NormalizedLandmark2D>,
): Float32Array {
  const meanA = meanXY(handA);
  const meanB = meanXY(handB);

  const [leftHand, leftMean, rightHand, rightMean] =
    meanA.cx <= meanB.cx ? [handA, meanA, handB, meanB] : [handB, meanB, handA, meanA];

  const leftNorm = normalizeHandBohacek(leftHand);
  const rightNorm = normalizeHandBohacek(rightHand);

  const out = new Float32Array(FEATURE_DIM);
  out.set(leftNorm, 0);
  out.set(rightNorm, NUM_HAND_FEATURES);
  out[84] = rightMean.cx - leftMean.cx; // >= 0
  out[85] = rightMean.cy - leftMean.cy;
  return out;
}

/**
 * Build an 86-dim feature vector from a single detected hand (zero-pads second slot).
 */
export function buildSingleHandFeatures(
  hand: ReadonlyArray<NormalizedLandmark2D>,
): Float32Array {
  const handNorm = normalizeHandBohacek(hand);
  const out = new Float32Array(FEATURE_DIM);
  out.set(handNorm, 0);
  return out;
}

/**
 * Convert MediaPipe HandLandmarker `results.landmarks` (0, 1, or 2 hands) to 86 features.
 * Returns null if no hands are detected.
 */
export function extractBisindo86Features(
  hands: ReadonlyArray<ReadonlyArray<NormalizedLandmark2D>>,
): Float32Array | null {
  if (hands.length === 0) return null;
  if (hands.length >= 2) {
    return buildTwoHandFeatures(hands[0], hands[1]);
  }
  return buildSingleHandFeatures(hands[0]);
}

interface SquareHandStats {
  ptsSq: Array<{ x: number; y: number }>;
  iso: Float32Array;
  span: number;
  meanX: number;
  meanY: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

function computeSquareHandStats(
  landmarks: ReadonlyArray<NormalizedLandmark2D>,
  aspectRatio: number,
): SquareHandStats {
  if (landmarks.length < NUM_LANDMARKS) {
    throw new Error(`Expected at least ${NUM_LANDMARKS} landmarks, got ${landmarks.length}`);
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let sumX = 0;
  let sumY = 0;
  const ptsSq: Array<{ x: number; y: number }> = [];

  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const sx = landmarks[i].x * aspectRatio;
    const sy = landmarks[i].y;
    ptsSq.push({ x: sx, y: sy });
    if (sx < minX) minX = sx;
    if (sx > maxX) maxX = sx;
    if (sy < minY) minY = sy;
    if (sy > maxY) maxY = sy;
    sumX += sx;
    sumY += sy;
  }

  const span = Math.max(maxX - minX, maxY - minY, EPS);
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const iso = new Float32Array(NUM_HAND_FEATURES);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    iso[i * 2] = (ptsSq[i].x - cx) / span;
    iso[i * 2 + 1] = (ptsSq[i].y - cy) / span;
  }

  return {
    ptsSq,
    iso,
    span,
    meanX: sumX / NUM_LANDMARKS,
    meanY: sumY / NUM_LANDMARKS,
    minX,
    minY,
    maxX,
    maxY,
  };
}

/**
 * Convert 1 or 2 hands into 222 aspect/distance-invariant BISINDO V2 features.
 * Matches Python `build_bisindo_v2_features` bit-for-bit within float32 tolerance.
 */
export function extractBisindo222Features(
  hands: ReadonlyArray<ReadonlyArray<NormalizedLandmark2D>>,
  aspectRatio = 1,
  swapSlots = false,
): Float32Array | null {
  if (hands.length === 0) return null;

  const out = new Float32Array(FEATURE_DIM_V2);
  if (hands.length === 1) {
    const h1 = computeSquareHandStats(hands[0], aspectRatio);
    out.set(h1.iso, 0);
    out.set(h1.iso, 84);
    out[217] = 0;
    out[218] = 0;
    out[219] = 0;
    out[220] = 1;
    out[221] = 0;
    return out;
  }

  const s0 = computeSquareHandStats(hands[0], aspectRatio);
  const s1 = computeSquareHandStats(hands[1], aspectRatio);
  const useFirstLeft = (s0.meanX <= s1.meanX) !== swapSlots;
  const left = useFirstLeft ? s0 : s1;
  const right = useFirstLeft ? s1 : s0;

  out.set(left.iso, 0);
  out.set(right.iso, 42);

  const bMinX = Math.min(left.minX, right.minX);
  const bMaxX = Math.max(left.maxX, right.maxX);
  const bMinY = Math.min(left.minY, right.minY);
  const bMaxY = Math.max(left.maxY, right.maxY);
  const bSpan = Math.max(bMaxX - bMinX, bMaxY - bMinY, EPS);
  const bCx = (bMinX + bMaxX) * 0.5;
  const bCy = (bMinY + bMaxY) * 0.5;

  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[84 + i * 2] = (left.ptsSq[i].x - bCx) / bSpan;
    out[84 + i * 2 + 1] = (left.ptsSq[i].y - bCy) / bSpan;
    out[126 + i * 2] = (right.ptsSq[i].x - bCx) / bSpan;
    out[126 + i * 2 + 1] = (right.ptsSq[i].y - bCy) / bSpan;
  }

  let offset = 168;
  for (const rIdx of INTER_HAND_KEY_INDICES) {
    const lp = left.ptsSq[rIdx];
    for (const cIdx of INTER_HAND_KEY_INDICES) {
      const rp = right.ptsSq[cIdx];
      out[offset++] = Math.hypot(lp.x - rp.x, lp.y - rp.y) / bSpan;
    }
  }

  out[217] = 1;
  out[218] = (right.meanX - left.meanX) / bSpan;
  out[219] = (right.meanY - left.meanY) / bSpan;
  out[220] = left.span / bSpan;
  out[221] = right.span / bSpan;
  return out;
}

/**
 * Return true when two detected hands have close/overlapping horizontal centroids
 * where slot-order averaging stabilizes predictions.
 */
export function shouldUseTwoHandSlotEnsemble(
  hands: ReadonlyArray<ReadonlyArray<NormalizedLandmark2D>>,
  aspectRatio = 1,
): boolean {
  if (hands.length < 2) return false;
  const s0 = computeSquareHandStats(hands[0], aspectRatio);
  const s1 = computeSquareHandStats(hands[1], aspectRatio);
  const bSpan = Math.max(
    Math.max(s0.maxX, s1.maxX) - Math.min(s0.minX, s1.minX),
    Math.max(s0.maxY, s1.maxY) - Math.min(s0.minY, s1.minY),
    EPS,
  );
  return Math.abs(s1.meanX - s0.meanX) / bSpan < 0.30;
}

export interface PredictionCandidate {
  label: string;
  confidence: number;
}

export interface HoldProgressState {
  topCandidates: PredictionCandidate[];
  bestLabel: string | null;
  bestConfidence: number;
  progress: number; // 0..1
  completed: boolean;
}

/**
 * FPS-independent pose hold validator & probability smoother with flicker drop grace.
 */
export class HoldPoseValidator {
  private readonly holdMs: number;
  private readonly minFrames: number;
  private readonly confidenceThreshold: number;
  private readonly smoothWindow: number;
  private readonly dropGraceMs: number;

  private holdStartMs: number | null = null;
  private lastMatchMs: number | null = null;
  private validFrameCount = 0;
  private probHistory: Float32Array[] = [];

  constructor(options?: {
    holdMs?: number;
    minFrames?: number;
    confidenceThreshold?: number;
    smoothWindow?: number;
    dropGraceMs?: number;
  }) {
    this.holdMs = options?.holdMs ?? 1500;
    this.minFrames = options?.minFrames ?? 5;
    this.confidenceThreshold = options?.confidenceThreshold ?? 0.75;
    this.smoothWindow = options?.smoothWindow ?? 4;
    this.dropGraceMs = options?.dropGraceMs ?? 250;
  }

  reset(): void {
    this.holdStartMs = null;
    this.lastMatchMs = null;
    this.validFrameCount = 0;
    this.probHistory = [];
  }

  /**
   * Handle a frame with no detected hands, preserving progress briefly during dropGraceMs.
   */
  notifyNoHands(nowMs: number = performance.now()): HoldProgressState {
    if (
      this.holdStartMs !== null &&
      this.lastMatchMs !== null &&
      nowMs - this.lastMatchMs <= this.dropGraceMs
    ) {
      const steadyElapsed = Math.max(0, this.lastMatchMs - this.holdStartMs);
      const steadyProgress =
        this.holdMs <= 0 ? 1 : Math.min(1, steadyElapsed / this.holdMs);
      return {
        topCandidates: [],
        bestLabel: null,
        bestConfidence: 0,
        progress: steadyProgress,
        completed: false,
      };
    }

    this.reset();
    return {
      topCandidates: [],
      bestLabel: null,
      bestConfidence: 0,
      progress: 0,
      completed: false,
    };
  }

  update(
    rawScores: ArrayLike<number>,
    labels: ReadonlyArray<string>,
    targetLabel: string | null,
    nowMs: number = performance.now(),
    confidenceThresholdOverride?: number,
  ): HoldProgressState {
    const numClasses = labels.length;
    const current = new Float32Array(numClasses);
    for (let i = 0; i < numClasses; i++) current[i] = rawScores[i] ?? 0;

    this.probHistory.push(current);
    if (this.probHistory.length > this.smoothWindow) {
      this.probHistory.shift();
    }

    const averaged = new Float32Array(numClasses);
    for (const frameScores of this.probHistory) {
      for (let i = 0; i < numClasses; i++) averaged[i] += frameScores[i];
    }
    for (let i = 0; i < numClasses; i++) averaged[i] /= this.probHistory.length;

    const candidates: PredictionCandidate[] = labels
      .map((label, idx) => ({ label, confidence: averaged[idx] }))
      .filter((c) => c.label !== "NOTHING")
      .sort((a, b) => b.confidence - a.confidence);

    const best = candidates[0] ?? null;
    const activeThreshold = confidenceThresholdOverride ?? this.confidenceThreshold;
    const effectiveThreshold =
      this.holdStartMs !== null
        ? Math.max(0.42, activeThreshold - 0.12)
        : activeThreshold;
    const isTargetMatch =
      best !== null &&
      best.confidence >= effectiveThreshold &&
      (targetLabel === null || best.label === targetLabel);

    if (!isTargetMatch) {
      if (
        this.holdStartMs !== null &&
        this.lastMatchMs !== null &&
        nowMs - this.lastMatchMs <= this.dropGraceMs
      ) {
        const steadyElapsed = Math.max(0, this.lastMatchMs - this.holdStartMs);
        const steadyProgress =
          this.holdMs <= 0 ? 1 : Math.min(1, steadyElapsed / this.holdMs);
        return {
          topCandidates: candidates.slice(0, 3),
          bestLabel: best?.label ?? null,
          bestConfidence: best?.confidence ?? 0,
          progress: steadyProgress,
          completed: false,
        };
      }
      this.holdStartMs = null;
      this.lastMatchMs = null;
      this.validFrameCount = 0;
      return {
        topCandidates: candidates.slice(0, 3),
        bestLabel: best?.label ?? null,
        bestConfidence: best?.confidence ?? 0,
        progress: 0,
        completed: false,
      };
    }

    if (this.holdStartMs === null) {
      this.holdStartMs = nowMs;
      this.validFrameCount = 1;
    } else {
      this.validFrameCount += 1;
    }
    this.lastMatchMs = nowMs;

    const elapsed = Math.max(0, nowMs - this.holdStartMs);
    const timeProgress = this.holdMs <= 0 ? 1 : Math.min(1, elapsed / this.holdMs);
    const completed =
      this.holdMs <= 0
        ? true
        : elapsed >= this.holdMs && this.validFrameCount >= this.minFrames;

    return {
      topCandidates: candidates.slice(0, 3),
      bestLabel: best.label,
      bestConfidence: best.confidence,
      progress: timeProgress,
      completed,
    };
  }
}
