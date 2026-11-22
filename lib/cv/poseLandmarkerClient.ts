/**
 * Browser MediaPipe PoseLandmarker wrapper and upper-body anchor renderer for WL-BISINDO.
 */

import type { PoseLandmarker as PoseLandmarkerType, PoseLandmarkerResult } from "@mediapipe/tasks-vision";
import {
  fetchCachedModelBuffer,
  getVisionFileset,
  resolvePoseModelUrl,
} from "./mediapipeLoader";
import type { NormalizedLandmark2D } from "./normalizeLandmarks";

// Order matches src/bisindo/word_level.py POSE_ANCHOR_IDX:
// [0:nose, 1:L_shoulder(11), 2:R_shoulder(12), 3:L_ear(7), 4:R_ear(8), 5:L_elbow(13), 6:R_elbow(14)]
export const POSE_ANCHOR_INDICES = [0, 11, 12, 7, 8, 13, 14] as const;

const ANCHOR_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [1, 2], // L_shoulder <-> R_shoulder
  [1, 5], // L_shoulder <-> L_elbow
  [2, 6], // R_shoulder <-> R_elbow
  [0, 3], // nose <-> L_ear
  [0, 4], // nose <-> R_ear
];

let posePromise: Promise<PoseLandmarkerType> | null = null;

async function createPoseLandmarker(): Promise<PoseLandmarkerType> {
  const [{ PoseLandmarker }, modelUrl] = await Promise.all([
    import("@mediapipe/tasks-vision"),
    resolvePoseModelUrl(),
  ]);

  let buffer: Uint8Array | null = null;
  let fileset: Awaited<ReturnType<typeof getVisionFileset>>;

  try {
    [fileset, buffer] = await Promise.all([
      getVisionFileset(),
      fetchCachedModelBuffer("pose", modelUrl),
    ]);
  } catch {
    fileset = await getVisionFileset();
  }

  const commonOptions = {
    runningMode: "VIDEO" as const,
    numPoses: 1,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
    outputSegmentationMasks: false,
  };

  if (buffer) {
    try {
      return await PoseLandmarker.createFromOptions(fileset, {
        ...commonOptions,
        baseOptions: { modelAssetBuffer: buffer.slice(), delegate: "GPU" },
      });
    } catch {
      try {
        return await PoseLandmarker.createFromOptions(fileset, {
          ...commonOptions,
          baseOptions: { modelAssetBuffer: buffer.slice(), delegate: "CPU" },
        });
      } catch {
        // Fallback to URL path if buffer creation fails
      }
    }
  }

  try {
    return await PoseLandmarker.createFromOptions(fileset, {
      ...commonOptions,
      baseOptions: { modelAssetPath: modelUrl, delegate: "GPU" },
    });
  } catch {
    return await PoseLandmarker.createFromOptions(fileset, {
      ...commonOptions,
      baseOptions: { modelAssetPath: modelUrl, delegate: "CPU" },
    });
  }
}

/**
 * Load and cache MediaPipe PoseLandmarker configured for single-person WL-BISINDO tracking.
 */
export function loadPoseLandmarker(): Promise<PoseLandmarkerType> {
  if (!posePromise) {
    posePromise = createPoseLandmarker().catch((err) => {
      posePromise = null;
      throw err;
    });
  }
  return posePromise;
}

/**
 * Run pose detection on an HTMLVideoElement frame.
 */
export function detectVideoPose(
  landmarker: PoseLandmarkerType,
  video: HTMLVideoElement,
  timestampMs: number,
): PoseLandmarkerResult {
  return landmarker.detectForVideo(video, timestampMs);
}

/**
 * Extract the 7 WL-BISINDO upper-body anchors from MediaPipe's 33 pose landmarks.
 */
export function extractPoseAnchors(
  poseLandmarks: ReadonlyArray<NormalizedLandmark2D> | undefined,
): NormalizedLandmark2D[] | null {
  if (!poseLandmarks || poseLandmarks.length < 15) return null;
  return POSE_ANCHOR_INDICES.map((idx) => poseLandmarks[idx]);
}

/**
 * Draw the 7 upper-body pose anchors (shoulders, elbows, ears, nose) onto a 2D canvas
 * without clearing existing drawings so it composes cleanly after `drawHandLandmarks`.
 */
export function drawPoseAnchors(
  ctx: CanvasRenderingContext2D,
  anchors: ReadonlyArray<NormalizedLandmark2D> | null,
  width: number,
  height: number,
  mirror = false,
): void {
  if (!anchors || anchors.length < 7) return;

  ctx.save();
  ctx.lineWidth = Math.max(2, width * 0.004);
  ctx.strokeStyle = "#FFC93C";

  for (const [startIdx, endIdx] of ANCHOR_CONNECTIONS) {
    const a = anchors[startIdx];
    const b = anchors[endIdx];
    if (!a || !b) continue;
    const ax = (mirror ? 1 - a.x : a.x) * width;
    const ay = a.y * height;
    const bx = (mirror ? 1 - b.x : b.x) * width;
    const by = b.y * height;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // Subtle shoulder-midpoint neck connector to nose for clearer upper-body context
  const nose = anchors[0];
  const lShoulder = anchors[1];
  const rShoulder = anchors[2];
  if (nose && lShoulder && rShoulder) {
    const midX = (mirror ? 1 - (lShoulder.x + rShoulder.x) / 2 : (lShoulder.x + rShoulder.x) / 2) * width;
    const midY = ((lShoulder.y + rShoulder.y) / 2) * height;
    const nx = (mirror ? 1 - nose.x : nose.x) * width;
    const ny = nose.y * height;
    ctx.beginPath();
    ctx.moveTo(midX, midY);
    ctx.lineTo(nx, ny);
    ctx.stroke();
  }

  const radius = Math.max(4, width * 0.007);
  for (let i = 0; i < anchors.length; i++) {
    const pt = anchors[i];
    const cx = (mirror ? 1 - pt.x : pt.x) * width;
    const cy = pt.y * height;
    ctx.beginPath();
    ctx.arc(cx, cy, i === 0 ? radius * 1.15 : radius, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#FF5D73" : "#FFC93C";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#1B1B3A";
    ctx.stroke();
  }
  ctx.restore();
}
