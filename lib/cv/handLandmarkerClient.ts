/**
 * Browser MediaPipe HandLandmarker wrapper (numHands: 2) and canvas overlay renderer.
 */

import type { HandLandmarker as HandLandmarkerType, HandLandmarkerResult } from "@mediapipe/tasks-vision";
import {
  fetchCachedModelBuffer,
  getVisionFileset,
  resolveHandModelUrl,
} from "./mediapipeLoader";
import type { NormalizedLandmark2D } from "./normalizeLandmarks";

const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], // Pinky & palm
];

let landmarkerPromise: Promise<HandLandmarkerType> | null = null;
let handLandmarkerReady = false;

export function isHandLandmarkerReady(): boolean {
  return handLandmarkerReady;
}

async function createHandLandmarker(): Promise<HandLandmarkerType> {
  const [{ HandLandmarker }, modelUrl] = await Promise.all([
    import("@mediapipe/tasks-vision"),
    resolveHandModelUrl(),
  ]);

  let buffer: Uint8Array | null = null;
  let fileset: Awaited<ReturnType<typeof getVisionFileset>>;

  try {
    [fileset, buffer] = await Promise.all([
      getVisionFileset(),
      fetchCachedModelBuffer("hand", modelUrl),
    ]);
  } catch {
    fileset = await getVisionFileset();
  }

  const commonOptions = {
    runningMode: "VIDEO" as const,
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  };

  if (buffer) {
    try {
      return await HandLandmarker.createFromOptions(fileset, {
        ...commonOptions,
        baseOptions: { modelAssetBuffer: buffer.slice(), delegate: "GPU" },
      });
    } catch {
      try {
        return await HandLandmarker.createFromOptions(fileset, {
          ...commonOptions,
          baseOptions: { modelAssetBuffer: buffer.slice(), delegate: "CPU" },
        });
      } catch {
        // Fallback to URL path if buffer creation fails
      }
    }
  }

  try {
    return await HandLandmarker.createFromOptions(fileset, {
      ...commonOptions,
      baseOptions: { modelAssetPath: modelUrl, delegate: "GPU" },
    });
  } catch {
    return await HandLandmarker.createFromOptions(fileset, {
      ...commonOptions,
      baseOptions: { modelAssetPath: modelUrl, delegate: "CPU" },
    });
  }
}

/**
 * Load and cache MediaPipe HandLandmarker configured for 2-hand BISINDO detection.
 * Uses shared FilesetResolver and browser Cache API for fast cold and repeat loads.
 */
export function loadHandLandmarker(): Promise<HandLandmarkerType> {
  if (!landmarkerPromise) {
    landmarkerPromise = createHandLandmarker()
      .then((instance) => {
        handLandmarkerReady = true;
        return instance;
      })
      .catch((err) => {
        handLandmarkerReady = false;
        landmarkerPromise = null;
        throw err;
      });
  }
  return landmarkerPromise;
}

/**
 * Run 2-hand landmark detection on an HTMLVideoElement frame.
 */
export function detectVideoHands(
  landmarker: HandLandmarkerType,
  video: HTMLVideoElement,
  timestampMs: number,
): HandLandmarkerResult {
  return landmarker.detectForVideo(video, timestampMs);
}

/**
 * Draw 21-point hand skeletons onto an overlay 2D canvas.
 */
export function drawHandLandmarks(
  ctx: CanvasRenderingContext2D,
  hands: ReadonlyArray<ReadonlyArray<NormalizedLandmark2D>>,
  width: number,
  height: number,
  mirror = false,
): void {
  ctx.clearRect(0, 0, width, height);
  if (hands.length === 0) return;

  const palette = ["#22C55E", "#38BDF8"];

  hands.forEach((landmarks, handIdx) => {
    const strokeColor = palette[handIdx % palette.length];
    ctx.lineWidth = Math.max(2, width * 0.004);
    ctx.strokeStyle = strokeColor;

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const a = landmarks[startIdx];
      const b = landmarks[endIdx];
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

    const radius = Math.max(3, width * 0.006);
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      const cx = (mirror ? 1 - lm.x : lm.x) * width;
      const cy = lm.y * height;

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? "#FACC15" : "#FFFFFF";
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#111827";
      ctx.stroke();
    }
  });
}
