/**
 * Shared MediaPipe WASM FilesetResolver, persistent browser Cache API storage,
 * and download progress tracking for HandLandmarker & PoseLandmarker.
 */

import type { FilesetResolver as FilesetResolverType } from "@mediapipe/tasks-vision";

export const MEDIAPIPE_WASM_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";
export const MEDIAPIPE_WASM_LOCAL = "/models/mediapipe/wasm";

export const HAND_MODEL_CDN =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
export const HAND_MODEL_LOCAL = "/models/mediapipe/hand_landmarker.task";

export const POSE_MODEL_CDN =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
export const POSE_MODEL_LOCAL = "/models/mediapipe/pose_landmarker_lite.task";

export const MODEL_CACHE_NAME = "signify-cv-models-v1";

export type ModelKey = "hand" | "pose";

type WasmFileset = Awaited<ReturnType<typeof FilesetResolverType.forVisionTasks>>;

const modelProgress: Record<ModelKey, number> = {
  hand: 0,
  pose: 0,
};

const progressListeners: Record<ModelKey, Set<(progress: number) => void>> = {
  hand: new Set(),
  pose: new Set(),
};

const inFlightBuffers = new Map<string, Promise<Uint8Array>>();
let localMediaPipeAvailablePromise: Promise<boolean> | null = null;
let sharedFilesetPromise: Promise<WasmFileset> | null = null;

function notifyProgress(key: ModelKey, value: number): void {
  const clamped = Math.max(0, Math.min(1, value));
  modelProgress[key] = clamped;
  progressListeners[key].forEach((listener) => listener(clamped));
}

export function getModelProgress(key: ModelKey): number {
  return modelProgress[key];
}

export function subscribeModelProgress(
  key: ModelKey,
  listener: (progress: number) => void,
): () => void {
  progressListeners[key].add(listener);
  listener(modelProgress[key]);
  return () => {
    progressListeners[key].delete(listener);
  };
}

/**
 * Determine whether local `/models/mediapipe/` assets exist (dev only unless explicitly forced).
 * In production builds, skips local probes to avoid 2x 404 latency before hitting CDN.
 */
export function shouldUseLocalMediaPipe(): Promise<boolean> {
  if (
    process.env.NODE_ENV === "production" &&
    process.env.NEXT_PUBLIC_MEDIAPIPE_LOCAL !== "true"
  ) {
    return Promise.resolve(false);
  }

  if (typeof window === "undefined") {
    return Promise.resolve(false);
  }

  if (!localMediaPipeAvailablePromise) {
    localMediaPipeAvailablePromise = fetch(HAND_MODEL_LOCAL, { method: "HEAD" })
      .then((res) => res.ok)
      .catch(() => false);
  }

  return localMediaPipeAvailablePromise;
}

export async function resolveHandModelUrl(): Promise<string> {
  return (await shouldUseLocalMediaPipe()) ? HAND_MODEL_LOCAL : HAND_MODEL_CDN;
}

export async function resolvePoseModelUrl(): Promise<string> {
  return (await shouldUseLocalMediaPipe()) ? POSE_MODEL_LOCAL : POSE_MODEL_CDN;
}

/**
 * Singleton MediaPipe WASM FilesetResolver shared across HandLandmarker and PoseLandmarker.
 */
export function getVisionFileset(): Promise<WasmFileset> {
  if (!sharedFilesetPromise) {
    sharedFilesetPromise = (async () => {
      const [{ FilesetResolver }, useLocal] = await Promise.all([
        import("@mediapipe/tasks-vision"),
        shouldUseLocalMediaPipe(),
      ]);
      const wasmBase = useLocal ? MEDIAPIPE_WASM_LOCAL : MEDIAPIPE_WASM_CDN;
      return FilesetResolver.forVisionTasks(wasmBase);
    })().catch((err) => {
      sharedFilesetPromise = null;
      throw err;
    });
  }
  return sharedFilesetPromise;
}

/**
 * Fetch a `.task` model binary with persistent browser Cache API storage and streaming progress.
 */
export function fetchCachedModelBuffer(key: ModelKey, url: string): Promise<Uint8Array> {
  const existing = inFlightBuffers.get(url);
  if (existing) return existing;

  const promise = (async (): Promise<Uint8Array> => {
    const canUseCache = typeof window !== "undefined" && "caches" in window;
    if (canUseCache) {
      try {
        const cache = await window.caches.open(MODEL_CACHE_NAME);
        const cached = await cache.match(url);
        if (cached) {
          const buf = new Uint8Array(await cached.arrayBuffer());
          notifyProgress(key, 1);
          return buf;
        }
      } catch {
        // Ignore Cache API read errors (e.g. restricted storage contexts)
      }
    }

    notifyProgress(key, 0.05);
    const res = await fetch(url, { mode: "cors" });
    if (!res.ok) {
      throw new Error(`Failed to download MediaPipe model (${res.status}): ${url}`);
    }

    let bytes: Uint8Array;
    const totalHeader = res.headers.get("content-length");
    const totalBytes = totalHeader ? Number(totalHeader) : 0;

    if (res.body && totalBytes > 0) {
      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];
      let received = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          received += value.byteLength;
          notifyProgress(key, 0.05 + 0.9 * Math.min(1, received / totalBytes));
        }
      }

      bytes = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
    } else {
      bytes = new Uint8Array(await res.arrayBuffer());
    }

    if (canUseCache) {
      try {
        const cache = await window.caches.open(MODEL_CACHE_NAME);
        await cache.put(
          url,
          new Response(bytes.slice(), {
            headers: { "Content-Type": "application/octet-stream" },
          }),
        );
      } catch {
        // Ignore quota / private-browsing cache write failures
      }
    }

    notifyProgress(key, 1);
    return bytes;
  })().catch((err) => {
    inFlightBuffers.delete(url);
    throw err;
  });

  inFlightBuffers.set(url, promise);
  return promise;
}

/**
 * Respect Save-Data and slow 2G connections before triggering background prefetches.
 */
export function shouldBackgroundPrefetch(): boolean {
  if (typeof navigator === "undefined") return false;
  const conn = (
    navigator as Navigator & {
      connection?: { saveData?: boolean; effectiveType?: string };
    }
  ).connection;
  if (!conn) return true;
  if (conn.saveData) return false;
  if (conn.effectiveType === "slow-2g" || conn.effectiveType === "2g") {
    return false;
  }
  return true;
}
