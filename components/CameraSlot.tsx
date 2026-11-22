"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import { speak } from "@/lib/audio";
import { handsOf } from "@/lib/bisindo";
import { loadAlphabetClassifier, type AlphabetClassifier } from "@/lib/cv/alphabetClassifier";
import { detectVideoHands, drawHandLandmarks, loadHandLandmarker } from "@/lib/cv/handLandmarkerClient";
import { subscribeModelProgress } from "@/lib/cv/mediapipeLoader";
import {
  extractBisindo222Features,
  extractBisindo86Features,
  FEATURE_DIM_V2,
  HoldPoseValidator,
  shouldUseTwoHandSlotEnsemble,
  type HoldProgressState,
} from "@/lib/cv/normalizeLandmarks";
import { useProgress } from "@/lib/storage/useProgress";

type VideoMode = "idle" | "webcam" | "simulation";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const HOLD_MS = 1500;
const SINGLE_HAND_CONFIDENCE_THRESHOLD = 0.68;
const TWO_HAND_CONFIDENCE_THRESHOLD = 0.55;

const EMPTY_HOLD_STATE: HoldProgressState = {
  topCandidates: [],
  bestLabel: null,
  bestConfidence: 0,
  progress: 0,
  completed: false,
};

function stopMediaStream(stream: MediaStream | null): void {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
}

function drawSimulationContain(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  width: number,
  height: number,
  dx = 0,
  dy = 0,
): void {
  ctx.fillStyle = "#18181B";
  ctx.fillRect(0, 0, width, height);
  if (!img.complete || img.naturalWidth <= 0 || img.naturalHeight <= 0) return;
  const scale = Math.min(width / img.naturalWidth, height / img.naturalHeight);
  const drawW = img.naturalWidth * scale;
  const drawH = img.naturalHeight * scale;
  const offsetX = (width - drawW) * 0.5 + dx;
  const offsetY = (height - drawH) * 0.5 + dy;
  ctx.drawImage(img, offsetX, offsetY, drawW, drawH);
}

export default function CameraSlot({ letter }: { letter: string }) {
  const { data, markLearned } = useProgress();
  const [mode, setMode] = useState<VideoMode>("idle");
  const [simLetter, setSimLetter] = useState<string>(letter);
  const [isLoadingEngine, setIsLoadingEngine] = useState(false);
  const [modelProgress, setModelProgress] = useState(0);
  const [hasLiveStream, setHasLiveStream] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [showLandmarks, setShowLandmarks] = useState(data.settings.showLandmarks);
  const [detectedHands, setDetectedHands] = useState(0);
  const [holdState, setHoldState] = useState<HoldProgressState>(EMPTY_HOLD_STATE);
  const [justCompleted, setJustCompleted] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const simImageRef = useRef<HTMLImageElement | null>(null);
  const activeStreamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const classifierRef = useRef<AlphabetClassifier | null>(null);
  const validatorRef = useRef(
    new HoldPoseValidator({
      holdMs: HOLD_MS,
      minFrames: 5,
      confidenceThreshold: SINGLE_HAND_CONFIDENCE_THRESHOLD,
      smoothWindow: 5,
      dropGraceMs: 320,
    }),
  );

  const rafRef = useRef<number | null>(null);
  const lastDetectTsRef = useRef<number>(0);
  const targetLetterRef = useRef(letter);
  const showLandmarksRef = useRef(showLandmarks);
  const audioEnabledRef = useRef(data.settings.audioEnabled);
  const celebratedRef = useRef(false);
  const isMountedRef = useRef(true);

  targetLetterRef.current = letter;
  showLandmarksRef.current = showLandmarks;
  audioEnabledRef.current = data.settings.audioEnabled;

  // Reset hold progress whenever the target letter changes.
  useEffect(() => {
    setSimLetter(letter);
    validatorRef.current.reset();
    celebratedRef.current = false;
    setJustCompleted(false);
    setHoldState(EMPTY_HOLD_STATE);
  }, [letter]);

  // Load chosen simulation sample image into ref whenever `simLetter` updates.
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `/samples/bisindo/${simLetter}.jpg`;
    img.onload = () => {
      simImageRef.current = img;
    };
  }, [simLetter]);

  useEffect(() => {
    return subscribeModelProgress("hand", setModelProgress);
  }, []);

  const stopAll = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    stopMediaStream(activeStreamRef.current);
    activeStreamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    const canvas = overlayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
    validatorRef.current.reset();
    celebratedRef.current = false;
    setHasLiveStream(false);
    setDetectedHands(0);
    setHoldState(EMPTY_HOLD_STATE);
    setJustCompleted(false);
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      stopAll();
    };
  }, [stopAll]);

  const ensureAiEngine = useCallback(async () => {
    if (landmarkerRef.current && classifierRef.current) {
      return { landmarker: landmarkerRef.current, classifier: classifierRef.current };
    }
    setIsLoadingEngine(true);
    try {
      const [landmarker, classifier] = await Promise.all([
        loadHandLandmarker(),
        loadAlphabetClassifier(),
      ]);
      landmarkerRef.current = landmarker;
      classifierRef.current = classifier;
      return { landmarker, classifier };
    } finally {
      if (isMountedRef.current) {
        setIsLoadingEngine(false);
      }
    }
  }, []);

  const startDetectionLoop = useCallback((activeMode: VideoMode) => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }

    const tick = (rafTime: number) => {
      const video = videoRef.current;
      const overlay = overlayCanvasRef.current;
      const landmarker = landmarkerRef.current;
      const classifier = classifierRef.current;

      // Keep offscreen simulation canvas streaming fresh frames with preserved aspect ratio.
      if (activeMode === "simulation" && offscreenCanvasRef.current && simImageRef.current) {
        const off = offscreenCanvasRef.current;
        const offCtx = off.getContext("2d");
        const img = simImageRef.current;
        if (offCtx && img.complete && img.naturalWidth > 0) {
          const dx = Math.sin(rafTime * 0.002) * 2;
          const dy = Math.cos(rafTime * 0.0017) * 2;
          drawSimulationContain(offCtx, img, off.width, off.height, dx, dy);
        }
      }

      if (
        video &&
        overlay &&
        landmarker &&
        classifier &&
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        video.videoHeight > 0
      ) {
        if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
          overlay.width = video.videoWidth;
          overlay.height = video.videoHeight;
        }

        const nowMs = Math.max(performance.now(), lastDetectTsRef.current + 1);
        lastDetectTsRef.current = nowMs;

        const result = detectVideoHands(landmarker, video, nowMs);
        const hands = result.landmarks ?? [];
        setDetectedHands(hands.length);

        const ctx = overlay.getContext("2d");
        if (ctx) {
          if (showLandmarksRef.current) {
            drawHandLandmarks(
              ctx,
              hands,
              overlay.width,
              overlay.height,
              activeMode === "webcam",
            );
          } else {
            ctx.clearRect(0, 0, overlay.width, overlay.height);
          }
        }

        const aspectRatio = video.videoWidth / Math.max(1, video.videoHeight);
        const features =
          classifier.inputDim === FEATURE_DIM_V2
            ? extractBisindo222Features(hands, aspectRatio)
            : extractBisindo86Features(hands);

        if (features) {
          let probs = classifier.predictProbabilities(features);
          if (
            classifier.inputDim === FEATURE_DIM_V2 &&
            shouldUseTwoHandSlotEnsemble(hands, aspectRatio)
          ) {
            const swappedFeatures = extractBisindo222Features(hands, aspectRatio, true);
            if (swappedFeatures) {
              const swappedProbs = classifier.predictProbabilities(swappedFeatures);
              const blended = new Float32Array(probs.length);
              for (let i = 0; i < probs.length; i++) {
                blended[i] = (probs[i] + swappedProbs[i]) * 0.5;
              }
              probs = blended;
            }
          }

          const targetThreshold =
            handsOf(targetLetterRef.current) === 2
              ? TWO_HAND_CONFIDENCE_THRESHOLD
              : SINGLE_HAND_CONFIDENCE_THRESHOLD;

          const nextState = validatorRef.current.update(
            probs,
            classifier.labels,
            targetLetterRef.current,
            nowMs,
            targetThreshold,
          );
          setHoldState(nextState);

          if (nextState.completed && !celebratedRef.current) {
            celebratedRef.current = true;
            setJustCompleted(true);
            if (activeMode === "webcam") {
              markLearned("bisindo", targetLetterRef.current, 10);
              speak(`Berhasil! Huruf ${targetLetterRef.current}`, audioEnabledRef.current);
            }
          }
        } else {
          setHoldState(validatorRef.current.notifyNoHands(nowMs));
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
  }, [markLearned]);

  const startWebcam = useCallback(async () => {
    stopAll();
    setCameraError(null);
    setMode("webcam");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Browser ini tidak mendukung akses kamera.");
      }

      const enginePromise = ensureAiEngine();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });

      if (!isMountedRef.current) {
        stopMediaStream(stream);
        return;
      }

      activeStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setHasLiveStream(true);

      await enginePromise;
      if (!isMountedRef.current) return;
      startDetectionLoop("webcam");
    } catch (err) {
      stopAll();
      setMode("idle");
      const message =
        err instanceof Error && err.name === "NotAllowedError"
          ? "Izin kamera ditolak oleh browser. Izinkan kamera atau coba Mode Video Simulasi."
          : err instanceof Error
            ? `Kamera tidak tersedia: ${err.message}`
            : "Kamera gagal dibuka. Gunakan Video Simulasi untuk menguji model AI.";
      setCameraError(message);
    }
  }, [ensureAiEngine, startDetectionLoop, stopAll]);

  const startSimulation = useCallback(async () => {
    stopAll();
    setCameraError(null);
    setMode("simulation");

    try {
      await ensureAiEngine();

      const canvas = offscreenCanvasRef.current ?? document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 480;
      offscreenCanvasRef.current = canvas;

      const expectedSuffix = `/samples/bisindo/${simLetter}.jpg`;
      const currentSrc = simImageRef.current?.src ?? "";
      if (!simImageRef.current || !simImageRef.current.complete || !currentSrc.endsWith(expectedSuffix)) {
        await new Promise<void>((resolve, reject) => {
          const img = new Image();
          img.crossOrigin = "anonymous";
          img.src = expectedSuffix;
          img.onload = () => {
            simImageRef.current = img;
            resolve();
          };
          img.onerror = () => reject(new Error(`Gagal memuat contoh foto huruf ${simLetter}.`));
        });
      }

      const offCtx = canvas.getContext("2d");
      if (offCtx && simImageRef.current) {
        drawSimulationContain(offCtx, simImageRef.current, canvas.width, canvas.height);
      }

      if (typeof canvas.captureStream !== "function") {
        throw new Error("Browser tidak mendukung canvas captureStream untuk simulasi video.");
      }

      const stream = canvas.captureStream(30);
      if (!isMountedRef.current) {
        stopMediaStream(stream);
        return;
      }

      activeStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      startDetectionLoop("simulation");
    } catch (err) {
      stopAll();
      setMode("idle");
      setCameraError(err instanceof Error ? err.message : "Gagal menjalankan video simulasi.");
    }
  }, [ensureAiEngine, simLetter, startDetectionLoop, stopAll]);

  const handleStop = useCallback(() => {
    stopAll();
    setCameraError(null);
    setMode("idle");
  }, [stopAll]);

  const requiredHands = handsOf(letter);
  const isAlreadyLearned = data.bisindo.learnedLetters.includes(letter);
  const holdPercent = Math.round(holdState.progress * 100);

  return (
    <div className="card overflow-hidden bg-ink p-2.5 text-white sm:p-4">
      <div className="relative aspect-[16/10] max-h-[30dvh] sm:max-h-[36dvh] min-h-[160px] w-full overflow-hidden rounded-xl sm:rounded-2xl border-2 border-white/20 bg-black/50">
        <video
          ref={videoRef}
          muted
          playsInline
          autoPlay
          className={`h-full w-full object-contain ${mode === "webcam" ? "-scale-x-100" : ""} ${mode === "idle" ? "hidden" : "block"}`}
        />
        <canvas
          ref={overlayCanvasRef}
          className={`pointer-events-none absolute inset-0 h-full w-full object-contain ${mode === "idle" ? "hidden" : "block"}`}
        />

        {mode !== "idle" && (
          <div className="pointer-events-none absolute inset-x-2 sm:inset-x-3 top-2 sm:top-3 flex flex-wrap items-center justify-between gap-1.5 sm:gap-2">
            <span
              className={`inline-flex items-center gap-1.5 sm:gap-2 rounded-full border border-ink sm:border-2 px-2.5 sm:px-3 py-0.5 sm:py-1 text-[10px] sm:text-xs font-bold ${
                mode === "webcam" ? "bg-coral text-white" : "bg-sun text-ink"
              }`}
            >
              <span className="h-2 w-2 sm:h-2.5 sm:w-2.5 rounded-full bg-current" aria-hidden="true" />
              {mode === "webcam" ? "LIVE WEBCAM" : `MODE SIMULASI (${simLetter})`}
            </span>
            <span
              className={`rounded-full border border-ink sm:border-2 px-2.5 sm:px-3 py-0.5 sm:py-1 text-[10px] sm:text-xs font-bold text-ink ${
                detectedHands >= requiredHands
                  ? "bg-mint"
                  : requiredHands === 2 && detectedHands === 1
                    ? "bg-sun"
                    : "bg-white"
              }`}
            >
              {detectedHands === 0
                ? `Belum ada tangan (butuh ${requiredHands})`
                : `${detectedHands}/${requiredHands} Tangan`}
            </span>
          </div>
        )}

        {mode !== "idle" && !isLoadingEngine && detectedHands === 0 && (
          <div className="pointer-events-none absolute inset-x-3 sm:inset-x-4 bottom-2.5 sm:bottom-4 rounded-xl border border-white/20 bg-ink/85 px-3 sm:px-4 py-1.5 sm:py-2 text-center text-xs sm:text-sm text-white">
            Arahkan {requiredHands === 2 ? "2 tangan" : "tangan"} ke dalam kotak kamera
          </div>
        )}

        {mode !== "idle" && !isLoadingEngine && requiredHands === 2 && detectedHands === 1 && (
          <div className="pointer-events-none absolute inset-x-3 sm:inset-x-4 bottom-2.5 sm:bottom-4 rounded-xl border border-sun/60 sm:border-2 bg-ink/90 px-3 sm:px-4 py-1.5 sm:py-2 text-center text-[11px] font-semibold text-sun sm:text-sm">
            Huruf {letter} membutuhkan 2 tangan (baru 1 tangan terbaca). Jauhkan sedikit dari kamera agar kedua tangan terlihat jelas.
          </div>
        )}

        {isLoadingEngine && mode === "webcam" && hasLiveStream && (
          <div className="pointer-events-none absolute inset-x-3 sm:inset-x-4 bottom-2.5 sm:bottom-4 flex flex-col items-center gap-1.5 rounded-xl border border-white/25 bg-ink/85 px-3 py-2 text-center backdrop-blur-sm">
            <p className="text-xs font-semibold text-white">
              Menyiapkan model AI... {Math.round(modelProgress * 100)}%
            </p>
            <div className="h-1.5 w-full max-w-[200px] overflow-hidden rounded-full bg-white/20">
              <div
                className="h-full rounded-full bg-sun transition-all duration-150"
                style={{ width: `${Math.round(modelProgress * 100)}%` }}
              />
            </div>
          </div>
        )}

        {isLoadingEngine && !(mode === "webcam" && hasLiveStream) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 sm:gap-3 bg-ink/90 p-4 sm:p-6 text-center">
            <div
              className="h-8 w-8 sm:h-10 sm:w-10 animate-spin rounded-full border-4 border-sun border-t-transparent"
              aria-hidden="true"
            />
            <p className="font-display text-sm sm:text-lg font-semibold">
              Memuat model AI MediaPipe &amp; BISINDO... {Math.round(modelProgress * 100)}%
            </p>
            <div className="h-2 w-48 overflow-hidden rounded-full bg-white/20">
              <div
                className="h-full rounded-full bg-sun transition-all duration-150"
                style={{ width: `${Math.round(modelProgress * 100)}%` }}
              />
            </div>
            <p className="text-xs sm:text-sm text-white/80">Semua pemrosesan berjalan langsung di browser perangkatmu.</p>
          </div>
        )}

        {mode === "idle" && !isLoadingEngine && (
          <div className="flex h-full flex-col items-center justify-center gap-2 sm:gap-3 p-3 sm:p-4 text-center">
            <p className="font-display text-base sm:text-lg font-semibold">Area kamera untuk huruf {letter}</p>
            <p className="max-w-sm text-[11px] sm:text-xs text-white/90">
              Diproses di perangkatmu, tidak dikirim ke server mana pun.
            </p>

            {cameraError && (
              <div role="alert" className="max-w-md rounded-xl border-2 border-coral bg-coral/20 px-3 py-2 text-xs sm:text-sm text-white">
                {cameraError}
              </div>
            )}

            <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-3 w-full sm:w-auto">
              <button type="button" onClick={startWebcam} className="btn bg-sun text-ink text-xs sm:text-base py-2.5 sm:py-3">
                Gunakan Kamera
              </button>
              <button type="button" onClick={startSimulation} className="btn bg-white text-ink text-xs sm:text-base py-2.5 sm:py-3">
                Gunakan Video Simulasi
              </button>
            </div>
          </div>
        )}
      </div>

      {mode !== "idle" && (
        <div className="mt-2 space-y-2">
          {/* KONTROL: tepat di bawah video agar selalu terlihat */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setShowLandmarks((prev) => !prev)}
              className="btn !min-h-[44px] !px-4 !text-sm bg-white text-ink"
            >
              {showLandmarks ? "Sembunyikan Titik Jari" : "Tampilkan Titik Jari"}
            </button>

            {mode === "webcam" ? (
              <button
                type="button"
                onClick={startSimulation}
                className="btn !min-h-[44px] !px-4 !text-sm bg-sun text-ink"
              >
                Coba Simulasi
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={startWebcam}
                  className="btn !min-h-[44px] !px-4 !text-sm bg-sun text-ink"
                >
                  Pakai Kamera
                </button>
                <label className="flex min-h-[44px] items-center gap-2 rounded-full border-2 border-white/30 bg-white/10 px-3 text-sm font-semibold">
                  <span>Klip:</span>
                  <select
                    aria-label="Pilih huruf video simulasi"
                    value={simLetter}
                    onChange={(e) => {
                      validatorRef.current.reset();
                      setSimLetter(e.target.value);
                    }}
                    className="rounded-lg bg-ink px-2 py-1 text-white"
                  >
                    {ALPHABET.map((l) => (
                      <option key={l} value={l}>
                        Huruf {l}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            )}

            <button
              type="button"
              onClick={handleStop}
              className="btn !min-h-[44px] !px-4 !text-sm ml-auto bg-coral text-white"
            >
              Matikan
            </button>
          </div>

          {justCompleted ? (
            <div
              role="status"
              aria-live="polite"
              className={`flex h-[60px] items-center justify-between gap-2 rounded-2xl border-2 border-ink px-3 text-ink ${
                mode === "webcam" ? "bg-mint" : "bg-sky"
              }`}
            >
              <span className="font-display text-sm font-bold sm:text-base">
                {mode === "webcam"
                  ? `Huruf ${letter} Berhasil Dikuasai! (+10 XP)`
                  : `Simulasi Huruf ${simLetter} Terbaca (Gunakan Kamera untuk Progres)`}
              </span>
              <span className="shrink-0 rounded-full bg-white px-3 py-1 text-xs font-bold">
                {mode === "webcam" ? "SELESAI" : "SIMULASI"}
              </span>
            </div>
          ) : (
            <div className="flex h-[60px] flex-col justify-center rounded-2xl border-2 border-white/20 bg-white/10 px-3">
              <div className="flex items-center justify-between gap-2 text-sm font-semibold">
                <span>
                  Target: Huruf <strong className="text-sun">{letter}</strong>
                  {isAlreadyLearned ? " (Sudah Dikuasai)" : ""}
                </span>
                <span className="shrink-0 tabular-nums">{holdPercent}% (Tahan 1,5 dtk)</span>
              </div>
              <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-black/50">
                <div
                  className="h-full rounded-full bg-mint transition-all duration-100"
                  style={{ width: `${holdPercent}%` }}
                />
              </div>
            </div>
          )}

          <div className="flex h-[52px] items-center gap-2 rounded-2xl border-2 border-white/20 bg-white/10 px-3">
            <p className="w-12 shrink-0 text-[10px] font-semibold uppercase leading-tight tracking-wider text-white/75">
              Prediksi AI
            </p>
            <div className="flex flex-1 justify-end gap-2" aria-live="off">
              {Array.from({ length: 3 }).map((_, i) => {
                const cand = holdState.topCandidates[i];
                const base =
                  "flex h-8 w-[4.75rem] shrink-0 items-center justify-center gap-1.5 rounded-lg border-2 tabular-nums";
                if (!cand) {
                  return (
                    <div
                      key={i}
                      className={`${base} border-white/10 bg-black/20 text-white/40`}
                      aria-hidden="true"
                    >
                      <span className="font-display text-base font-bold">–</span>
                    </div>
                  );
                }
                const pct = Math.round(cand.confidence * 100);
                const isTarget = cand.label === letter;
                return (
                  <div
                    key={i}
                    className={`${base} ${
                      isTarget
                        ? "border-mint bg-mint/20 text-white"
                        : i === 0
                          ? "border-sun/70 bg-white/10 text-white"
                          : "border-white/15 bg-black/30 text-white/80"
                    }`}
                  >
                    <span className="w-4 text-center font-display text-base font-bold">
                      {cand.label}
                    </span>
                    <span className="w-9 text-right text-xs font-semibold">{pct}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
