"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { speak } from "@/lib/audio";
import { WORD_ITEMS, WORD_MODEL_VERSION, type WordItem } from "@/lib/bisindo";
import { detectVideoHands, drawHandLandmarks, loadHandLandmarker } from "@/lib/cv/handLandmarkerClient";
import { subscribeModelProgress } from "@/lib/cv/mediapipeLoader";
import type { NormalizedLandmark2D } from "@/lib/cv/normalizeLandmarks";
import {
  detectVideoPose,
  drawPoseAnchors,
  extractPoseAnchors,
  loadPoseLandmarker,
} from "@/lib/cv/poseLandmarkerClient";
import {
  DEFAULT_WL_VARIANT,
  extractWlFrameFeatures,
  loadWordClassifier,
  loadWordVariants,
  type WlVariantSpec,
  type WordPredictionCandidate,
} from "@/lib/cv/wordClassifier";
import { useProgress } from "@/lib/storage/useProgress";
import type { WordDemoRow } from "@/lib/supabase/client";
import {
  Check,
  Maximize2,
  RefreshCw,
  Video,
  Volume2,
  X,
} from "react-feather";

type SlotMode = "idle" | "webcam" | "simulation";
type SlotStatus = "idle" | "loading" | "ready" | "error";

interface SimulationFrame {
  hands: number[][][];
  pose: number[][] | null;
}

type SimulationSequenceMap = Record<string, SimulationFrame[]>;

interface WordCameraSlotProps {
  word: WordItem;
  isLearned?: boolean;
  demo?: WordDemoRow;
  isLoadingDemo?: boolean;
  onSpeak?: () => void;
}

const RECORD_DURATION_MS = 2500;
const MIN_RECORD_FRAMES = 6;
const SIM_FRAME_INTERVAL_MS = 45;
const WEBCAM_MATCH_THRESHOLD = 0.5;
const SIM_MATCH_THRESHOLD = 0.65;

let cachedSequencesPromise: Promise<SimulationSequenceMap> | null = null;

function loadSimulationSequences(): Promise<SimulationSequenceMap> {
  if (!cachedSequencesPromise) {
    cachedSequencesPromise = fetch("/samples/bisindo/wl_sequences.json")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Gagal memuat sampel simulasi (${res.status}).`);
        }
        return res.json() as Promise<SimulationSequenceMap>;
      })
      .catch((err) => {
        cachedSequencesPromise = null;
        throw err;
      });
  }
  return cachedSequencesPromise;
}

function toLandmarkList(points: number[][]): NormalizedLandmark2D[] {
  return points.map((pt) => ({ x: pt[0] ?? 0, y: pt[1] ?? 0 }));
}

function isImageAnimationUrl(url: string, mimeType?: string): boolean {
  const mime = (mimeType ?? "").toLowerCase();
  const cleanUrl = url.toLowerCase().split("?")[0];
  return mime.startsWith("image/") || cleanUrl.endsWith(".webp") || cleanUrl.endsWith(".gif");
}

export default function WordCameraSlot({
  word,
  isLearned,
  demo,
  isLoadingDemo = false,
  onSpeak,
}: WordCameraSlotProps) {
  const { data, markWordLearned } = useProgress();
  const [mode, setMode] = useState<SlotMode>("idle");
  const [status, setStatus] = useState<SlotStatus>("idle");
  const [hasLiveStream, setHasLiveStream] = useState<boolean>(false);
  const [handLoadProgress, setHandLoadProgress] = useState<number>(0);
  const [poseLoadProgress, setPoseLoadProgress] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(data.settings.showLandmarks);
  const [simSampleWord, setSimSampleWord] = useState<string>(word.id);
  const [simFrameIndex, setSimFrameIndex] = useState<number>(0);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordProgress, setRecordProgress] = useState<number>(0);
  const [handDetected, setHandDetected] = useState<boolean>(false);
  const [poseDetected, setPoseDetected] = useState<boolean>(false);
  const [predictions, setPredictions] = useState<WordPredictionCandidate[]>([]);
  const [matched, setMatched] = useState<boolean>(false);
  const [variants, setVariants] = useState<WlVariantSpec[]>([DEFAULT_WL_VARIANT]);
  const [activeVariantId, setActiveVariantId] = useState<string>(DEFAULT_WL_VARIANT.id);
  const [isSwitchingVariant, setIsSwitchingVariant] = useState<boolean>(false);

  // Demo playback and zoom modal states
  const [sourceIndex, setSourceIndex] = useState<number>(0);
  const [isDemoExpanded, setIsDemoExpanded] = useState<boolean>(false);
  const [mediaFailed, setMediaFailed] = useState<boolean>(false);
  const [retryCount, setRetryCount] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const showLandmarksRef = useRef<boolean>(showLandmarks);
  const audioEnabledRef = useRef<boolean>(data.settings.audioEnabled);
  const isRecordingRef = useRef<boolean>(false);
  const recordStartRef = useRef<number>(0);
  const recordedFramesRef = useRef<Float32Array[]>([]);
  const classifierRef = useRef<Awaited<ReturnType<typeof loadWordClassifier>> | null>(null);
  const activeVariantIdRef = useRef<string>(activeVariantId);
  const hasUserPickedVariantRef = useRef<boolean>(false);
  const lastVideoTsRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  const celebratedRef = useRef<boolean>(false);
  const demoVideoRef = useRef<HTMLVideoElement | null>(null);
  const pipVideoRef = useRef<HTMLVideoElement | null>(null);

  const activeVariant =
    variants.find((v) => v.id === activeVariantId) ?? variants[0] ?? DEFAULT_WL_VARIANT;
  const {
    basePath: variantBasePath,
    format: variantFormat,
    featureMode: variantFeatureMode,
  } = activeVariant;

  useEffect(() => {
    activeVariantIdRef.current = activeVariantId;
  }, [activeVariantId]);

  useEffect(() => {
    let cancelled = false;
    void loadWordVariants().then((registry) => {
      if (cancelled) return;
      setVariants(registry.variants);
      if (!hasUserPickedVariantRef.current) {
        const nextDefault =
          registry.variants.find((v) => v.id === registry.defaultVariant)?.id ??
          registry.variants[0]?.id ??
          DEFAULT_WL_VARIANT.id;
        setActiveVariantId(nextDefault);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleVariantChange = useCallback((nextId: string) => {
    hasUserPickedVariantRef.current = true;
    setActiveVariantId(nextId);
  }, []);

  useEffect(() => {
    showLandmarksRef.current = showLandmarks;
  }, [showLandmarks]);

  useEffect(() => {
    audioEnabledRef.current = data.settings.audioEnabled;
  }, [data.settings.audioEnabled]);

  useEffect(() => {
    const unsubHand = subscribeModelProgress("hand", setHandLoadProgress);
    const unsubPose = subscribeModelProgress("pose", setPoseLoadProgress);
    return () => {
      unsubHand();
      unsubPose();
    };
  }, []);

  // Reset states when word changes
  useEffect(() => {
    setSourceIndex(0);
    setMediaFailed(false);
    setRetryCount(0);
    setIsDemoExpanded(false);
  }, [word.id, demo?.video_url, word.localVideoPath]);

  // Handle ESC key to close expand modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsDemoExpanded(false);
      }
    };
    if (isDemoExpanded) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isDemoExpanded]);

  const stopMediaStream = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setHasLiveStream(false);
  }, []);

  const resetPracticeState = useCallback(() => {
    isRecordingRef.current = false;
    recordedFramesRef.current = [];
    celebratedRef.current = false;
    setIsRecording(false);
    setRecordProgress(0);
    setPredictions([]);
    setMatched(false);
    setHandDetected(false);
    setPoseDetected(false);
    setErrorMsg(null);
  }, []);

  useEffect(() => {
    setSimSampleWord(word.id);
    setSimFrameIndex(0);
    resetPracticeState();
  }, [word.id, resetPracticeState]);

  const handlePredictions = useCallback(
    (
      candidates: WordPredictionCandidate[],
      threshold: number,
      sourceMode: "webcam" | "simulation"
    ) => {
      setPredictions(candidates.slice(0, 3));
      const top = candidates[0];
      const isMatch = Boolean(top && top.word === word.id && top.confidence >= threshold);
      setMatched(isMatch);
      if (isMatch && !celebratedRef.current && sourceMode === "webcam") {
        celebratedRef.current = true;
        markWordLearned(word.id, 15);
        speak(`Bagus! Gerakan kata ${word.title} berhasil dikenali.`, audioEnabledRef.current);
      }
    },
    [markWordLearned, word.id, word.title],
  );

  const startRecording = useCallback(() => {
    if (mode !== "webcam" || status !== "ready" || isRecordingRef.current) return;
    recordedFramesRef.current = [];
    recordStartRef.current = performance.now();
    celebratedRef.current = false;
    isRecordingRef.current = true;
    setMatched(false);
    setPredictions([]);
    setRecordProgress(0);
    setIsRecording(true);
  }, [mode, status]);

  const switchMode = useCallback(
    (nextMode: SlotMode) => {
      stopMediaStream();
      resetPracticeState();
      setSimFrameIndex(0);
      setStatus(nextMode === "idle" ? "idle" : "loading");
      setMode(nextMode);
    },
    [resetPracticeState, stopMediaStream],
  );

  useEffect(() => {
    if (mode !== "webcam") return;

    let cancelled = false;
    let rafId = 0;

    async function initWebcamLoop() {
      try {
        setStatus("loading");
        setErrorMsg(null);

        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Peramban ini belum mendukung akses kamera secara langsung.");
        }

        const modelsPromise = Promise.all([
          loadHandLandmarker(),
          loadPoseLandmarker(),
          (async () => {
            const registry = await loadWordVariants();
            const initialVariant =
              registry.variants.find((v) => v.id === activeVariantIdRef.current) ??
              registry.variants.find((v) => v.id === registry.defaultVariant) ??
              registry.variants[0] ??
              DEFAULT_WL_VARIANT;
            return loadWordClassifier(initialVariant.basePath, {
              format: initialVariant.format,
              featureMode: initialVariant.featureMode,
            });
          })(),
        ]);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });

        if (cancelled) {
          for (const track of stream.getTracks()) track.stop();
          return;
        }

        streamRef.current = stream;
        const videoEl = videoRef.current;
        if (!videoEl) {
          for (const track of stream.getTracks()) track.stop();
          return;
        }
        videoEl.srcObject = stream;
        await videoEl.play();
        if (cancelled) {
          for (const track of stream.getTracks()) track.stop();
          return;
        }
        setHasLiveStream(true);

        const [handLandmarker, poseLandmarker, classifier] = await modelsPromise;

        if (cancelled) return;

        classifierRef.current = classifier;

        lastVideoTimeRef.current = -1;
        setStatus("ready");

        const renderFrame = () => {
          if (cancelled) return;
          const video = videoRef.current;
          const canvas = canvasRef.current;

          if (video && canvas && video.readyState >= 2) {
            const vw = video.videoWidth || 640;
            const vh = video.videoHeight || 360;
            if (canvas.width !== vw || canvas.height !== vh) {
              canvas.width = vw;
              canvas.height = vh;
            }

            const now = performance.now();
            const currentTime = video.currentTime;

            if (currentTime !== lastVideoTimeRef.current) {
              lastVideoTimeRef.current = currentTime;
              const timestamp = now > lastVideoTsRef.current ? now : lastVideoTsRef.current + 1;
              lastVideoTsRef.current = timestamp;

              const handResult = detectVideoHands(handLandmarker, video, timestamp);
              const poseResult = detectVideoPose(poseLandmarker, video, timestamp);

              const hands = handResult.landmarks;
              const poseAnchors = extractPoseAnchors(poseResult.landmarks[0]);

              setHandDetected(hands.length > 0);
              setPoseDetected(poseAnchors !== null);

              const ctx = canvas.getContext("2d");
              if (ctx) {
                if (showLandmarksRef.current) {
                  drawHandLandmarks(ctx, hands, vw, vh, true);
                  drawPoseAnchors(ctx, poseAnchors, vw, vh, true);
                } else {
                  ctx.clearRect(0, 0, vw, vh);
                }
              }

              if (isRecordingRef.current) {
                const elapsed = performance.now() - recordStartRef.current;
                const progress = Math.min(1, elapsed / RECORD_DURATION_MS);
                setRecordProgress(progress);

                const frameFeatures = extractWlFrameFeatures(hands, poseAnchors);
                recordedFramesRef.current.push(frameFeatures);

                if (elapsed >= RECORD_DURATION_MS) {
                  isRecordingRef.current = false;
                  setIsRecording(false);
                  setRecordProgress(1);

                  const seq = recordedFramesRef.current;
                  if (seq.length >= MIN_RECORD_FRAMES && classifierRef.current) {
                    const ranked = classifierRef.current.predictFromFeatures(seq);
                    handlePredictions(ranked, WEBCAM_MATCH_THRESHOLD, "webcam");
                  } else {
                    setErrorMsg("Frame rekaman belum cukup. Pastikan tangan terlihat jelas lalu coba lagi.");
                  }
                }
              }
            }
          }

          rafId = requestAnimationFrame(renderFrame);
        };

        rafId = requestAnimationFrame(renderFrame);
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof Error
            ? err.message
            : "Gagal menyalakan kamera. Pastikan izin akses telah diberikan.";
        setErrorMsg(msg);
        setStatus("error");
      }
    }

    void initWebcamLoop();

    return () => {
      cancelled = true;
      if (rafId) cancelAnimationFrame(rafId);
      classifierRef.current = null;
      stopMediaStream();
    };
  }, [handlePredictions, mode, stopMediaStream]);

  useEffect(() => {
    if (mode !== "webcam" || !classifierRef.current) return;

    let cancelled = false;
    setIsSwitchingVariant(true);
    setErrorMsg(null);

    void loadWordClassifier(variantBasePath, {
      format: variantFormat,
      featureMode: variantFeatureMode,
    })
      .then((nextClassifier) => {
        if (cancelled) return;
        classifierRef.current = nextClassifier;
        setPredictions([]);
        setMatched(false);
        setIsSwitchingVariant(false);
      })
      .catch((err) => {
        if (cancelled) return;
        const msg =
          err instanceof Error
            ? err.message
            : "Gagal mengganti model WL-BISINDO.";
        setErrorMsg(msg);
        setIsSwitchingVariant(false);
      });

    return () => {
      cancelled = true;
    };
  }, [mode, variantBasePath, variantFormat, variantFeatureMode]);

  useEffect(() => {
    if (mode !== "simulation") return;

    let cancelled = false;
    let timerId = 0;

    async function runSimulationLoop() {
      try {
        setStatus("loading");
        setErrorMsg(null);

        const [classifier, sequenceMap] = await Promise.all([
          loadWordClassifier(variantBasePath, {
            format: variantFormat,
            featureMode: variantFeatureMode,
          }),
          loadSimulationSequences(),
        ]);

        if (cancelled) return;

        const frames = sequenceMap[simSampleWord];
        if (!frames || frames.length === 0) {
          throw new Error(`Sampel simulasi untuk kata "${simSampleWord}" tidak ditemukan.`);
        }

        const parsedFrames = frames.map((frame) => {
          const hands = frame.hands.map(toLandmarkList);
          const poseAnchors = frame.pose ? toLandmarkList(frame.pose) : null;
          return {
            hands,
            poseAnchors,
            features: extractWlFrameFeatures(hands, poseAnchors),
          };
        });

        setStatus("ready");
        let idx = 0;
        let inferredOnce = false;

        const tick = () => {
          if (cancelled) return;
          const currentFrame = parsedFrames[idx];
          const canvas = canvasRef.current;

          setSimFrameIndex(idx);
          setHandDetected(currentFrame.hands.length > 0);
          setPoseDetected(currentFrame.poseAnchors !== null);

          if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) {
              if (showLandmarksRef.current) {
                drawHandLandmarks(ctx, currentFrame.hands, canvas.width, canvas.height, false);
                drawPoseAnchors(ctx, currentFrame.poseAnchors, canvas.width, canvas.height, false);
              } else {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
              }
            }
          }

          if (!inferredOnce && idx === parsedFrames.length - 1) {
            inferredOnce = true;
            const featureSequence = parsedFrames.map((item) => item.features);
            const ranked = classifier.predictFromFeatures(featureSequence);
            handlePredictions(ranked, SIM_MATCH_THRESHOLD, "simulation");
          }

          idx = (idx + 1) % parsedFrames.length;
          timerId = window.setTimeout(tick, SIM_FRAME_INTERVAL_MS);
        };

        tick();
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof Error
            ? err.message
            : "Gagal menjalankan simulasi rangka WL-BISINDO.";
        setErrorMsg(msg);
        setStatus("error");
      }
    }

    void runSimulationLoop();

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [
    mode,
    simSampleWord,
    handlePredictions,
    variantBasePath,
    variantFormat,
    variantFeatureMode,
  ]);

  const alreadyLearned = isLearned ?? (data.bisindo.learnedWords ?? []).includes(word.id);

  // Robust Media Resolution: Supabase URL first, local /videos/bisindo/<id>.mp4 fallback
  const localFallbackUrl = word.localVideoPath
    ? `${word.localVideoPath}?v=${encodeURIComponent(WORD_MODEL_VERSION)}`
    : `/videos/bisindo/${word.id}.mp4?v=${encodeURIComponent(WORD_MODEL_VERSION)}`;

  const candidates = [demo?.video_url?.trim() || null, localFallbackUrl].filter(
    (url, idx, arr): url is string => Boolean(url) && arr.indexOf(url) === idx,
  );

  const activeCandidate = candidates[sourceIndex] ?? null;
  const isImgAnim = activeCandidate
    ? isImageAnimationUrl(activeCandidate, sourceIndex === 0 ? demo?.mime_type : undefined)
    : false;

  const mediaUrl = activeCandidate
    ? retryCount > 0
      ? `${activeCandidate}${activeCandidate.includes("?") ? "&" : "?"}retry=${retryCount}`
      : activeCandidate
    : null;

  const showLoadingSkeleton = isLoadingDemo && !activeCandidate;

  const handleMediaError = () => {
    if (sourceIndex + 1 < candidates.length) {
      setSourceIndex((prev) => prev + 1);
      setRetryCount(0);
    } else {
      setMediaFailed(true);
    }
  };

  // Zoom/Expand Modal Preview
  const renderExpandModal = () => (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Detail gerakan kata ${word.title}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/75 p-4 backdrop-blur-sm animate-pop"
      onClick={() => setIsDemoExpanded(false)}
    >
      <div
        className="card w-full max-w-lg space-y-3 bg-white p-4 sm:p-5 shadow-[8px_8px_0px_#1B1B3A]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b-2 border-ink/10 pb-2">
          <div className="flex items-center gap-2">
            <span className="rounded-xl border-2 border-ink bg-grape px-2.5 py-1 font-display text-base sm:text-lg font-bold text-white shadow-[2px_2px_0px_#1B1B3A]">
              {word.title}
            </span>
            <span className="rounded-full border-2 border-ink bg-mint px-2 py-0.5 text-xs font-bold text-ink">
              {word.hands} Tangan
            </span>
            {alreadyLearned && (
              <span className="rounded-full border-2 border-ink bg-sun px-2 py-0.5 text-xs font-bold text-ink">
                Dikuasai <Check className="inline h-3 w-3" />
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => setIsDemoExpanded(false)}
            className="btn !min-h-[34px] bg-coral px-3 py-1 text-xs font-bold text-white shadow-[2px_2px_0px_#1B1B3A]"
          >
            <X className="mr-1 inline h-3.5 w-3.5" />
            Tutup Preview
          </button>
        </div>

        <div className="relative aspect-video w-full overflow-hidden rounded-xl border-2 border-ink bg-ink">
          {mediaUrl && !mediaFailed ? (
            isImgAnim ? (
              <img
                key={mediaUrl}
                src={mediaUrl}
                alt={`Animasi gerakan ${word.title}`}
                className="h-full w-full object-contain"
                onError={handleMediaError}
              />
            ) : (
              <video
                key={mediaUrl}
                src={mediaUrl}
                autoPlay
                loop
                muted
                controls
                playsInline
                className="h-full w-full object-contain"
                onError={handleMediaError}
              >
                <source src={mediaUrl} type="video/mp4" />
              </video>
            )
          ) : (
            <div className="flex h-full w-full items-center justify-center p-4 text-center text-xs text-white/80">
              {word.guide || "Video contoh gerakan belum tersedia."}
            </div>
          )}
        </div>

        {word.guide && (
          <div className="rounded-xl border border-ink/20 bg-surface-secondary p-3 text-xs sm:text-sm text-ink/85 leading-relaxed">
            <strong className="text-ink">Panduan Gerakan:</strong> {word.guide}
          </div>
        )}

        <div className="flex justify-end pt-1">
          <button
            type="button"
            onClick={() => setIsDemoExpanded(false)}
            className="btn w-full sm:w-auto bg-sun px-4 py-2 text-xs font-bold text-ink shadow-[2px_2px_0px_#1B1B3A]"
          >
            {mode === "idle" ? "Mengerti, Tutup Preview" : "Lanjut Latihan Kamera"}
          </button>
        </div>
      </div>
    </div>
  );

  const variantSwitcherNode =
    variants.length > 1 ? (
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border-2 border-ink bg-paper px-2.5 py-1.5 text-xs">
        <div className="flex flex-wrap items-center gap-2 min-w-0 flex-1">
          <label
            htmlFor="wl-variant-select"
            className="font-bold text-ink whitespace-nowrap"
          >
            Switch Model AI (A/B Test):
          </label>
          <select
            id="wl-variant-select"
            value={activeVariant.id}
            onChange={(e) => handleVariantChange(e.target.value)}
            disabled={isSwitchingVariant}
            className="min-w-0 max-w-full rounded-lg border-2 border-ink bg-white px-2 py-1 text-xs font-semibold text-ink"
          >
            {variants.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </div>
        {isSwitchingVariant && (
          <span className="rounded-full border border-ink bg-sun px-2 py-0.5 text-[10px] font-bold text-ink whitespace-nowrap">
            Memuat...
          </span>
        )}
      </div>
    ) : null;

  // MODE IDLE: Full Demo Video & Practice Launcher
  if (mode === "idle") {
    return (
      <div className="card space-y-3 bg-white p-3.5 sm:p-4 animate-pop">
        {/* Word Info Header */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b-2 border-ink/10 pb-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-xl border-2 border-ink bg-grape px-2.5 py-1 font-display text-lg sm:text-xl font-bold text-white shadow-[2px_2px_0px_#1B1B3A]">
              {word.title}
            </span>
            <span className="rounded-full border-2 border-ink bg-mint px-2.5 py-0.5 text-xs font-bold text-ink">
              {word.hands} Tangan
            </span>
            {alreadyLearned && (
              <span className="rounded-full border-2 border-ink bg-sun px-2.5 py-0.5 text-xs font-bold text-ink">
                Dikuasai <Check className="inline h-3 w-3" />
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {onSpeak && (
              <button
                type="button"
                aria-label={`Dengarkan pengucapan kata ${word.title}`}
                onClick={onSpeak}
                className="btn !min-h-[36px] !min-w-[36px] bg-sun !p-1.5 text-ink shadow-[2px_2px_0px_#1B1B3A]"
                title="Dengarkan suara kata"
              >
                <Volume2 className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Demo Video Container */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <p className="text-xs sm:text-sm font-bold text-ink">
              Ikuti gerakan seperti ini:
            </p>
            {mediaUrl && !mediaFailed && !isImgAnim && (
              <button
                type="button"
                onClick={() => setIsDemoExpanded(true)}
                className="inline-flex items-center gap-1 rounded-full border border-ink bg-sky/30 px-2 py-0.5 text-[11px] font-bold text-ink hover:bg-sky"
              >
                <Maximize2 className="h-3 w-3" />
                Perbesar
              </button>
            )}
          </div>

          <div className="relative aspect-video max-h-[28dvh] sm:max-h-[32dvh] min-h-[170px] w-full overflow-hidden rounded-2xl border-2 border-ink bg-ink">
            {showLoadingSkeleton ? (
              <div
                className="flex h-full w-full animate-pulse items-center justify-center p-4 text-xs font-semibold text-white/70"
                role="status"
              >
                Memuat contoh gerakan {word.title}...
              </div>
            ) : mediaUrl && !mediaFailed ? (
              isImgAnim ? (
                <img
                  key={mediaUrl}
                  src={mediaUrl}
                  alt={`Animasi contoh gerakan BISINDO kata ${word.title}`}
                  onError={handleMediaError}
                  className="h-full w-full object-contain"
                />
              ) : (
                <video
                  ref={demoVideoRef}
                  key={mediaUrl}
                  src={mediaUrl}
                  autoPlay
                  loop
                  muted
                  playsInline
                  preload="auto"
                  onClick={() => setIsDemoExpanded(true)}
                  aria-label={`Video contoh gerakan kata ${word.title}`}
                  onError={handleMediaError}
                  className="h-full w-full cursor-pointer object-contain"
                  title="Klik untuk memperbesar contoh gerakan"
                >
                  <source src={mediaUrl} type="video/mp4" />
                </video>
              )
            ) : (
              <div className="flex h-full w-full flex-col items-center justify-center p-4 text-center text-xs text-white/80">
                <p className="font-semibold text-white">
                  {mediaFailed
                    ? `Gagal memuat video contoh ${word.title}.`
                    : `Video contoh ${word.title} belum tersedia.`}
                </p>
                {mediaFailed && (
                  <button
                    type="button"
                    onClick={() => {
                      setSourceIndex(0);
                      setMediaFailed(false);
                      setRetryCount((prev) => prev + 1);
                    }}
                    className="mt-2 inline-flex items-center gap-1.5 rounded-full border-2 border-ink bg-white px-3 py-1 text-xs font-bold text-ink hover:bg-sun"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    Muat ulang
                  </button>
                )}
              </div>
            )}
          </div>

          {word.guide && (
            <p className="text-xs leading-snug text-ink/80 bg-surface-secondary p-2 rounded-xl border border-ink/15">
              <strong className="text-ink">Panduan:</strong> {word.guide}
            </p>
          )}
        </div>

        {variantSwitcherNode}

        {/* Action Buttons to Start Practice */}
        <div className="flex flex-col sm:flex-row items-center gap-2 pt-0.5">
          <button
            type="button"
            onClick={() => switchMode("webcam")}
            className="btn w-full flex-1 !min-h-[40px] bg-sun text-ink text-xs sm:text-sm font-bold py-2 shadow-[3px_3px_0px_#1B1B3A]"
          >
            <Video className="mr-1.5 inline h-4 w-4" />
            Gunakan Kamera Langsung
          </button>
          <button
            type="button"
            onClick={() => switchMode("simulation")}
            className="btn w-full sm:w-auto !min-h-[40px] bg-white text-ink text-xs sm:text-sm font-semibold py-2"
          >
            Mode Simulasi Rangka
          </button>
        </div>

        {isDemoExpanded && renderExpandModal()}
      </div>
    );
  }

  // ACTIVE MODE: Layout according to user's exact sketch
  return (
    <div className="card space-y-2.5 sm:space-y-3 bg-white p-3 sm:p-4">
      {/* 1. TOP BAR: [ ancaman ] [ tutup ] [ speaker ] */}
      <div className="flex items-center justify-between gap-2 border-b-2 border-ink/10 pb-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="rounded-xl border-2 border-ink bg-grape px-3 py-1 font-display text-base sm:text-lg font-bold text-white shadow-[2px_2px_0px_#1B1B3A] truncate">
            {word.title}
          </span>
          <span className="rounded-full border-2 border-ink bg-mint px-2.5 py-0.5 text-xs font-bold text-ink whitespace-nowrap">
            {word.hands} Tangan
          </span>
          {alreadyLearned && (
            <span className="hidden sm:inline-flex items-center rounded-full border-2 border-ink bg-sun px-2 py-0.5 text-xs font-bold text-ink whitespace-nowrap">
              Dikuasai <Check className="ml-1 inline h-3 w-3" />
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Toggle Rangka */}
          <button
            type="button"
            onClick={() => setShowLandmarks((prev) => !prev)}
            className="btn !min-h-[34px] bg-paper px-2.5 py-1 text-xs text-ink shadow-[2px_2px_0px_#1B1B3A]"
            title="Tampilkan / sembunyikan titik rangka"
          >
            {showLandmarks ? "Rangka: On" : "Rangka: Off"}
          </button>

          {/* Tombol Tutup */}
          <button
            type="button"
            onClick={() => switchMode("idle")}
            className="btn !min-h-[34px] bg-coral px-3.5 py-1 text-xs font-bold text-white shadow-[2px_2px_0px_#1B1B3A]"
            title="Tutup kamera dan kembali ke panduan"
          >
            Tutup
          </button>

          {/* Tombol Suara */}
          {onSpeak && (
            <button
              type="button"
              aria-label={`Dengarkan pengucapan kata ${word.title}`}
              onClick={onSpeak}
              className="btn !min-h-[34px] !min-w-[34px] bg-sun !p-1.5 text-ink shadow-[2px_2px_0px_#1B1B3A]"
              title="Dengarkan suara kata"
            >
              <Volume2 className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {variantSwitcherNode}

      {mode === "simulation" && (
        <div className="flex flex-wrap items-center gap-2 rounded-xl border border-ink/40 bg-paper px-3 py-1.5 text-xs">
          <label htmlFor="wl-sim-select" className="font-bold text-ink">
            Uji sampel gerakan:
          </label>
          <select
            id="wl-sim-select"
            value={simSampleWord}
            onChange={(e) => {
              setSimSampleWord(e.target.value);
              setSimFrameIndex(0);
            }}
            className="rounded-lg border-2 border-ink bg-white px-2 py-1 text-xs font-semibold text-ink"
          >
            {WORD_ITEMS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* 2. MAIN UPPER BOX: WEBCAM FEED WITH OVERLAPPING RECORD BUTTON */}
      <div className="relative overflow-hidden rounded-2xl border-2 border-ink bg-ink shadow-[3px_3px_0px_#1B1B3A]">
        {/* OVERLAPPING RECORD BUTTON (Top-Left inside Camera) */}
        <button
          type="button"
          onClick={startRecording}
          disabled={isRecording || status !== "ready"}
          className={`btn absolute top-3 left-3 z-20 !min-h-[36px] px-3.5 py-1.5 text-xs font-bold border-2 border-ink shadow-[2px_2px_0px_#1B1B3A] transition-all ${
            isRecording
              ? "bg-coral text-white scale-105 shadow-none"
              : "bg-sun text-ink hover:bg-sun-light active:scale-95"
          }`}
        >
          <span
            className={`mr-1.5 inline-block h-2.5 w-2.5 rounded-full ${
              isRecording ? "bg-white animate-ping" : "bg-coral"
            }`}
          />
          {isRecording ? (
            <span>Merekam ({Math.max(0, 2.5 - recordProgress * 2.5).toFixed(1)}s)</span>
          ) : predictions.length > 0 ? (
            <span>Ulangi Rekam</span>
          ) : (
            <span>Rekam</span>
          )}
        </button>

        {/* WEBCAM VIDEO / CANVAS CONTAINER */}
        <div className="relative aspect-video max-h-[26dvh] sm:max-h-[30dvh] min-h-[185px] w-full overflow-hidden bg-ink">
          {mode === "webcam" && (
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              className="h-full w-full -scale-x-100 object-cover"
            />
          )}
          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />

          {/* LOADING OVERLAYS */}
          {status === "loading" && mode === "webcam" && hasLiveStream && (
            <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-col items-center gap-1.5 rounded-xl border border-white/25 bg-ink/85 px-3 py-2 text-center backdrop-blur-sm">
              <p className="text-xs font-semibold text-white">
                Menyiapkan AI tangan &amp; pose... {Math.round(((handLoadProgress + poseLoadProgress) / 2) * 100)}%
              </p>
              <div className="h-1.5 w-full max-w-[200px] overflow-hidden rounded-full bg-white/20">
                <div
                  className="h-full rounded-full bg-sun transition-all duration-150"
                  style={{ width: `${Math.round(((handLoadProgress + poseLoadProgress) / 2) * 100)}%` }}
                />
              </div>
            </div>
          )}

          {status === "loading" && !(mode === "webcam" && hasLiveStream) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-ink/85 p-4 text-center text-white">
              <p className="font-semibold text-xs sm:text-sm">
                {mode === "webcam"
                  ? `Memuat kamera, pelacak tangan, pose & model kata... ${Math.round(((handLoadProgress + poseLoadProgress) / 2) * 100)}%`
                  : "Memuat model kata BISINDO dan animasi rangka..."}
              </p>
              {mode === "webcam" && (
                <div className="h-2 w-48 overflow-hidden rounded-full bg-white/20">
                  <div
                    className="h-full rounded-full bg-sun transition-all duration-150"
                    style={{ width: `${Math.round(((handLoadProgress + poseLoadProgress) / 2) * 100)}%` }}
                  />
                </div>
              )}
            </div>
          )}

          {mode === "simulation" && status === "ready" && !showLandmarks && (
            <div className="absolute inset-0 flex items-center justify-center bg-ink/75 p-4 text-center text-white">
              <p className="text-sm font-semibold">
                Rangka sedang disembunyikan. Tekan Rangka: On untuk melihat animasi gerakan.
              </p>
            </div>
          )}
        </div>
      </div>

      {errorMsg && (
        <div className="rounded-xl border-2 border-ink bg-coral p-2.5 text-xs sm:text-sm font-semibold text-white">
          {errorMsg}
        </div>
      )}

      {/* 3. BOTTOM SECTION: SIDE-BY-SIDE [ PREDIKSI TERATAS ] & [ VIDEO DEMO ] */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {/* BOTTOM LEFT: [ PREDIKSI TERATAS ] */}
        <div className="rounded-2xl border-2 border-ink bg-white p-3 shadow-[3px_3px_0px_#1B1B3A] flex flex-col justify-between overflow-hidden min-h-[145px] max-h-[175px]">
          <div className="flex items-center justify-between border-b border-ink/15 pb-1">
            <span className="text-[11px] font-bold text-ink/75 uppercase tracking-wider">
              PREDIKSI TERATAS
            </span>
            {matched ? (
              <span className="rounded-full bg-mint px-2 py-0.5 text-[10px] font-bold text-ink border border-ink">
                ✓ Cocok (+15 XP)
              </span>
            ) : (
              <span className="text-[10px] font-semibold text-ink/50 truncate max-w-[140px]">
                {isSwitchingVariant ? "Memuat model..." : activeVariant.id}
              </span>
            )}
          </div>

          {/* Prediksi Rows (Always structured with 3 rows) */}
          <div className="space-y-1.5 py-1">
            {(predictions.length > 0
              ? predictions.slice(0, 3).map((cand) => ({
                  label: cand.label,
                  pct: Math.round(cand.confidence * 100),
                  isTarget: cand.word === word.id,
                }))
              : [
                  { label: word.title, pct: 0, isTarget: true },
                  { label: "-", pct: 0, isTarget: false },
                  { label: "-", pct: 0, isTarget: false },
                ]
            ).map((cand, idx) => (
              <div key={idx} className="space-y-0.5">
                <div className="flex items-center justify-between text-xs font-bold text-ink">
                  <span className={cand.isTarget ? "text-grape font-extrabold" : "text-ink"}>
                    {cand.label} {cand.isTarget && cand.pct > 0 ? " (Target)" : ""}
                  </span>
                  <span className="font-mono text-xs">{cand.pct}%</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full border border-ink bg-paper">
                  <div
                    className={`h-full transition-all duration-300 ${cand.isTarget ? "bg-mint" : "bg-sky"}`}
                    style={{ width: `${cand.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* BOTTOM RIGHT: [ VIDEO DEMO ] (Tanpa Opsi Zoom) */}
        <div className="rounded-2xl border-2 border-ink bg-ink overflow-hidden shadow-[3px_3px_0px_#1B1B3A] flex flex-col min-h-[145px] max-h-[175px]">
          {/* Header Video Demo */}
          <div className="flex items-center justify-between border-b-2 border-ink bg-white px-3 py-1.5 shrink-0">
            <span className="flex items-center gap-1.5 text-xs font-bold text-ink truncate">
              <span className="h-2 w-2 shrink-0 rounded-full bg-mint animate-pulse" />
              Video Demo: {word.title}
            </span>
            <span className="text-[10px] font-semibold text-ink/60">
              BISINDO
            </span>
          </div>

          {/* Video Player Box */}
          <div className="relative flex-1 w-full bg-black flex items-center justify-center overflow-hidden">
            {showLoadingSkeleton ? (
              <div className="flex h-full w-full items-center justify-center p-2 text-center text-xs text-white/70 animate-pulse">
                Memuat video contoh...
              </div>
            ) : mediaUrl && !mediaFailed ? (
              isImgAnim ? (
                <img
                  key={mediaUrl}
                  src={mediaUrl}
                  alt={`Animasi contoh ${word.title}`}
                  className="h-full w-full object-contain"
                  onError={handleMediaError}
                />
              ) : (
                <video
                  ref={pipVideoRef}
                  key={mediaUrl}
                  src={mediaUrl}
                  autoPlay
                  loop
                  muted
                  playsInline
                  preload="auto"
                  className="h-full w-full object-contain"
                  onError={handleMediaError}
                >
                  <source src={mediaUrl} type="video/mp4" />
                </video>
              )
            ) : (
              <div className="flex h-full w-full items-center justify-center p-2 text-center text-[10px] text-white/80">
                {word.guide || "Video demo belum tersedia"}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* EXPAND MODAL OVERLAY */}
      {isDemoExpanded && renderExpandModal()}
    </div>
  );
}
