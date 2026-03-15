'use client';

import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

type Point = { x: number; y: number };

function calculateAngle(a: Point, b: Point, c: Point): number {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) {
        angle = 360.0 - angle;
    }
    return angle;
}

const EXERCISES = [
    { id: "sit_and_stand", name: "Sit and Stand", targetReps: 5, prepTime: 5, testTime: 45, feedback_idle: "Sit on a chair, facing slightly sideways. Get ready." },
    { id: "shoulder_abduction", name: "Shoulder Abduction", targetReps: 5, prepTime: 5, testTime: 30, feedback_idle: "Face front. Raise arms to the sides." },
    { id: "standing_march", name: "Standing March", targetReps: 5, prepTime: 5, testTime: 40, feedback_idle: "Face completely sideways. March knees up alternating." }
];

export default function MobilityGame() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState("Loading AI Model...");
  const [feedback, setFeedback] = useState("Get Ready!");

  const [currentTestIdx, setCurrentTestIdx] = useState(0);
  const [reps, setReps] = useState(0);

  const [phaseUI, setPhaseUI] = useState<"PREP" | "TEST" | "DONE">("PREP");
  const [timeLeftUI, setTimeLeftUI] = useState<number>(0);

  const phaseRef = useRef<"PREP" | "TEST" | "DONE">("PREP");
  const phaseStartMs = useRef<number>(0);

  const isPlaying = useRef(false);
  const isEnded = useRef(false);
  const animationRef = useRef<number>(0);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);

  // FSM State refs to avoid excessive re-renders
  const fsmState = useRef<any>({
      main: "IDLE",
      subStateL: "IDLE",
      subStateR: "IDLE",
      halfReps: 0
  });
  
  const currentReps = useRef(0);
  const currentTestIndexRef = useRef(0);

  const initMediaPipe = async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      const landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU"
        },
        outputSegmentationMasks: false,
        runningMode: "VIDEO",
        numPoses: 1
      });
      poseLandmarkerRef.current = landmarker;
      startCamera();
    } catch (err) {
      console.error(err);
      setStatus("Failed to load Pose AI.");
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          isPlaying.current = true;
          setStatus("Active");
          setFeedback(EXERCISES[0].feedback_idle);
          phaseRef.current = "PREP";
          setPhaseUI("PREP");
          phaseStartMs.current = performance.now();
          predict();
        };
      }
    } catch (err) {
      setStatus("Camera Access Denied");
    }
  };

  const drawSkeletons = (ctx: CanvasRenderingContext2D, landmarks: any, width: number, height: number) => {
    ctx.clearRect(0, 0, width, height);

    const connections = PoseLandmarker.POSE_CONNECTIONS;
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 4;
    connections.forEach((conn) => {
        const start = landmarks[conn.start];
        const end = landmarks[conn.end];

        if (start.visibility > 0.5 && end.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(start.x * width, start.y * height);
            ctx.lineTo(end.x * width, end.y * height);
            ctx.stroke();
        }
    });

    ctx.fillStyle = "#ffffff";
    landmarks.forEach((p: any, idx: number) => {
        if (p.visibility > 0.5 && idx > 10) {
            ctx.beginPath();
            ctx.arc(p.x * width, p.y * height, 6, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
  };

  const nextTest = () => {
    fsmState.current = { main: "IDLE", subStateL: "IDLE", subStateR: "IDLE", halfReps: 0 };
    currentReps.current = 0;

    if (currentTestIndexRef.current + 1 < EXERCISES.length) {
        currentTestIndexRef.current += 1;
        setCurrentTestIdx(currentTestIndexRef.current);
        setReps(0);
        
        phaseRef.current = "PREP";
        setPhaseUI("PREP");
        phaseStartMs.current = performance.now();
        setFeedback(EXERCISES[currentTestIndexRef.current].feedback_idle);      
    } else {
        phaseRef.current = "DONE";
        setPhaseUI("DONE");
        endGame();
    }
  };

  const processAngles = (landmarks: any[]) => {
      const idx = currentTestIndexRef.current;
      if (idx >= EXERCISES.length) return;
      const ex = EXERCISES[idx];

      if (ex.id === "sit_and_stand") {
          // Requires user to face slightly sideways so the camera sees 12 (Shoulder), 24 (Hip), 26 (Knee), 28 (Ankle)
          // Using right side landmarks
          if (landmarks[24].visibility > 0.4 && landmarks[26].visibility > 0.4 && landmarks[28].visibility > 0.4 && landmarks[12].visibility > 0.4) {
              const hip_angle = calculateAngle(landmarks[12], landmarks[24], landmarks[26]);
              const knee_angle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
              
              if (fsmState.current.main === "IDLE" && knee_angle < 120 && hip_angle < 130) {
                  fsmState.current.main = "STANDING_UP";
                  setFeedback("Good resting pos! Now stand up fully.");
              } else if (fsmState.current.main === "STANDING_UP" && knee_angle > 150 && hip_angle > 150) {
                  fsmState.current.main = "STANDING";
                  setFeedback("Great! Now sit back down.");
              } else if (fsmState.current.main === "STANDING" && knee_angle < 120 && hip_angle < 130) {
                  fsmState.current.main = "IDLE";
                  currentReps.current += 1;
                  setReps(currentReps.current);
                  setFeedback(`Good rep! [${currentReps.current}] Stand up again.`);      
                  if (currentReps.current >= ex.targetReps) nextTest();
              }
          }
      } else if (ex.id === "shoulder_abduction") {
          if (landmarks[23].visibility > 0.4 && landmarks[11].visibility > 0.4 && landmarks[13].visibility > 0.4) {
             const angle = calculateAngle(landmarks[23], landmarks[11], landmarks[13]);
             
             if (fsmState.current.main === "IDLE" && angle > 40) {
                 fsmState.current.main = "RAISING";
                 setFeedback("Keep raising! Reach for the sky.");
             } else if (fsmState.current.main === "RAISING" && angle > 150) {
                 fsmState.current.main = "PEAK";
                 setFeedback("Excellent height! Lower arm slowly.");
             } else if (fsmState.current.main === "PEAK" && angle < 40) {       
                 fsmState.current.main = "IDLE";
                 currentReps.current += 1;
                 setReps(currentReps.current);
                 setFeedback(`Great form! [${currentReps.current}] Raise again.`); 
                 if (currentReps.current >= ex.targetReps) nextTest();
             }
          }
      } else if (ex.id === "standing_march") {
          let side_completed = false;

          // Right leg March FSM (12=Shoulder, 24=Hip, 26=Knee)
          if (landmarks[12].visibility > 0.4 && landmarks[24].visibility > 0.4 && landmarks[26].visibility > 0.4) {
              const right_angle = calculateAngle(landmarks[12], landmarks[24], landmarks[26]);
              if (fsmState.current.subStateR === "IDLE" && right_angle < 130) {
                  fsmState.current.subStateR = "LIFTING";
              } else if (fsmState.current.subStateR === "LIFTING" && right_angle < 100) {
                  fsmState.current.subStateR = "PEAK";
              } else if (fsmState.current.subStateR === "PEAK" && right_angle > 150) {
                  fsmState.current.subStateR = "IDLE";
                  fsmState.current.halfReps += 1;
                  side_completed = true;
              }
          }

          // Left leg March FSM (11=Shoulder, 23=Hip, 25=Knee)
          if (landmarks[11].visibility > 0.4 && landmarks[23].visibility > 0.4 && landmarks[25].visibility > 0.4) {
              const left_angle = calculateAngle(landmarks[11], landmarks[23], landmarks[25]);
              if (fsmState.current.subStateL === "IDLE" && left_angle < 130) {
                  fsmState.current.subStateL = "LIFTING";
              } else if (fsmState.current.subStateL === "LIFTING" && left_angle < 100) {
                  fsmState.current.subStateL = "PEAK";
              } else if (fsmState.current.subStateL === "PEAK" && left_angle > 150) {
                  fsmState.current.subStateL = "IDLE";
                  fsmState.current.halfReps += 1;
                  side_completed = true;
              }
          }

          // Compute total reps (2 halves = 1 full rep)
          if (side_completed) {
              let totalCompleted = Math.floor(fsmState.current.halfReps / 2);
              if (totalCompleted > currentReps.current) {
                  currentReps.current = totalCompleted;
                  setReps(currentReps.current);
                  setFeedback(`Solid march! [${currentReps.current}] Next rep.`);
                  if (currentReps.current >= ex.targetReps) nextTest();
              } else {
                  setFeedback(`Good! Now alternate to the other leg.`);
              }
          }
      }
  };

  const predict = () => {
    if (!isPlaying.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    const now = performance.now();
    const ex = EXERCISES[currentTestIndexRef.current];

    // Core Timer & State Manager
    if (phaseRef.current !== "DONE" && ex) {
        const elapsedSeconds = (now - phaseStartMs.current) / 1000;
        const totalDuration = phaseRef.current === "PREP" ? ex.prepTime : ex.testTime;
        let remaining = Math.max(0, totalDuration - elapsedSeconds);
        
        setTimeLeftUI(Math.ceil(remaining));

        if (remaining === 0) {
            if (phaseRef.current === "PREP") {
                phaseRef.current = "TEST";
                setPhaseUI("TEST");
                phaseStartMs.current = now;
                setFeedback("Go! Start the exercise.");
            } else if (phaseRef.current === "TEST") {
                // Testing phase ran out of time -> Force go to next test 
                nextTest();
            }
        }
    }

    if (video && canvas && video.readyState >= 2 && poseLandmarkerRef.current) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const results = poseLandmarkerRef.current.detectForVideo(video, now);
        
        if (results.landmarks && results.landmarks[0]) {
            drawSkeletons(ctx, results.landmarks[0], canvas.width, canvas.height);
            // Only process reps and exercise FSM if we are in the active TEST phase
            if (phaseRef.current === "TEST" && !isEnded.current) {
                processAngles(results.landmarks[0]);
            }
        } else {
            ctx.clearRect(0,0, canvas.width, canvas.height);
        }
      }
    }

    if (!isEnded.current) {
      animationRef.current = requestAnimationFrame(predict);
    }
  };

  const endGame = async () => {
    if (isEnded.current) return;
    isEnded.current = true;
    isPlaying.current = false;
    if (animationRef.current) cancelAnimationFrame(animationRef.current);

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }

    setStatus("Test Complete! Sending Results...");

    const tg = (window as any).Telegram?.WebApp;
    let uid = tg?.initDataUnsafe?.user?.id;
    if (!uid) {
      const p = new URLSearchParams(window.location.search);
      uid = p.get('uid');
    }

    let totalReps = EXERCISES.reduce((sum, ex) => sum + ex.targetReps, 0);

    if (uid) {
        try {
          await fetch('/api/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              telegram_id: String(uid),
              game_type: 'mobility_score',
              score: totalReps,
              metrics: {"tests_completed": EXERCISES.length}
            })
          });
        } catch (err) {
            console.error("Failed to POST score", err);
        }
    }

    if (tg) {
      setTimeout(() => tg.close(), 1500);
    } else {
      setStatus("Done! Please close window.");
    }
  };

  useEffect(() => {
    initMediaPipe();
    return () => {
      isPlaying.current = false;
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const curEx = EXERCISES[currentTestIdx] || EXERCISES[0];

  return (
    <div className="flex-1 flex flex-col items-center max-h-full bg-gray-900 relative">
      
      {/* Exercise Information Overlay */}
      <div className="absolute top-4 left-4 z-10 font-mono flex flex-col gap-2">
        <div className="bg-black/60 rounded-xl p-2 text-white text-sm shadow-xl font-bold">
          {curEx.name}
        </div>
        
        {!isEnded.current && (
          <div className={`px-3 py-2 rounded-xl text-md font-bold shadow-xl border-2 transition-colors duration-500 flex items-center justify-center
            ${phaseUI === 'PREP' 
                ? 'bg-amber-500/90 border-amber-300 text-white animate-pulse' 
                : 'bg-red-500/90 border-red-300 text-white'}
          `}>
             {phaseUI === 'PREP' ? 'GET READY:' : 'TESTING:'} {timeLeftUI}s
          </div>
        )}
      </div>

      {!isEnded.current && (
        <div className="absolute top-4 right-4 z-10 bg-black/60 rounded-xl p-2 text-white font-mono text-sm shadow-xl font-bold">
          Reps: {reps} / {curEx.targetReps}
        </div>
      )}

      {/* Video element - hidden behind canvas visually, but handles stream */}
      <video
        ref={videoRef}
        className="absolute inset-0 w-full h-full object-cover -scale-x-100"
        autoPlay
        playsInline
        muted
      />
      {/* Canvas for drawing the tracking lines */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full object-cover -scale-x-100 z-0 pointer-events-none"
      />

      {/* Overlay Status Bar */}
      <div className="absolute bottom-10 left-4 right-4 z-20">
        <div className="bg-emerald-600/90 backdrop-blur-md rounded-2xl p-4 shadow-2xl text-center border-2 border-emerald-400">
           <h2 className="text-xl font-bold text-white mb-1">{status}</h2>
           {!isEnded.current ? (
             <p className="text-emerald-100 text-md font-bold drop-shadow-md">
               {feedback}
             </p>
           ) : (
             <p className="text-emerald-100 text-sm font-medium">
                Analysis complete. Sending telemetry to Health Buddy.
             </p>
           )}
        </div>
      </div>
    </div>
  );
}