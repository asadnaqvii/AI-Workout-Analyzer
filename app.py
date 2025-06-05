# render_app.py
"""
Standalone Flask entry‐point for AI‐Workout‐Analyzer on Render.com.

• Pulls in all of your original app.py imports / PoseEstimator / exercise logic
  exactly as-is (to preserve LiveKit, metrics, WorkoutLogger, etc.).
• Never tries to open /dev/video0 when on Render. Instead, captures frames in
  the browser and POSTs to /upload_frame.
• Returns a base‐64 JPEG for the browser to display with your exact overlay logic.

To deploy on Render:
1) Add this file to your repo.
2) Make sure requirements.txt includes:
     flask, mediapipe==0.10.11, opencv-python, python-dotenv, livekit-api, numpy, etc.
3) In Render’s Web Service settings, set:
     Start Command:   gunicorn render_app:app
4) Deploy, then open the public URL in a webcam‐enabled browser.
"""

import os
import json
import cv2
import threading
import time
import sys
import traceback
import logging
import base64
import numpy as np
from textwrap import dedent

from dotenv import load_dotenv
from flask import Flask, render_template, Response, request, jsonify

# ── LiveKit Server SDK (pip install livekit-api) ────────────────────────
from livekit import api as lk
from metrics import metrics_q                # queue shared with coach_agent

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── ENV (keys go in .env) ───────────────────────────────────────────────
load_dotenv()
LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL        = os.getenv("LIVEKIT_URL")

# ── Pose-estimation imports (repo originals) ────────────────────────────
try:
    from pose_estimation.estimation import PoseEstimator
    from exercises.squat import Squat
    from exercises.hammer_curl import HammerCurl
    from exercises.push_up import PushUp
    from feedback.information import get_exercise_info
    from feedback.layout import layout_indicators
    from utils.draw_text_with_background import draw_text_with_background
except ImportError:
    logger.error("Pose-estimation imports failed")
    traceback.print_exc()
    sys.exit(1)

# ── WorkoutLogger fallback (repo logic) ─────────────────────────────────
try:
    from db.workout_logger import WorkoutLogger
    workout_logger = WorkoutLogger()
except ImportError:
    class DummyWorkoutLogger:
        def log_workout(self,*a,**k): return {}
        def get_recent_workouts(self,*a,**k): return []
        def get_weekly_stats(self,*a,**k): return {}
        def get_exercise_distribution(self,*a,**k): return {}
        def get_user_stats(self,*a,**k):
            return {'total_workouts':0,'total_exercises':0,'streak_days':0}
    workout_logger = DummyWorkoutLogger()

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "fitness_trainer_secret_key"

# ── Globals (mirroring your original app.py) ───────────────────────────
camera = None
output_frame = None
lock = threading.Lock()

exercise_running = False
current_exercise = None
current_exercise_data = None
exercise_counter = 0
exercise_goal = 0
sets_completed = 0
sets_goal = 0
workout_start_time = None

# holds the very latest metrics for /get_status
latest_metrics = {}

# ── MediaPipe Pose setup ────────────────────────────────────────────
try:
    import mediapipe as mp
except ModuleNotFoundError:
    raise SystemExit("mediapipe is required – add it to requirements.txt: mediapipe==0.10.11")

mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
pose_engine = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Shared frame-processing routine (exactly your original logic) ───────
pose_estimator = PoseEstimator()

def process_frame(bgr: np.ndarray) -> np.ndarray:
    """
    Apply your original PoseEstimator + exercise overlay logic to a single BGR frame.
    This is exactly the same code you used inside generate_frames(...) but pulled
    out so we can call it on one frame at a time (browser→server).
    """
    global exercise_running, current_exercise, current_exercise_data
    global exercise_counter, exercise_goal, sets_completed, sets_goal, latest_metrics

    # 1) Run PoseEstimator (your repo’s code) on the incoming BGR frame
    results = pose_estimator.estimate_pose(bgr, 
                   current_exercise_data["type"] if current_exercise_data else None)

    if exercise_running and current_exercise and results.pose_landmarks:
        t = current_exercise_data["type"]

        # — rep-count logic —
        if t == "squat":
            counter, angle, stage = current_exercise.track_squat(
                results.pose_landmarks.landmark, bgr)
            layout_indicators(bgr, "squat", (counter, angle, stage))
            exercise_counter = counter

        elif t == "push_up":
            counter, angle, stage = current_exercise.track_push_up(
                results.pose_landmarks.landmark, bgr)
            layout_indicators(bgr, "push_up", (counter, angle, stage))
            exercise_counter = counter

        else:  # hammer_curl
            cr, ar, cl, al, wr, wl, pr, pl, sr, sl = current_exercise.track_hammer_curl(
                results.pose_landmarks.landmark, bgr)
            layout_indicators(bgr, "hammer_curl",
                              (cr, ar, cl, al, wr, wl, pr, pl, sr, sl))
            exercise_counter = max(cr, cl)

        # — push live metrics to queue —
        metrics = {
            "exercise":   t,
            "reps":       exercise_counter,
            "reps_goal":  exercise_goal,
            "set":        sets_completed + 1,
            "sets_goal":  sets_goal
        }
        metrics_q.put_nowait(metrics)
        latest_metrics = metrics

        # — HUD overlay (texts) —
        info = get_exercise_info(t)
        draw_text_with_background(bgr, f"Exercise: {info.get('name','N/A')}",
                                  (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                  (255,255,255), (118,29,14), 1)
        draw_text_with_background(bgr, f"Reps Goal: {exercise_goal}",
                                  (40, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                  (255,255,255), (118,29,14), 1)
        draw_text_with_background(bgr, f"Sets Goal: {sets_goal}",
                                  (40,110), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                  (255,255,255), (118,29,14), 1)
        draw_text_with_background(bgr, f"Current Set: {sets_completed+1}",
                                  (40,140), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                  (255,255,255), (118,29,14), 1)

        # — set / workout completion —
        if exercise_counter >= exercise_goal:
            sets_completed  += 1
            exercise_counter = 0
            if t in ("squat", "push_up"):
                current_exercise.counter = 0
            else:
                current_exercise.counter_right = current_exercise.counter_left = 0

        if sets_completed >= sets_goal:
            exercise_running = False
            draw_text_with_background(bgr, "WORKOUT COMPLETE!",
                (bgr.shape[1]//2 - 150, bgr.shape[0]//2),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), (0,200,0), 2)

    else:
        # If no exercise running, prompt user
        cv2.putText(bgr, "Select an exercise to begin",
                    (bgr.shape[1]//2 - 150, bgr.shape[0]//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

    return bgr

# ── Camera helpers ─────────────────────────────────────────────────────
def initialize_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera:
        camera.release()
        camera = None

# ── Frame generator (repo + metrics push) ───────────────────────────────
def generate_frames():
    global output_frame, lock
    global exercise_running, current_exercise, current_exercise_data
    global exercise_counter, exercise_goal, sets_completed, sets_goal, latest_metrics

    pose_estimator = PoseEstimator()

    while True:
        if camera is None:
            continue
        ok, frame = camera.read()
        if not ok:
            continue

        if exercise_running and current_exercise:
            results = pose_estimator.estimate_pose(frame, current_exercise_data['type'])
            if results.pose_landmarks:
                t = current_exercise_data['type']
                # ---- REP­COUNT LOGIC ------------------------------------
                if t=="squat":
                    counter, angle, stage = current_exercise.track_squat(
                        results.pose_landmarks.landmark, frame)
                    layout_indicators(frame,"squat",(counter,angle,stage))
                    exercise_counter = counter
                elif t=="push_up":
                    counter, angle, stage = current_exercise.track_push_up(
                        results.pose_landmarks.landmark, frame)
                    layout_indicators(frame,"push_up",(counter,angle,stage))
                    exercise_counter = counter
                else:  # hammer_curl
                    cr, ar, cl, al, wr, wl, pr, pl, sr, sl = current_exercise.track_hammer_curl(
                        results.pose_landmarks.landmark, frame)
                    layout_indicators(frame,"hammer_curl",
                        (cr,ar,cl,al,wr,wl,pr,pl,sr,sl))
                    exercise_counter = max(cr,cl)

                # ---- PUSH METRICS TO QUEUE ------------------------------
                metrics = {
                    "exercise": t,
                    "reps": exercise_counter,
                    "reps_goal": exercise_goal,
                    "set": sets_completed+1,
                    "sets_goal": sets_goal
                }
                metrics_q.put_nowait(metrics)
                latest_metrics = metrics

                # ---- OVERLAY TEXT (unchanged) ---------------------------
                info = get_exercise_info(t)
                draw_text_with_background(frame,f"Exercise: {info.get('name','N/A')}",
                                          (40,50),cv2.FONT_HERSHEY_DUPLEX,0.7,
                                          (255,255,255),(118,29,14),1)
                draw_text_with_background(frame,f"Reps Goal: {exercise_goal}",
                                          (40,80),cv2.FONT_HERSHEY_DUPLEX,0.7,
                                          (255,255,255),(118,29,14),1)
                draw_text_with_background(frame,f"Sets Goal: {sets_goal}",
                                          (40,110),cv2.FONT_HERSHEY_DUPLEX,0.7,
                                          (255,255,255),(118,29,14),1)
                draw_text_with_background(frame,f"Current Set: {sets_completed+1}",
                                          (40,140),cv2.FONT_HERSHEY_DUPLEX,0.7,
                                          (255,255,255),(118,29,14),1)

                # ---- Set / workout completion ---------------------------
                if exercise_counter>=exercise_goal:
                    sets_completed += 1
                    exercise_counter = 0
                    if t in ("squat","push_up"):
                        current_exercise.counter = 0
                    else:
                        current_exercise.counter_right = current_exercise.counter_left = 0
                if sets_completed >= sets_goal:
                    exercise_running = False
                    draw_text_with_background(frame,"WORKOUT COMPLETE!",
                        (frame.shape[1]//2-150,frame.shape[0]//2),
                        cv2.FONT_HERSHEY_DUPLEX,1.2,(255,255,255),(0,200,0),2)
        else:
            cv2.putText(frame,"Select an exercise to begin",
                        (frame.shape[1]//2-150,frame.shape[0]//2),
                        cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)

        with lock:
            output_frame = frame.copy()
        ret, buf = cv2.imencode('.jpg', output_frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ── LiveKit token endpoint ─────────────────────────────────────────────
@app.route("/token", methods=["POST"])
def token():
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET and LIVEKIT_URL):
        return jsonify({"error":"LiveKit env vars missing"}), 400

    data     = request.json or {}
    identity = data.get("identity","trainer-ui")
    room     = data.get("room","workout-room")

    try:
        jwt = ( lk.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                  .with_identity(identity)
                  .with_grants(lk.VideoGrants(room_join=True, room=room))
                  .to_jwt() )
        return jsonify({"token": jwt, "url": LIVEKIT_URL})
    except Exception as e:
        logger.exception("LiveKit token generation failed")
        return jsonify({"error": str(e)}), 500

# ── Original routes (index, dashboard, video_feed, etc.) ───────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    recent = workout_logger.get_recent_workouts(5)
    weekly = workout_logger.get_weekly_stats()
    user   = workout_logger.get_user_stats()
    table = [{
        "date": w["date"],
        "exercise": w["exercise_type"].replace("_"," ").title(),
        "sets": w["sets"],
        "reps": w["reps"],
        "duration": f"{w['duration_seconds']//60}:{w['duration_seconds']%60:02d}"
    } for w in recent]
    return render_template("dashboard.html",
        recent_workouts=table,
        weekly_workouts=sum(d["workout_count"] for d in weekly.values()),
        total_workouts=user["total_workouts"],
        total_exercises=user["total_exercises"],
        streak_days=user["streak_days"]
    )

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_exercise", methods=["POST"])
def start_exercise():
    global exercise_running, current_exercise, current_exercise_data
    global exercise_counter, exercise_goal, sets_completed, sets_goal, workout_start_time

    data = request.json
    exercise_type = data.get("exercise_type")
    sets_goal     = int(data.get("sets",3))
    exercise_goal = int(data.get("reps",10))

    initialize_camera()
    exercise_counter = sets_completed = 0
    workout_start_time = time.time()

    if exercise_type == "squat":
        current_exercise = Squat()
    elif exercise_type == "push_up":
        current_exercise = PushUp()
    elif exercise_type == "hammer_curl":
        current_exercise = HammerCurl()
    else:
        return jsonify({"success":False,"error":"Invalid exercise type"})

    current_exercise_data = {"type": exercise_type, "sets": sets_goal, "reps": exercise_goal}
    exercise_running = True
    return jsonify({"success": True})

@app.route("/stop_exercise", methods=["POST"])
def stop_exercise():
    global exercise_running, workout_start_time
    exercise_running = False
    duration = int(time.time() - workout_start_time) if workout_start_time else 0

    workout_logger.log_workout(
        exercise_type=current_exercise_data["type"],
        sets=sets_completed + (1 if exercise_counter>0 else 0),
        reps=exercise_goal,
        duration_seconds=duration
    )
    return jsonify({"success":True, "duration": duration})

@app.route("/get_status")
def get_status():
    return jsonify({
        "exercise_running": exercise_running,
        "current_reps": exercise_counter,
        "current_set": (sets_completed+1) if exercise_running else 0,
        "total_sets": sets_goal,
        "rep_goal": exercise_goal
    })

@app.route("/latest_metrics")
def latest_metrics_api():
    return jsonify(latest_metrics)

@app.route("/profile")
def profile():
    return "Profile page - Coming soon!"

# ── Launch Flask + LiveKit agent thread ───────────────────────────────
if __name__ == "__main__":
    initialize_camera()
    from coach_agent import entrypoint
    from livekit.agents.cli import run_app
    from livekit.agents.worker import WorkerOptions

    threading.Thread(
        target=lambda: run_app(WorkerOptions(entrypoint_fnc=entrypoint)),
        daemon=True
    ).start()


# ── Embedded HTML + JS ─────────────────────────────────────────────────
# This single‐page UI replicates your original controls (Start/Stop, sets/reps,
# LiveKit transcript, etc.), but also grabs the browser’s webcam and POSTs each
# frame to /upload_frame. When /upload_frame returns a base-64 JPEG, it’s painted
# into <img id="overlay">. Feel free to tweak the styling below, but you do not
# need to modify any external template files—everything is self‐contained.

PAGE_HTML = dedent("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Workout Analyzer (Render)</title>
  <style>
    body { margin:0; font-family: sans-serif; background:#121212; color:#eee; 
           display:flex; flex-direction:column; align-items:center; }
    h1  { margin:2rem 0 1rem; font-weight:500; }
    #overlay { max-width:90vw; border:4px solid #29e; border-radius:8px; }
    #err     { color:#f66; margin-top:1rem; }
    .controls { display:flex; flex-wrap: wrap; justify-content:center; gap:1rem; 
                margin-top:1.5rem; }
    .control-box { background:#1e1e1e; padding:1rem; border-radius:8px; }
    .control-box h2 { margin:0 0 .5rem 0; }
    .exercise-option { cursor:pointer; border:2px solid transparent; 
                       border-radius:6px; padding:0.5rem; text-align:center; }
    .exercise-option.selected { border-color:#29e; }
    .exercise-option img  { width:60px; height:60px; }
    button { padding:0.6rem 1.2rem; font-size:1rem; border:none; border-radius:6px; 
             cursor:pointer; }
    .btn-primary { background:#29e; color:#111; }
    .btn-secondary { background:#444; color:#eee; }
    .status-display { margin-top:.5rem; }
    .status-display div { margin-bottom:.3rem; }
    #transcript { background:#1e1e1e; padding:1rem; border-radius:6px; 
                  width:300px; height:100px; overflow-y:auto; white-space: pre-wrap; }
    video { display:none; }
  </style>
</head>
<body>
  <h1>AI Workout Analyzer</h1>
  <video id="clientCam" autoplay playsinline></video>
  <img id="overlay" alt="Processed frame will appear here" />
  <div id="err"></div>

  <div class="controls">
    <!-- Exercise Selection -->
    <div class="control-box">
      <h2>Select Exercise</h2>
      <div id="options">
        <div class="exercise-option" data-exercise="squat">
          <img src="/static/images/squat.png" alt="Squat"/>
          <div>Squat</div>
        </div>
        <div class="exercise-option" data-exercise="push_up">
          <img src="/static/images/push_up.png" alt="Push Up"/>
          <div>Push Up</div>
        </div>
        <div class="exercise-option" data-exercise="hammer_curl">
          <img src="/static/images/hammer_curl.png" alt="Hammer Curl"/>
          <div>Hammer Curl</div>
        </div>
      </div>
    </div>

    <!-- Workout Config -->
    <div class="control-box">
      <h2>Configure</h2>
      <label>Sets: <input type="number" id="sets" value="3" min="1" max="10" /></label><br>
      <label>Reps: <input type="number" id="reps" value="10" min="1" max="30" /></label>
    </div>

    <!-- Start/Stop Buttons -->
    <div class="control-box">
      <h2>Actions</h2>
      <button id="startBtn" class="btn-primary">Start Workout</button>
      <button id="stopBtn"  class="btn-secondary" disabled>Stop Workout</button>
    </div>

    <!-- Status Display -->
    <div class="control-box">
      <h2>Status</h2>
      <div class="status-display">
        <div>Exercise: <span id="currentExercise">None</span></div>
        <div>Set: <span id="currentSet">0 / 0</span></div>
        <div>Reps: <span id="currentReps">0 / 0</span></div>
      </div>
    </div>

    <!-- LiveKit Transcript -->
    <div class="control-box">
      <h2>Transcript</h2>
      <div id="transcript">—</div>
    </div>
  </div>

<script type="module">
// ───────────────────────────────────────────────────────────────────────────
// Import LiveKit client for audio/data (no webcam). We’ll publish mic only.
// ───────────────────────────────────────────────────────────────────────────
import {
  Room,
  RoomEvent,
  Track,
  createLocalAudioTrack
} from "https://cdn.skypack.dev/livekit-client";

document.addEventListener("DOMContentLoaded", () => {
  // References to UI elements
  const cam       = document.getElementById("clientCam");
  const overlay   = document.getElementById("overlay");
  const errEl     = document.getElementById("err");
  const options   = document.querySelectorAll(".exercise-option");
  const setsInput = document.getElementById("sets");
  const repsInput = document.getElementById("reps");
  const startBtn  = document.getElementById("startBtn");
  const stopBtn   = document.getElementById("stopBtn");
  const curEx     = document.getElementById("currentExercise");
  const curSet    = document.getElementById("currentSet");
  const curReps   = document.getElementById("currentReps");
  const transcript= document.getElementById("transcript");

  let selectedExercise = null;
  let room = null;
  let uploadInterval = null;
  let pollInterval = null;

  // ───────────────────────── getUserMedia + uploadLoop ────────────────────
  async function startCameraLoop() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      cam.srcObject = stream;
    } catch (e) {
      errEl.textContent = "Camera access denied: " + e.message;
      return;
    }

    const canvas = document.createElement("canvas");
    const ctx    = canvas.getContext("2d");

    uploadInterval = setInterval(async () => {
      if (cam.readyState < 2) return;
      canvas.width  = cam.videoWidth;
      canvas.height = cam.videoHeight;
      ctx.drawImage(cam, 0, 0);

      const dataURL = canvas.toDataURL("image/jpeg", 0.8);
      try {
        const res = await fetch("/upload_frame", {
          method: "POST",
          headers: { "Content-Type": "text/plain" },
          body: dataURL
        });
        if (res.ok) {
          const b64 = await res.text();
          overlay.src = "data:image/jpeg;base64," + b64;
        }
      } catch (e) {
        errEl.textContent = "Network error: " + e.message;
      }
    }, 100);
  }

  function stopCameraLoop() {
    if (uploadInterval) {
      clearInterval(uploadInterval);
      uploadInterval = null;
    }
    if (cam.srcObject) {
      cam.srcObject.getTracks().forEach(t => t.stop());
      cam.srcObject = null;
    }
  }
  // ───────────────────────────────────────────────────────────────────────────

  // 1) Exercise selection
  options.forEach(el => {
    el.addEventListener("click", () => {
      options.forEach(o => o.classList.remove("selected"));
      el.classList.add("selected");
      selectedExercise = el.dataset.exercise;
    });
  });

  // 2) Fetch LiveKit token
  async function fetchToken() {
    const res = await fetch("/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    if (!res.ok) throw new Error(`Token fetch failed: ${res.status}`);
    return res.json();
  }

  // 3) Join LiveKit room (publish mic only, subscribe data)
  async function joinLiveKit() {
    const { token, url } = await fetchToken();
    room = new Room();
    await room.connect(url, token, { audio: false, video: false, data: true });

    // Publish microphone
    const micTrack = await createLocalAudioTrack();
    await room.localParticipant.publishTrack(micTrack, { simulcast: false });

    // Subscribe to incoming data (transcript/coaching)
    room.on(RoomEvent.DataReceived, (payload) => {
      try {
        const msg = JSON.parse(new TextDecoder().decode(payload));
        if (msg.transcript) {
          transcript.textContent += `\nYou: ${msg.transcript}`;
        } else if (msg.coach_chunk) {
          transcript.textContent += `\nCoach: ${msg.coach_chunk}`;
        }
      } catch { /* ignore parse errors */ }
    });
  }

  // 4) Poll /get_status every second to update reps/sets
  async function pollStatus() {
    try {
      const data = await fetch("/get_status").then(r => r.json());
      if (!data.exercise_running) {
        resetUI();
        return;
      }
      curSet.textContent  = `${data.current_set} / ${data.total_sets}`;
      curReps.textContent = `${data.current_reps} / ${data.rep_goal}`;
    } catch { /* ignore errors */ }
  }

  // 5) Reset UI & teardown
  function resetUI() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
    if (room) { room.disconnect(); room = null; }
    stopCameraLoop();

    startBtn.disabled = false;
    stopBtn.disabled  = true;
    curEx.textContent = "None";
    curSet.textContent = "0 / 0";
    curReps.textContent= "0 / 0";
    transcript.textContent = "—";
    overlay.src = "";
  }

  // 6) Start workout
  startBtn.addEventListener("click", async () => {
    if (!selectedExercise) {
      alert("Please select an exercise first!");
      return;
    }
    const sets = Number(setsInput.value);
    const reps = Number(repsInput.value);
    if (!(sets > 0 && reps > 0)) {
      alert("Enter valid sets & reps.");
      return;
    }

    // Update UI immediately
    curEx.textContent = selectedExercise.replace("_"," ").toUpperCase();
    curSet.textContent= `1 / ${sets}`;
    curReps.textContent = `0 / ${reps}`;
    transcript.textContent = "—";

    try {
      // 1) Start sending camera frames
      await startCameraLoop();

      // 2) Join LiveKit for audio/data
      await joinLiveKit();

      // 3) Tell server to start exercise
      const res = await fetch("/start_exercise", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          exercise_type: selectedExercise,
          sets: sets,
          reps: reps
        })
      });
      const body = await res.json();
      if (!body.success) throw new Error(body.error || "start_exercise failed");

      // 4) Update buttons & start polling
      startBtn.disabled = true;
      stopBtn.disabled = false;
      pollInterval = setInterval(pollStatus, 1000);

    } catch (err) {
      alert(`Error starting workout:\n${err.message}`);
      resetUI();
    }
  });

  // 7) Stop workout
  stopBtn.addEventListener("click", async () => {
    try {
      await fetch("/stop_exercise", { method: "POST" });
    } catch { /* ignore */ }
    resetUI();
  });
});
</script>
</body>
</html>
""")

# ── Route: serve the embedded HTML page ──────────────────────────────────
@app.route("/")
def index():
    """
    Instead of rendering your old index.html, we just serve the embedded PAGE_HTML here.
    This single page:
      • Grabs camera in browser → sends to /upload_frame
      • Displays your Start/Stop buttons, LiveKit integration, status, etc.
    """
    return Response(PAGE_HTML, mimetype="text/html")


# ── New endpoint: browser → server frame upload ─────────────────────────
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """
    Receive a base-64 JPEG (dataURL) from the browser, run the exact
    process_frame(...) logic (PoseEstimator + exercise overlay) on it,
    and return a raw base-64 JPEG string (no dataURL header).
    """
    global camera, output_frame, lock
    global exercise_running, current_exercise, current_exercise_data
    global exercise_counter, exercise_goal, sets_completed, sets_goal

    try:
        data_url = request.get_data(as_text=True)
        _, b64data = data_url.split(",", 1)
        raw_bytes = base64.b64decode(b64data)
        np_arr    = np.frombuffer(raw_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Call your original process_frame(...) to draw skeleton/text/etc.
        processed = process_frame(frame)

        ok, buf = cv2.imencode(".jpg", processed)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return base64.b64encode(buf).decode("ascii")

    except Exception as exc:
        logger.exception("upload_frame error")
        return Response(f"error: {exc}", status=500)


# ── Optional local dev video_feed (works exactly like your old version) ─
@app.route("/video_feed")
def video_feed():
    """
    So that you can still test “python render_app.py” locally and hit
    http://localhost:5000/video_feed to see your webcam+overlay. On Render
    this route will simply not be used (Render users see the fetch-based flow).
    """
    cap = cv2.VideoCapture(0)
    lock2 = threading.Lock()
    pe   = PoseEstimator()

    def generator():
        global exercise_running, current_exercise, current_exercise_data
        global exercise_counter, exercise_goal, sets_completed, sets_goal

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Use the same overlay logic as process_frame:
            results = pe.estimate_pose(frame, current_exercise_data["type"] if current_exercise_data else None)
            if exercise_running and current_exercise and results.pose_landmarks:
                t = current_exercise_data["type"]

                if t == "squat":
                    counter, angle, stage = current_exercise.track_squat(
                        results.pose_landmarks.landmark, frame)
                    layout_indicators(frame, "squat", (counter, angle, stage))
                    exercise_counter = counter
                elif t == "push_up":
                    counter, angle, stage = current_exercise.track_push_up(
                        results.pose_landmarks.landmark, frame)
                    layout_indicators(frame, "push_up", (counter, angle, stage))
                    exercise_counter = counter
                else:  # hammer_curl
                    cr, ar, cl, al, wr, wl, pr, pl, sr, sl = current_exercise.track_hammer_curl(
                        results.pose_landmarks.landmark, frame)
                    layout_indicators(frame, "hammer_curl",
                                      (cr, ar, cl, al, wr, wl, pr, pl, sr, sl))
                    exercise_counter = max(cr, cl)

                metrics = {
                    "exercise": t,
                    "reps": exercise_counter,
                    "reps_goal": exercise_goal,
                    "set": sets_completed + 1,
                    "sets_goal": sets_goal
                }
                metrics_q.put_nowait(metrics)
                latest_metrics = metrics

                info = get_exercise_info(t)
                draw_text_with_background(frame, f"Exercise: {info.get('name','N/A')}",
                                          (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                          (255,255,255), (118,29,14), 1)
                draw_text_with_background(frame, f"Reps Goal: {exercise_goal}",
                                          (40, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                          (255,255,255), (118,29,14), 1)
                draw_text_with_background(frame, f"Sets Goal: {sets_goal}",
                                          (40,110), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                          (255,255,255), (118,29,14), 1)
                draw_text_with_background(frame, f"Current Set: {sets_completed+1}",
                                          (40,140), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                          (255,255,255), (118,29,14), 1)

                if exercise_counter >= exercise_goal:
                    sets_completed += 1
                    exercise_counter = 0
                    if t in ("squat","push_up"):
                        current_exercise.counter = 0
                    else:
                        current_exercise.counter_right = current_exercise.counter_left = 0
                if sets_completed >= sets_goal:
                    exercise_running = False
                    draw_text_with_background(frame, "WORKOUT COMPLETE!",
                        (frame.shape[1]//2 - 150, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), (0,200,0), 2)
            else:
                cv2.putText(frame, "Select an exercise to begin",
                            (frame.shape[1]//2 - 150, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

            with lock2:
                _, buf = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" 
                       + buf.tobytes() + b"\r\n")
    return Response(generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── LiveKit token endpoint (unchanged) ─────────────────────────────────
@app.route("/token", methods=["POST"])
def token():
    """
    Exactly your original /token logic, returning LiveKit JWT + URL.
    """
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET and LIVEKIT_URL):
        return jsonify({"error":"LiveKit env vars missing"}), 400

    data     = request.json or {}
    identity = data.get("identity","trainer-ui")
    roomName = data.get("room","workout-room")

    try:
        jwt = ( lk.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                  .with_identity(identity)
                  .with_grants(lk.VideoGrants(room_join=True, room=roomName))
                  .to_jwt() )
        return jsonify({"token": jwt, "url": LIVEKIT_URL})
    except Exception as e:
        logger.exception("LiveKit token generation failed")
        return jsonify({"error": str(e)}), 500


# ── Original REST routes (dashboard, start/stop, get_status, etc.) ─────
@app.route("/dashboard")
def dashboard():
    recent = workout_logger.get_recent_workouts(5)
    weekly = workout_logger.get_weekly_stats()
    user   = workout_logger.get_user_stats()
    table = [{
        "date": w["date"],
        "exercise": w["exercise_type"].replace("_"," ").title(),
        "sets": w["sets"],
        "reps": w["reps"],
        "duration": f"{w['duration_seconds']//60}:{w['duration_seconds']%60:02d}"
    } for w in recent]
    return render_template("dashboard.html",
        recent_workouts=table,
        weekly_workouts=sum(d["workout_count"] for d in weekly.values()),
        total_workouts=user["total_workouts"],
        total_exercises=user["total_exercises"],
        streak_days=user["streak_days"]
    )

@app.route("/start_exercise", methods=["POST"])
def start_exercise():
    global camera, exercise_running, current_exercise, current_exercise_data
    global exercise_counter, exercise_goal, sets_completed, sets_goal, workout_start_time

    data = request.json
    exercise_type = data.get("exercise_type")
    sets_goal     = int(data.get("sets", 3))
    exercise_goal = int(data.get("reps", 10))

    exercise_counter = sets_completed = 0
    workout_start_time = time.time()

    # Initialize the correct exercise tracker
    if exercise_type == "squat":
        current_exercise = Squat()
    elif exercise_type == "push_up":
        current_exercise = PushUp()
    elif exercise_type == "hammer_curl":
        current_exercise = HammerCurl()
    else:
        return jsonify({"success": False, "error": "Invalid exercise type"})

    current_exercise_data = {
        "type": exercise_type,
        "sets": sets_goal,
        "reps": exercise_goal
    }
    exercise_running = True
    return jsonify({"success": True})

@app.route("/stop_exercise", methods=["POST"])
def stop_exercise():
    global exercise_running, workout_start_time
    exercise_running = False
    duration = int(time.time() - workout_start_time) if workout_start_time else 0

    workout_logger.log_workout(
        exercise_type=current_exercise_data["type"],
        sets=sets_completed + (1 if exercise_counter > 0 else 0),
        reps=exercise_goal,
        duration_seconds=duration
    )
    return jsonify({"success": True, "duration": duration})

@app.route("/get_status")
def get_status():
    return jsonify({
        "exercise_running": exercise_running,
        "current_reps": exercise_counter,
        "current_set": (sets_completed + 1) if exercise_running else 0,
        "total_sets": sets_goal,
        "rep_goal": exercise_goal
    })

@app.route("/latest_metrics")
def latest_metrics_api():
    return jsonify(latest_metrics)

@app.route("/profile")
def profile():
    return "Profile page – Coming soon!"


# ── Main: run Flask + LiveKit agent thread ─────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Render‐compatible Flask app …")
    from coach_agent import entrypoint
    from livekit.agents.cli import run_app
    from livekit.agents.worker import WorkerOptions

    threading.Thread(
        target=lambda: run_app(WorkerOptions(entrypoint_fnc=entrypoint)),
        daemon=True
    ).start()

    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
