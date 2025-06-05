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

# ── Detect Render environment ───────────────────────────────────────────
# On Render.com, the environment variable "RENDER" is typically set.
# If running on Render, we will NOT attempt to open /dev/video0.
ON_RENDER = os.getenv("RENDER") is not None

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

# ── Flask globals ───────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "fitness_trainer_secret_key"

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

# holds the very latest metrics for polling
latest_metrics = {}

# ── Camera helpers ─────────────────────────────────────────────────────
def initialize_camera():
    global camera
    if ON_RENDER:
        # On Render, do not initialize the camera
        return None

    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera:
        camera.release()
        camera = None

# ── Frame processing logic (shared by upload and streaming) ─────────────
def process_frame(frame):
    """
    Run pose estimation + overlay logic on a single BGR frame.
    Update metrics_q and latest_metrics if an exercise is running.
    Returns the annotated frame.
    """
    global exercise_running, current_exercise, current_exercise_data
    global exercise_counter, exercise_goal, sets_completed, sets_goal, latest_metrics

    pose_estimator = PoseEstimator()

    if exercise_running and current_exercise:
        results = pose_estimator.estimate_pose(frame, current_exercise_data['type'])
        if results.pose_landmarks:
            t = current_exercise_data['type']
            # ---- REP­COUNT LOGIC ------------------------------------
            if t == "squat":
                counter, angle, stage = current_exercise.track_squat(
                    results.pose_landmarks.landmark, frame
                )
                layout_indicators(frame, "squat", (counter, angle, stage))
                exercise_counter = counter
            elif t == "push_up":
                counter, angle, stage = current_exercise.track_push_up(
                    results.pose_landmarks.landmark, frame
                )
                layout_indicators(frame, "push_up", (counter, angle, stage))
                exercise_counter = counter
            else:  # hammer_curl
                cr, ar, cl, al, wr, wl, pr, pl, sr, sl = current_exercise.track_hammer_curl(
                    results.pose_landmarks.landmark, frame
                )
                layout_indicators(frame, "hammer_curl", (cr, ar, cl, al, wr, wl, pr, pl, sr, sl))
                exercise_counter = max(cr, cl)

            # ---- PUSH METRICS TO QUEUE ------------------------------
            metrics = {
                "exercise": t,
                "reps": exercise_counter,
                "reps_goal": exercise_goal,
                "set": sets_completed + 1,
                "sets_goal": sets_goal
            }
            metrics_q.put_nowait(metrics)
            latest_metrics = metrics

            # ---- OVERLAY TEXT (unchanged) ---------------------------
            info = get_exercise_info(t)
            draw_text_with_background(
                frame,
                f"Exercise: {info.get('name','N/A')}",
                (40, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                (118, 29, 14),
                1
            )
            draw_text_with_background(
                frame,
                f"Reps Goal: {exercise_goal}",
                (40, 80),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                (118, 29, 14),
                1
            )
            draw_text_with_background(
                frame,
                f"Sets Goal: {sets_goal}",
                (40, 110),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                (118, 29, 14),
                1
            )
            draw_text_with_background(
                frame,
                f"Current Set: {sets_completed + 1}",
                (40, 140),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                (118, 29, 14),
                1
            )

            # ---- Set / workout completion ---------------------------
            if exercise_counter >= exercise_goal:
                sets_completed += 1
                exercise_counter = 0
                if t in ("squat", "push_up"):
                    current_exercise.counter = 0
                else:
                    current_exercise.counter_right = current_exercise.counter_left = 0

            if sets_completed >= sets_goal:
                exercise_running = False
                draw_text_with_background(
                    frame,
                    "WORKOUT COMPLETE!",
                    (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2,
                    (255, 255, 255),
                    (0, 200, 0),
                    2
                )
        else:
            # No landmarks detected; optionally overlay a waiting message
            pass
    else:
        cv2.putText(
            frame,
            "Select an exercise to begin",
            (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1
        )

    return frame

# ── Frame generator (repo + metrics push) ───────────────────────────────
def generate_frames():
    """
    Yields JPEG-encoded frames from the local camera if not on Render.
    On Render, initialize_camera() returns None, so this loop will simply spin.
    """
    global output_frame, lock

    initialize_camera()
    pose_estimator = PoseEstimator()  # To warm up or initialize any internal models

    while True:
        if camera is None:
            # No camera available (e.g., on Render); spin until frames are uploaded via /upload_frame
            time.sleep(0.1)
            continue

        ok, frame = camera.read()
        if not ok:
            continue

        # Process and annotate the frame
        annotated = process_frame(frame)

        # Encode annotated frame as JPEG
        with lock:
            output_frame = annotated.copy()
        ret, buf = cv2.imencode('.jpg', output_frame)
        if not ret:
            continue

        # Yield a multipart response chunk
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

# ── LiveKit token endpoint ─────────────────────────────────────────────
@app.route("/token", methods=["POST"])
def token():
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET and LIVEKIT_URL):
        return jsonify({"error": "LiveKit env vars missing"}), 400

    data     = request.json or {}
    identity = data.get("identity", "trainer-ui")
    room     = data.get("room", "workout-room")

    try:
        jwt = (
            lk.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_grants(lk.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
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
        "exercise": w["exercise_type"].replace("_", " ").title(),
        "sets": w["sets"],
        "reps": w["reps"],
        "duration": f"{w['duration_seconds']//60}:{w['duration_seconds']%60:02d}"
    } for w in recent]
    return render_template(
        "dashboard.html",
        recent_workouts=table,
        weekly_workouts=sum(d["workout_count"] for d in weekly.values()),
        total_workouts=user["total_workouts"],
        total_exercises=user["total_exercises"],
        streak_days=user["streak_days"]
    )

@app.route("/video_feed")
def video_feed():
    """
    Streams multipart MJPEG frames from the local camera.
    On Render, this will not start since initialize_camera() returns None.
    """
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/start_exercise", methods=["POST"])
def start_exercise():
    global exercise_running, current_exercise, current_exercise_data
    global exercise_counter, exercise_goal, sets_completed, sets_goal, workout_start_time

    data = request.json
    exercise_type = data.get("exercise_type")
    sets_goal     = int(data.get("sets", 3))
    exercise_goal = int(data.get("reps", 10))

    # On local, initialize the camera; on Render, this is a no-op
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
    return "Profile page - Coming soon!"

# ── New route: receive browser‐captured frame, process, and return base64 JPEG ──
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    """
    Expects raw JPEG bytes in the request body (Content-Type: image/jpeg).
    Processes the frame with the exact overlay logic, then returns a
    JSON payload with the annotated frame as a base64-encoded JPEG.
    """
    global output_frame, lock

    # Read raw bytes from the request
    frame_data = request.data
    if not frame_data:
        return jsonify({"error": "No image data received"}), 400

    # Decode JPEG into numpy BGR image
    np_img = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # Process and annotate the frame
    annotated = process_frame(frame)

    # Encode annotated frame as JPEG
    ret, buf = cv2.imencode('.jpg', annotated)
    if not ret:
        return jsonify({"error": "Failed to encode annotated image"}), 500

    # Base64‐encode the JPEG and return
    jpg_bytes = buf.tobytes()
    b64_str = base64.b64encode(jpg_bytes).decode('utf-8')
    return jsonify({"frame": b64_str})

# ── Launch Flask + LiveKit agent thread ────────────────────────────────
if __name__ == "__main__":
    if not ON_RENDER:
        # Only initialize the camera locally
        initialize_camera()

    from coach_agent import entrypoint
    from livekit.agents.cli import run_app
    from livekit.agents.worker import WorkerOptions

    threading.Thread(
        target=lambda: run_app(WorkerOptions(entrypoint_fnc=entrypoint)),
        daemon=True
    ).start()

    logger.info("Flask running on port %s", os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
