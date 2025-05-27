import {
  Room,
  RoomEvent,
  Track,
  createLocalAudioTrack
} from "https://cdn.skypack.dev/livekit-client";

document.addEventListener("DOMContentLoaded", () => {
  // UI elements
  const exerciseOptions   = document.querySelectorAll(".exercise-option");
  const startBtn          = document.getElementById("start-btn");
  const stopBtn           = document.getElementById("stop-btn");
  const setsInput         = document.getElementById("sets");
  const repsInput         = document.getElementById("reps");
  const currentExerciseEl = document.getElementById("current-exercise");
  const currentSetEl      = document.getElementById("current-set");
  const currentRepsEl     = document.getElementById("current-reps");
  const transcriptEl      = document.getElementById("transcript");

  let selectedExercise = null;
  let room             = null;
  let pollInterval     = null;

  // 1) Exercise selection
  exerciseOptions.forEach(opt => {
    opt.addEventListener("click", () => {
      exerciseOptions.forEach(x => x.classList.remove("selected"));
      opt.classList.add("selected");
      selectedExercise = opt.dataset.exercise;
    });
  });

  // 2) Fetch LiveKit token + URL
  async function fetchToken() {
    const res = await fetch("/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    if (!res.ok) throw new Error(`Token error ${res.status}`);
    return res.json();
  }

  // 3) Join LiveKit room + publish mic
  async function joinLiveKit() {
    const { token, url } = await fetchToken();
    room = new Room();
    await room.connect(url, token, { audio: false, video: false, data: true });

    // publish your microphone
    const micTrack = await createLocalAudioTrack();
    await room.localParticipant.publishTrack(micTrack, { simulcast: false });

    // subscribe to remote audio
    room.remoteParticipants.forEach(subscribeParticipant);
    room.on(RoomEvent.ParticipantConnected, subscribeParticipant);

    // subscribe to room-level data messages
    subscribeData();
  }

  // audio subscription
  function subscribeParticipant(participant) {
    participant.on(RoomEvent.TrackSubscribed, track => {
      if (track.kind === Track.Kind.Audio) {
        const audioEl = track.attach();
        audioEl.setAttribute('playsinline', '');
        audioEl.autoplay = true;
        audioEl.controls = false;
        document.body.appendChild(audioEl);
        audioEl.play().catch(() => {});
      }
    });
  }

  // data subscription at ROOM level
  function subscribeData() {
    room.on(RoomEvent.DataReceived, (payload, participant) => {
      const str = new TextDecoder().decode(payload);
      try {
        const msg = JSON.parse(str);

        if (msg.transcript) {
          transcriptEl.textContent += `\nYou: ${msg.transcript}`;
        }
        else if (msg.coach_chunk) {
          transcriptEl.textContent += `\nCoach: ${msg.coach_chunk}`;
        }
      } catch {
        // ignore non-JSON
      }
    });
  }

  // 5) Poll status (fallback)
  async function pollStatus() {
    try {
      const data = await fetch("/get_status").then(r => r.json());
      if (!data.exercise_running) {
        resetUI();
        return;
      }
      currentSetEl.textContent  = `${data.current_set} / ${data.total_sets}`;
      currentRepsEl.textContent = `${data.current_reps} / ${data.rep_goal}`;
    } catch {
      // ignore errors
    }
  }

  // reset UI & cleanup
  function resetUI() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
    if (room)         { room.disconnect(); room = null; }
    startBtn.disabled = false;
    stopBtn.disabled  = true;
    currentExerciseEl.textContent = "None";
    currentSetEl.textContent      = "0 / 0";
    currentRepsEl.textContent     = "0 / 0";
    transcriptEl.textContent      = "—";
  }

  // 6) Start workout
  startBtn.addEventListener("click", async () => {
    if (!selectedExercise) return alert("Please select an exercise first!");
    const sets = +setsInput.value, reps = +repsInput.value;
    if (!(sets > 0 && reps > 0)) return alert("Enter valid sets & reps.");

    // initialize UI
    currentExerciseEl.textContent = selectedExercise.replace("_"," ").toUpperCase();
    currentSetEl.textContent      = `1 / ${sets}`;
    currentRepsEl.textContent     = `0 / ${reps}`;
    transcriptEl.textContent      = "—";

    try {
      await joinLiveKit();
      const res = await fetch("/start_exercise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exercise_type: selectedExercise, sets, reps })
      });
      const body = await res.json();
      if (!body.success) throw new Error(body.error||"start_exercise failed");

      startBtn.disabled = true;
      stopBtn.disabled  = false;
      pollInterval = setInterval(pollStatus, 1000);
    } catch (err) {
      alert(`Error starting workout:\n${err.message}`);
      resetUI();
    }
  });

  // 7) Stop workout
  stopBtn.addEventListener("click", async () => {
    try { await fetch("/stop_exercise", { method: "POST" }); }
    catch {}
    resetUI();
  });
});
