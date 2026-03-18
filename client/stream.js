/**
 * Palmera Live - Client-side WebSocket Video Stream
 */

const webcamEl = document.getElementById("webcam");
const outputCanvas = document.getElementById("output");
const ctx = outputCanvas.getContext("2d");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fpsCounter = document.getElementById("fpsCounter");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let ws = null;
let webcamStream = null;
let sendInterval = null;
let frameCount = 0;
let lastFpsTime = Date.now();

// --- Upload Handlers ---

document.getElementById("avatarInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  setStatus("processing", "Uploading avatar...");

  const res = await fetch("/api/upload_avatar", { method: "POST", body: formData });
  const data = await res.json();

  if (data.status === "ready") {
    const area = document.getElementById("avatarUpload");
    area.classList.add("has-image");
    area.innerHTML = `<img src="${URL.createObjectURL(file)}">`;
    startBtn.disabled = false;
    setStatus("connected", "Avatar ready");
  } else {
    setStatus("disconnected", "Upload failed: " + (data.error || "unknown"));
  }
});

document.getElementById("bgInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/upload_background", { method: "POST", body: formData });
  const data = await res.json();

  if (data.status === "background set") {
    const area = document.getElementById("bgUpload");
    area.classList.add("has-image");
    area.innerHTML = `<img src="${URL.createObjectURL(file)}">`;
  }
});

// --- Webcam ---

async function initWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 768, height: 1024, facingMode: "user" },
      audio: false,
    });
    webcamEl.srcObject = webcamStream;
  } catch (err) {
    console.error("Webcam error:", err);
    setStatus("disconnected", "Webcam access denied");
  }
}

// --- Capture & Send ---

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = webcamEl.videoWidth;
  canvas.height = webcamEl.videoHeight;
  const tmpCtx = canvas.getContext("2d");
  tmpCtx.drawImage(webcamEl, 0, 0);

  canvas.toBlob(
    (blob) => {
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;

      const reader = new FileReader();
      reader.onloadend = () => {
        const b64 = reader.result.split(",")[1];
        ws.send(JSON.stringify({ type: "frame", data: b64 }));
      };
      reader.readAsDataURL(blob);
    },
    "image/jpeg",
    0.8
  );
}

// --- Receive & Display ---

function handleMessage(event) {
  const message = JSON.parse(event.data);

  if (message.type === "frame") {
    const img = new Image();
    img.onload = () => {
      outputCanvas.width = img.width;
      outputCanvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      frameCount++;
      const now = Date.now();
      if (now - lastFpsTime >= 1000) {
        fpsCounter.textContent = `${frameCount} FPS`;
        frameCount = 0;
        lastFpsTime = now;
      }
    };
    img.src = "data:image/jpeg;base64," + message.data;
  }
}

// --- WebSocket ---

function connectWS() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${window.location.host}/api/ws/stream`;

  ws = new WebSocket(url);

  ws.onopen = () => {
    setStatus("connected", "Connected");
    // Send frames at ~10 FPS
    sendInterval = setInterval(captureFrame, 100);
  };

  ws.onmessage = handleMessage;

  ws.onclose = () => {
    setStatus("disconnected", "Disconnected");
    clearInterval(sendInterval);
  };

  ws.onerror = (err) => {
    console.error("WS error:", err);
    setStatus("disconnected", "Connection error");
  };
}

// --- Controls ---

async function startStream() {
  await initWebcam();
  connectWS();
  startBtn.disabled = true;
  stopBtn.disabled = false;
}

function stopStream() {
  if (ws) ws.close();
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
  }
  clearInterval(sendInterval);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("disconnected", "Stopped");
}

// --- Status ---

function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

// Init
setStatus("disconnected", "Upload an avatar to begin");
