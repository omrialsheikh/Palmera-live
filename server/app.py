"""
Palmera Live - FastAPI Server
WebSocket-based video streaming with avatar animation.
"""

import asyncio
import base64
import json
import time
import sys
import os

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.pipeline import PalmeraLivePipeline

app = FastAPI(title="Palmera Live")

# Global pipeline instance
pipeline: PalmeraLivePipeline = None


@app.on_event("startup")
async def startup():
    global pipeline
    print("\n" + "=" * 50)
    print("  Palmera Live - Starting Server")
    print("=" * 50 + "\n")

    pipeline = PalmeraLivePipeline()
    pipeline.load_models()

    print("\n[Server] Ready and waiting for connections.")


# --- REST Endpoints ---

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "initialized": pipeline.is_initialized if pipeline else False,
    }


@app.post("/api/upload_avatar")
async def upload_avatar(file: UploadFile = File(...)):
    """Upload reference avatar image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image file"}

    result = pipeline.init_avatar(image)
    return result


@app.post("/api/upload_background")
async def upload_background(file: UploadFile = File(...)):
    """Upload background image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image file"}

    pipeline.set_background(image)
    return {"status": "background set"}


@app.post("/api/reset")
async def reset():
    """Reset the pipeline."""
    pipeline.reset()
    return {"status": "reset"}


# --- WebSocket Video Stream ---

@app.websocket("/api/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Bidirectional video stream.
    Client sends: base64 encoded webcam frames
    Server sends: base64 encoded generated frames
    """
    await ws.accept()
    print("[WS] Client connected.")

    try:
        while True:
            # Receive webcam frame from client
            data = await ws.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                # Decode base64 frame
                frame_bytes = base64.b64decode(message["data"])
                nparr = np.frombuffer(frame_bytes, np.uint8)
                webcam_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if webcam_frame is None:
                    continue

                # Process through pipeline
                output_frame = pipeline.process_frame(webcam_frame)

                # Send back any queued frames
                while pipeline.has_frames():
                    frame = pipeline.get_next_frame()
                    if frame is not None:
                        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_b64 = base64.b64encode(buffer).decode("utf-8")

                        await ws.send_text(json.dumps({
                            "type": "frame",
                            "data": frame_b64,
                            "timestamp": time.time(),
                        }))

            elif message.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error: {e}")
        await ws.close()


# --- Static Files (Client) ---

client_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client")
if os.path.exists(client_dir):
    app.mount("/static", StaticFiles(directory=client_dir), name="static")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(client_dir, "index.html"))


# --- Run ---

if __name__ == "__main__":
    import uvicorn

    config = OmegaConf.load("configs/pipeline.yaml")
    uvicorn.run(
        app,
        host=config.get("server", {}).get("host", "0.0.0.0"),
        port=config.get("server", {}).get("port", 7860),
    )
