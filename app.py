from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime
from collections import deque
import asyncio
from starlette.websockets import WebSocketDisconnect, WebSocketState
import logging

app = FastAPI()

SAVE_VIDEO_DIR = "./static/saved_videos"
DETECTION_HISTORY_FILE = os.path.join(SAVE_VIDEO_DIR, "detection_history.json")
os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO("./best_yolo11_final.pt")

CAMERA_CONFIGS = {
    "camera1": {"id": "Camera 1", "location": "Location_1"},
    "camera2_browser": {"id": "Camera 2 Browser", "location": "Location_2"},
    "camera3_browser": {"id": "Camera 3 Browser", "location": "Location_3"},
    "camera4_browser": {"id": "Camera 4 Browser", "location": "Location_4"}
}

@app.get("/", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == "admin" and password == "admin":
        return RedirectResponse("/home", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/services", response_class=HTMLResponse)
def services(request: Request):
    return templates.TemplateResponse("services.html", {"request": request})




@app.get("/gas_detection", response_class=HTMLResponse)
def gas_detection(request: Request):
    return templates.TemplateResponse("gas_detection.html", {"request": request})

@app.get("/camera_manage", response_class=HTMLResponse)
def camera_manage(request: Request):
    return templates.TemplateResponse("camera_manage.html", {
        "request": request,
        "camera_configs": list(CAMERA_CONFIGS.values())
    })



@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    detection_records = []
    if os.path.exists(DETECTION_HISTORY_FILE):
        try:
            with open(DETECTION_HISTORY_FILE, "r") as f:
                detection_records = json.load(f)
        except Exception:
            pass
    detection_records.reverse()  # Show latest first
    return templates.TemplateResponse("history.html", {
        "request": request,
        "history_list": detection_records
    })

async def process_camera(websocket: WebSocket, camera_id: str, location: str):
    await websocket.accept()
    logging.info(f"[{camera_id}] WebSocket accepted")

    MAX_FPS = 10
    PRE_SECONDS = 10
    POST_SECONDS = 10
    frame_buffer = deque(maxlen=MAX_FPS * PRE_SECONDS)
    post_record_frames_left = 0
    recording = False
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    consecutive_detection_count = 0
    CONSECUTIVE_DETECTION_THRESHOLD = 3

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                logging.warning(f"[{camera_id}] WebSocket state not connected.")
                break

            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                logging.info(f"[{camera_id}] WebSocket disconnected")
                break
            except Exception as e:
                logging.error(f"[{camera_id}] Error receiving data: {e}")
                break

            try:
                image_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                logging.error(f"[{camera_id}] Frame decode error: {e}")
                continue

            frame_buffer.append(frame.copy())

            try:
                results = model(frame)[0]
            except Exception as e:
                logging.error(f"[{camera_id}] YOLO inference error: {e}")
                continue

            gas_detected = False
            detections = []

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(conf, 2)})
                    gas_detected = True

            try:
                await websocket.send_json(detections)
            except Exception as e:
                logging.error(f"[{camera_id}] Error sending detections: {e}")
                break

            if gas_detected:
                consecutive_detection_count += 1
            else:
                consecutive_detection_count = 0

            if consecutive_detection_count >= CONSECUTIVE_DETECTION_THRESHOLD and not recording:
                recording = True
                post_record_frames_left = MAX_FPS * POST_SECONDS
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"{camera_id}_{timestamp}.mp4"
                video_path = os.path.join(SAVE_VIDEO_DIR, video_filename)
                out = cv2.VideoWriter(video_path, fourcc, MAX_FPS, (frame.shape[1], frame.shape[0]))
                for buffered_frame in frame_buffer:
                    out.write(buffered_frame)

                record = {
                    "camera": camera_id,
                    "location": location,
                    "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "video_path": video_path.replace("\\", "/")
                }
                history = []
                if os.path.exists(DETECTION_HISTORY_FILE):
                    try:
                        with open(DETECTION_HISTORY_FILE, "r") as f:
                            history = json.load(f)
                    except Exception:
                        pass
                history.append(record)
                with open(DETECTION_HISTORY_FILE, "w") as f:
                    json.dump(history, f, indent=4)

            if recording and out:
                out.write(frame)
                if not gas_detected:
                    post_record_frames_left -= 1
                    if post_record_frames_left <= 0:
                        recording = False
                        out.release()
                        out = None

    finally:
        if out:
            out.release()
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        logging.info(f"[{camera_id}] WebSocket cleanup completed")


@app.websocket("/ws/{camera_key}")
async def generic_camera_ws(websocket: WebSocket, camera_key: str):
    if camera_key in CAMERA_CONFIGS:
        config = CAMERA_CONFIGS[camera_key]
        await process_camera(websocket, config["id"], config["location"])
    else:
        await websocket.close()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detect_from_ip_camera("http://192.168.0.116:8080/video?type=motionjpeg", "Camera 2 IP", "Entrance Hall"))
    asyncio.create_task(detect_from_ip_camera("http://192.168.0.116:8080/video?type=motionjpeg", "Camera 3 IP", "Corridor"))
    asyncio.create_task(detect_from_ip_camera("http://192.168.0.116:8080/video?type=motionjpeg", "Camera 4 IP", "Lab Area"))

async def detect_from_ip_camera(ip_url, camera_id, location):
    print(f"[{camera_id}] Connecting to {ip_url}")
    cap = cv2.VideoCapture(ip_url)
    if not cap.isOpened():
        print(f"[{camera_id}] Failed to open video stream!")
        return

    frame_buffer = deque(maxlen=100)
    consecutive_detection_count = 0
    recording = False
    post_record_frames_left = 100
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(1)
                continue

            frame_buffer.append(frame.copy())
            results = model(frame)[0]
            gas_detected = any(float(box.conf[0]) > 0.4 for box in results.boxes)

            if gas_detected:
                consecutive_detection_count += 1
            else:
                consecutive_detection_count = 0

            if consecutive_detection_count >= 3 and not recording:
                recording = True
                post_record_frames_left = 100
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"{camera_id}_{timestamp}.mp4"
                video_path = os.path.join(SAVE_VIDEO_DIR, video_filename)
                out = cv2.VideoWriter(video_path, fourcc, 10, (frame.shape[1], frame.shape[0]))
                for f in frame_buffer:
                    out.write(f)

                record = {
                    "camera": camera_id,
                    "location": location,
                    "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "video_path": video_path.replace("\\", "/")
                }
                history = []
                if os.path.exists(DETECTION_HISTORY_FILE):
                    try:
                        with open(DETECTION_HISTORY_FILE, "r") as f:
                            history = json.load(f)
                    except Exception:
                        pass
                history.append(record)
                with open(DETECTION_HISTORY_FILE, "w") as f:
                    json.dump(history, f, indent=4)

            if recording and out:
                out.write(frame)
                if not gas_detected:
                    post_record_frames_left -= 1
                    if post_record_frames_left <= 0:
                        out.release()
                        out = None
                        recording = False

            await asyncio.sleep(0.01)

    finally:
        cap.release()
        if out:
            out.release()
        print(f"[{camera_id}] Released stream.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


