# model.py
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import time
from collections import deque
import asyncio
import json

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Definitions for Model Module ---
SAVE_VIDEO_DIR = "static/saved_videos"

# --- VideoClipSaver Class ---
class VideoClipSaver:
    def __init__(self, fps=10, buffer_seconds=10, post_seconds=10, output_dir=None,
                 camera_id='unknown', camera_name='Camera', location='Location'):
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        self.post_seconds = post_seconds
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.recording = False
        self.frames_after_detection = 0

        self.output_dir = output_dir if output_dir else SAVE_VIDEO_DIR

        self.video_writer = None
        self.frame_width = None
        self.frame_height = None
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.location = location

    def update(self, frame, detection_occurred):
        if frame is None:
            return None

        if self.frame_width is None or self.frame_height is None or \
           self.frame_width != frame.shape[1] or self.frame_height != frame.shape[0]:
            self.frame_height, self.frame_width = frame.shape[:2]
            if self.video_writer and self.video_writer.isOpened():
                logging.warning(f"[{self.camera_name}] Warning: Frame dimensions changed during recording. Releasing old writer.")
                self.release()
                self.video_writer = None

        self.frame_buffer.append(frame.copy())

        filename = None
        if detection_occurred and not self.recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{self.camera_name.replace(' ', '_')}_{self.location.replace(' ', '_')}_{timestamp}.mp4"
            filepath = os.path.join(self.output_dir, filename) # ADDED THIS LINE
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for broader compatibility

            if self.frame_width <= 0 or self.frame_height <= 0:
                logging.error(f"Warning: Invalid frame dimensions ({self.frame_width}x{self.frame_height}) for {self.camera_name}. Cannot start video writer.")
                return None

            try:
                self.video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.frame_width, self.frame_height))
                if not self.video_writer.isOpened():
                    raise IOError(f"Could not open video writer at {filepath}.")
            except Exception as e:
                logging.error(f"Error initializing VideoWriter for {self.camera_name}: {e}")
                self.video_writer = None
                return None

            for buffered_frame in self.frame_buffer:
                if buffered_frame.shape[0] == self.frame_height and buffered_frame.shape[1] == self.frame_width:
                    self.video_writer.write(buffered_frame)
                else:
                    logging.warning(f"[{self.camera_name}] Skipped buffered frame with inconsistent dimensions ({buffered_frame.shape[1]}x{buffered_frame.shape[0]}).")

            self.recording = True
            self.frames_after_detection = 0
            logging.info(f"[{self.camera_name}] Started saving video clip: {filepath}")

        if self.recording and self.video_writer is not None and self.video_writer.isOpened():
            if frame.shape[0] == self.frame_height and frame.shape[1] == self.frame_width:
                self.video_writer.write(frame)
                self.frames_after_detection += 1
                if self.frames_after_detection >= self.post_seconds * self.fps:
                    self.recording = False
                    self.release()
                    self.frame_buffer.clear()
                    logging.info(f"[{self.camera_name}] Finished saving video clip.")
            else:
                logging.warning(f"[{self.camera_name}] Skipped current frame with inconsistent dimensions ({frame.shape[1]}x{frame.shape[0]}).")
        return filename

    def stop_recording(self):
        if self.recording:
            logging.info(f"[{self.camera_name}] Forced stop recording.")
        self.recording = False
        self.release()
        self.frame_buffer.clear()

    def release(self):
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            logging.info(f"[{self.camera_name}] VideoWriter released.")

# --- SmokeDetectionService Class ---
class SmokeDetectionService:
    def __init__(self):
        MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "best_yolo11_final.pt"))
        logging.info(f"Attempting to load YOLO model from: {MODEL_PATH}")
        try:
            self.model = YOLO(MODEL_PATH, task="detect")
            logging.info("YOLO model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading YOLO model from {MODEL_PATH}: {e}")
            self.model = None

        if self.model and self.model.device.type != 'cpu':
            try:
                self.model.model.half()
                logging.info("YOLO model converted to half precision (FP16) for faster inference.")
            except Exception as e:
                logging.warning(f"Could not convert YOLO model to half precision: {e}. Falling back to full precision.")
        elif self.model:
            logging.info("YOLO model running on CPU. Half precision not applicable.")

        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.target_fps = 10
        self.processing_width = 416
        self.processing_height = 416

        self.streak_timeout_frames = self.target_fps * 2
        self.min_detection_streak = 2

        self.frame_skip_count = {}
        self.frame_skip_interval = 2 # Process every 2nd frame for IP cameras

        self.processing_lock = asyncio.Lock()

        self.clip_savers = {}
        self.detection_streaks = {}
        self.streak_timers = {}

        self.camera_configs = {
            "1": {"name": "PC Webcam", "location": "Room 1", "type": "webcam", "url": None},
            "2": {"name": "Phone Camera 1", "location": "Hallway", "type": "ip", "url": "http://10.250.82.123:8080/video", "stream_task": None, "status": "Stopped", "video_saver": None},
            "3": {"name": "Phone Camera 2", "location": "Kitchen", "type": "ip", "url": "http://192.168.11.138:8080/video", "stream_task": None, "status": "Stopped", "video_saver": None},
            "4": {"name": "Phone Camera 3", "location": "Garage", "type": "ip", "url": "http://10.250.82.125:8080/video", "stream_task": None, "status": "Stopped", "video_saver": None},
        }

        for cam_id, config in self.camera_configs.items():
            clip_saver_instance = VideoClipSaver(
                fps=self.target_fps,
                buffer_seconds=5,
                post_seconds=5,
                output_dir=SAVE_VIDEO_DIR,
                camera_id=cam_id,
                camera_name=config["name"],
                location=config["location"]
            )
            self.clip_savers[cam_id] = clip_saver_instance
            config["video_saver"] = clip_saver_instance

            self.detection_streaks[cam_id] = 0
            self.streak_timers[cam_id] = 0
            self.frame_skip_count[cam_id] = 0

    async def process_and_send_results(self, websocket, camera_id: str, frame: np.ndarray):
        if frame is None or self.model is None:
            if self.model is None:
                logging.error(f"[{camera_id}] Model not loaded. Cannot process frame.")
                await websocket.send_json({"camera_id": camera_id, "status": "Error: Model not loaded"})
            return

        async with self.processing_lock:
            loop = asyncio.get_event_loop()

            frame_processed = await loop.run_in_executor(
                self.thread_pool, self._preprocess_frame_optimized, frame
            )

            try:
                results = await loop.run_in_executor(
                    self.thread_pool,
                    self._run_yolo_inference,
                    frame_processed
                )
            except Exception as e:
                logging.error(f"[{camera_id}] YOLO inference error: {e}")
                return

            current_frame_has_detection = False
            boxes = []
            confidences = []

            if results and len(results) > 0:
                annotated_frame = results[0].plot()

                for result in results:
                    if result.boxes is not None and len(result.boxes.data) > 0:
                        for det in result.boxes.data.tolist():
                            x1, y1, x2, y2, conf, cls = det
                            if conf > 0.5:
                                current_frame_has_detection = True
                                boxes.append({
                                    "x1": int(x1), "y1": int(y1),
                                    "x2": int(x2), "y2": int(y2)
                                })
                                confidences.append(float(conf))
            else:
                annotated_frame = frame_processed


            confirmed_detection = False
            if current_frame_has_detection:
                self.detection_streaks[camera_id] = self.detection_streaks.get(camera_id, 0) + 1
                self.streak_timers[camera_id] = 0
                if self.detection_streaks[camera_id] >= self.min_detection_streak:
                    confirmed_detection = True
            else:
                self.streak_timers[camera_id] = self.streak_timers.get(camera_id, 0) + 1
                if self.streak_timers[camera_id] >= self.streak_timeout_frames:
                    self.detection_streaks[camera_id] = 0
                    self.streak_timers[camera_id] = 0

            clip_saver = self.clip_savers.get(camera_id)
            video_filename = None
            if clip_saver:
                video_filename = await loop.run_in_executor(
                    self.thread_pool, clip_saver.update, frame_processed, confirmed_detection
                )
                if video_filename:
                    asyncio.create_task(self.log_detection_async(camera_id, video_filename, confirmed_detection))

            encoded_frame = await loop.run_in_executor(
                self.thread_pool, self._encode_frame_to_base64_optimized, annotated_frame
            )
            frame_to_display = f"data:image/jpeg;base64,{encoded_frame}"

            await websocket.send_json({
                "camera_id": camera_id,
                "boxes": boxes if confirmed_detection else [],
                "confidences": confidences if confirmed_detection else [],
                "smoke_detected": confirmed_detection,
                "frame_with_overlay": frame_to_display
            })

    def _preprocess_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        frame_resized = cv2.resize(frame, (self.processing_width, self.processing_height), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if lap_var > 300:
            return cv2.GaussianBlur(frame_resized, (3, 3), 0)
        else:
            return frame_resized

    def _run_yolo_inference(self, frame: np.ndarray):
        if self.model is None:
            logging.error("YOLO model is not loaded, skipping inference.")
            return None
        try:
            results = self.model(
                frame,
                imgsz=self.processing_width,
                conf=0.5,
                iou=0.45,
                verbose=False,
                half=True if self.model.device.type != 'cpu' else False,
                device=self.model.device
            )
            return results
        except Exception as e:
            logging.error(f"Error during YOLO inference: {e}")
            return None

    def _encode_frame_to_base64_optimized(self, frame: np.ndarray) -> str:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return base64.b64encode(buffer).decode('utf-8')

    async def log_detection_async(self, camera_id: str, filename: str, confirmed: bool):
        log_data = {
            "camera": self.clip_savers[camera_id].camera_name,
            "location": self.clip_savers[camera_id].location,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_file": filename,
            "confirmed_detection": confirmed
        }

        history_file = os.path.join(SAVE_VIDEO_DIR, "detection_history.json")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, self._write_log_to_file, history_file, log_data)

    def _write_log_to_file(self, filepath: str, log_entry: dict):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            history = []
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                with open(filepath, "r") as f:
                    try:
                        history = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning(f"Existing detection history file '{filepath}' is corrupted. Starting new history.")
                        history = []
            history.append(log_entry)

            with open(filepath, "w") as f:
                json.dump(history, f, indent=4)
            logging.info(f"[{log_entry['camera']}] Logged detection: {log_entry['video_file']}")
        except Exception as e:
            logging.error(f"Error saving log file '{filepath}': {e}")

    def cleanup(self):
        logging.info("Cleaning up SmokeDetectionService resources...")
        for camera_id, clip_saver in self.clip_savers.items():
            if clip_saver.recording:
                clip_saver.stop_recording()
            elif clip_saver.video_writer and clip_saver.video_writer.isOpened():
                clip_saver.release()
        self.thread_pool.shutdown(wait=True)
        logging.info("SmokeDetectionService resources cleaned up.")

# --- Asynchronous IP Camera Streaming Function ---
async def stream_ip_camera_frames(websocket, camera_id: str, url: str, detection_service_instance: SmokeDetectionService):
    logging.info(f"[{camera_id}] Starting IP camera streaming from URL: {url}")
    session = aiohttp.ClientSession()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), ssl=False) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "multipart/x-mixed-replace" in content_type:
                boundary = content_type.split('boundary=')[1].strip().encode()
                buffer = b''
                last_frame_time = time.time()
                target_frame_duration = 1.0 / detection_service_instance.target_fps

                async for chunk in response.content.iter_any():
                    if asyncio.current_task().cancelled():
                        logging.info(f"[{camera_id}] Stream task was cancelled during chunk read.")
                        break

                    buffer += chunk

                    parts = buffer.split(b'--' + boundary)
                    if len(parts) > 1:
                        for i in range(len(parts) - 1):
                            image_data_raw = parts[i]
                            if b'\r\n\r\n' in image_data_raw:
                                header, image_bytes = image_data_raw.split(b'\r\n\r\n', 1)
                                if image_bytes:
                                    detection_service_instance.frame_skip_count[camera_id] = detection_service_instance.frame_skip_count.get(camera_id, 0) + 1
                                    if detection_service_instance.frame_skip_count[camera_id] % detection_service_instance.frame_skip_interval != 0:
                                        continue

                                    np_arr = np.frombuffer(image_bytes, np.uint8)
                                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                                    if frame is not None:
                                        current_time = time.time()
                                        elapsed_time = current_time - last_frame_time
                                        if elapsed_time < target_frame_duration:
                                            await asyncio.sleep(target_frame_duration - elapsed_time)
                                        last_frame_time = time.time()

                                        await detection_service_instance.process_and_send_results(websocket, camera_id, frame)
                                    else:
                                        logging.warning(f"[{camera_id}] Failed to decode image from MJPEG chunk.")

                        buffer = parts[-1]

                    await asyncio.sleep(0.001)

            elif "image/jpeg" in content_type or "image/png" in content_type:
                logging.info(f"[{camera_id}] Detected single image endpoint. Polling for images.")
                while True:
                    if asyncio.current_task().cancelled():
                        logging.info(f"[{camera_id}] Stream task was cancelled during single image fetch.")
                        break

                    try:
                        detection_service_instance.frame_skip_count[camera_id] = detection_service_instance.frame_skip_count.get(camera_id, 0) + 1
                        if detection_service_instance.frame_skip_count[camera_id] % detection_service_instance.frame_skip_interval != 0:
                            await asyncio.sleep(1.0 / detection_service_instance.target_fps)
                            continue

                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5), ssl=False) as img_resp:
                            img_resp.raise_for_status()
                            image_bytes = await img_resp.read()
                            np_arr = np.frombuffer(image_bytes, np.uint8)
                            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            await detection_service_instance.process_and_send_results(websocket, camera_id, frame)
                        else:
                            logging.warning(f"[{camera_id}] Failed to decode image from single image endpoint.")

                        await asyncio.sleep(1.0 / detection_service_instance.target_fps)
                    except asyncio.CancelledError:
                        raise
                    except aiohttp.ClientError as e:
                        logging.error(f"[{camera_id}] Error fetching single image from {url}: {e}. Retrying in 3s...")
                        await websocket.send_json({
                            "camera_id": camera_id,
                            "status": f"Connection Error for camera {camera_id}: {e}. Retrying in 3s."
                        })
                        await asyncio.sleep(3)
            else:
                logging.error(f"[{camera_id}] Unsupported content type for IP camera: {content_type} at {url}")
                await websocket.send_json({
                    "camera_id": camera_id,
                    "status": f"Unsupported stream type for camera {camera_id}: {content_type}"
                })

    except asyncio.CancelledError:
        logging.info(f"[{camera_id}] IP camera stream task cancelled gracefully.")
        await websocket.send_json({
            "camera_id": camera_id,
            "status": f"IP camera {camera_id} stream stopped (Backend)."
        })
    except aiohttp.ClientError as e:
        logging.error(f"[{camera_id}] Connection error for IP Camera {url}: {e}")
        await websocket.send_json({
            "camera_id": camera_id,
            "status": f"Connection Error for camera {camera_id}: {e}. Please check URL."
        })
    except Exception as e:
        logging.critical(f"[{camera_id}] An unexpected error occurred during IP camera streaming: {e}", exc_info=True)
        await websocket.send_json({
            "camera_id": camera_id,
            "status": f"An unexpected error occurred for camera {camera_id}: {e}. Please check server logs."
        })
    finally:
        await session.close()
        logging.info(f"[{camera_id}] IP camera streaming session closed.")