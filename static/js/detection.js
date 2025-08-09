document.addEventListener("DOMContentLoaded", () => {
  const cameraConfigs = [
    { id: 1, type: "local" },
    { id: 2, type: "ip", ip: "192.168.0.116", wsKey: "camera2_browser" },
    { id: 3, type: "ip", ip: "192.168.0.116", wsKey: "camera3_browser" },  // different IPs or ports
    { id: 4, type: "ip", ip: "192.168.0.116", wsKey: "camera4_browser" }
  ];

  cameraConfigs.forEach(config => setupVideoCamera(config));

  function setupVideoCamera(config) {
    const video = document.getElementById(`camera${config.id}Feed`);
    const canvas = document.getElementById(`canvasOverlay${config.id}`);
    const ctx = canvas.getContext("2d");
    const startBtn = document.getElementById(`startBtn${config.id}`);
    const stopBtn = document.getElementById(`stopBtn${config.id}`);
    const status = document.getElementById(`status${config.id}`);
    const fpsCounter = document.getElementById(`fpsCounter${config.id}`);

    let ws = null;
    let streaming = false;
    let lastFrameTime = performance.now();
    let frameRequestId = null;

    function setCanvasSize() {
      let w, h;
      if (video.tagName === "IMG") {
        w = video.naturalWidth || 640;
        h = video.naturalHeight || 480;
      } else {
        w = video.videoWidth || 640;
        h = video.videoHeight || 480;
      }
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = video.clientWidth + "px";
      canvas.style.height = video.clientHeight + "px";
    }
    
    function sendFrame() {
      if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) {
        frameRequestId = null;
        return;
      }

      const now = performance.now();
      const elapsed = now - lastFrameTime;

      if (elapsed >= 100) { // ~10 FPS
        lastFrameTime = now;
        const fps = Math.round(1000 / elapsed);
        fpsCounter.textContent = `FPS: ${fps}`;

        const tmpCanvas = document.createElement("canvas");
        let w, h;
        if (video.tagName === "IMG") {
          w = video.naturalWidth || 640;
          h = video.naturalHeight || 480;
        } else {
          w = video.videoWidth || 640;
          h = video.videoHeight || 480;
        }
        tmpCanvas.width = w;
        tmpCanvas.height = h;
        const tmpCtx = tmpCanvas.getContext("2d");

        tmpCtx.drawImage(video, 0, 0, w, h);

        const dataURL = tmpCanvas.toDataURL("image/jpeg", 0.8); // compress quality 80%
        try {
          ws.send(dataURL);
        } catch (err) {
          console.warn(`Failed to send frame for camera ${config.id}`, err);
        }
      }

      frameRequestId = requestAnimationFrame(sendFrame);
    }

    function drawDetections(detections) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let detected = false;

      detections.forEach(d => {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
        detected = true;
      });

      status.textContent = detected ? "⚠️ Gas Detected!" : "✅ No Gas Detected";
      status.className = detected ? "alert alert-danger" : "alert alert-success";
    }

    const startStreaming = () => {
      if (streaming) return;

      ws = new WebSocket(`ws://${location.host}/ws/${config.wsKey || `camera${config.id}`}`);
      streaming = true;

      ws.onopen = () => {
        if (config.type === "local") {
          navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
              video.srcObject = stream;
              video.onloadedmetadata = () => {
                setCanvasSize();
                video.play();
              };
              video.onplaying = () => {
                if (!frameRequestId) sendFrame();
              };
            })
            .catch(err => {
              console.error(`Camera ${config.id} error:`, err);
              status.textContent = "❌ Webcam error";
              status.className = "alert alert-danger";
            });
        } else if (config.type === "ip") {
          const streamURL = `http://${config.ip}:8080/video`;
          if (video.tagName === "IMG") {
            video.crossOrigin = "anonymous"; // helps avoid CORS errors if backend allows
            video.src = streamURL;
            video.onload = () => {
              setCanvasSize();
              if (!frameRequestId) sendFrame();
              status.textContent = "✅ Stream loaded";
              status.className = "alert alert-success";
            };
            video.onerror = () => {
              console.warn(`Camera ${config.id} failed to load stream`);
              status.textContent = "❌ Failed to load stream";
              status.className = "alert alert-warning";
            };
            // Sometimes `onplay` event fires for <img>, so catch that to set canvas size & start frames:
            video.onplay = () => {
              setCanvasSize();
              if (!frameRequestId) sendFrame();
            };
          } else {
            // For video tags with IP streams if you ever add them
            video.crossOrigin = "anonymous";
            video.src = streamURL;
            video.onloadedmetadata = () => {
              setCanvasSize();
              if (!frameRequestId) sendFrame();
              status.textContent = "✅ Stream loaded";
              status.className = "alert alert-success";
            };
            video.onerror = () => {
              console.warn(`Camera ${config.id} failed to load stream`);
              status.textContent = "❌ Failed to load stream";
              status.className = "alert alert-warning";
            };
          }
        }
      };

      ws.onmessage = event => {
        try {
          const detections = JSON.parse(event.data);
          drawDetections(detections);
        } catch (e) {
          console.error(`Error parsing detection data (camera ${config.id}):`, e);
        }
      };

      ws.onerror = e => {
        console.error(`WebSocket error (camera ${config.id}):`, e);
        status.textContent = "❌ WebSocket error";
        status.className = "alert alert-danger";
      };

      ws.onclose = () => {
        console.log(`WebSocket closed (camera ${config.id})`);
        ws = null;
      };

      startBtn.style.display = "none";
      stopBtn.style.display = "inline-block";
    };

    const stopStreaming = () => {
      if (!streaming) return;
      streaming = false;

      if (ws) ws.close();
      ws = null;

      if (frameRequestId) {
        cancelAnimationFrame(frameRequestId);
        frameRequestId = null;
      }

      if (config.type === "local" && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
      } else {
        video.src = "";
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      status.textContent = "✅ No Gas Detected";
      status.className = "alert alert-success";
      fpsCounter.textContent = "FPS: 0";

      startBtn.style.display = "inline-block";
      stopBtn.style.display = "none";
    };

    startBtn.onclick = startStreaming;
    stopBtn.onclick = stopStreaming;
  }
});


