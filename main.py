import cv2
import time
import threading
import json
import subprocess
import os
from ultralytics import YOLO
from datetime import datetime
import glob
import http.server
import socketserver

# Define target classes to detect
TARGET_CLASSES = ["car", "truck", "bus", "motorcycle"]
CLASS_NAME_TO_ID = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7
}

# List of cameras to process
CAMERAS = [
    {"name": "九如二路、天津街", "url": "https://cctv5.kctmc.nat.gov.tw/stream/hls/hls/Cam264.m3u8"},
    {"name": "九如二路、自立一路", "url": "https://cctv3.kctmc.nat.gov.tw/51d496c0/&t=1749968518299"},
    {"name": "自立一路、建國三路", "url": "https://cctv6.kctmc.nat.gov.tw/1dd04532/&t=1749968617874"},
    {"name": "九如二路、重慶街", "url": "https://cctv6.kctmc.nat.gov.tw/ae6689ba/&t=1749968664751"},
    {"name": "站西路", "url": "https://cctv4.kctmc.nat.gov.tw/c9653df7/&t=1749968701234"},
    {"name": "站西路、建國三路", "url": "https://cctv6.kctmc.nat.gov.tw/abc0307b/&t=1749968773923"},
    {"name": "站東路南側路口", "url": "https://cctv1.kctmc.nat.gov.tw/02f22e11/&t=1749968804433"},
    {"name": "站東路", "url": "https://cctv6.kctmc.nat.gov.tw/7f82a59c/&t=1749968871527"},
    {"name": "九如二路、松江路", "url": "https://cctv1.kctmc.nat.gov.tw/6e559e58/&t=1749968945459"},
    {"name": "建國二路、南華路", "url": "https://cctv6.kctmc.nat.gov.tw/d5ce1b72/&t=1749968985350"},
]

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def fetch_snapshot_ffmpeg(name, url):
    safe_name = name.replace("/", "-").replace(" ", "_")
    output_path = f"static/snapshots/{safe_name}.jpg"
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", url,
            "-frames:v", "1", output_path
        ], check=True, timeout=5)
        if os.path.exists(output_path):
            print(f"📸 快照成功：{name}")
            return output_path
        else:
            print(f"❌ 快照失敗（找不到圖片）：{name}")
            return None
    except Exception as e:
        print(f"❌ FFmpeg 抓取失敗：{name} ({e})")
        return None

def detect_vehicles_from_snapshot(name, snapshot_path):
    frame = cv2.imread(snapshot_path)
    if frame is None:
        print(f"❌ 無法載入快照：{snapshot_path}")
        return None

    try:
        results = model.predict(frame, classes=list(CLASS_NAME_TO_ID.values()), conf=0.4, verbose=False)[0]
        count = {cls: 0 for cls in TARGET_CLASSES}
        for r in results.boxes.cls:
            label = int(r.item())
            for cls_name, cls_id in CLASS_NAME_TO_ID.items():
                if label == cls_id:
                    count[cls_name] += 1

        print(f"✅ {name} 車輛數：{count}")
        return {"name": name, "count": count, "timestamp": int(time.time())}
    except Exception as e:
        print(f"❌ 偵測錯誤：{name} ({e})")
        return None

def update_all():
    results = []
    for cam in CAMERAS:
        snapshot_path = fetch_snapshot_ffmpeg(cam["name"], cam["url"])
        if snapshot_path:
            result = detect_vehicles_from_snapshot(cam["name"], snapshot_path)
            if result:
                results.append(result)
                continue

        # Error handling: mark vehicle count as X
        count_x = {cls: "X" for cls in TARGET_CLASSES}
        results.append({"name": cam["name"], "count": count_x, "timestamp": int(time.time())})

    os.makedirs("static/data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = f"static/data/vehicle_data_{timestamp}.json"

    # Save new data file
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Update latest.json
    latest_path = "static/data/latest.json"
    try:
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(data_path), latest_path)
    except:
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Keep only the latest JSON file
    old_files = sorted(glob.glob("static/data/vehicle_data_*.json"))
    for old in old_files[:-1]:
        os.remove(old)

    # === Snapshot latest symlink or copy ===
    snapshot_files = sorted(glob.glob("static/snapshots/*.jpg"))
    latest_snapshots_dir = "static/snapshots/latest"
    os.makedirs(latest_snapshots_dir, exist_ok=True)

    # Clear old latest snapshots
    for f in glob.glob(f"{latest_snapshots_dir}/*.jpg"):
        os.remove(f)

    # Copy or link latest snapshot images
    for snapshot in snapshot_files:
        filename = os.path.basename(snapshot)
        target_path = os.path.join(latest_snapshots_dir, filename)
        try:
            os.symlink(os.path.relpath(snapshot, latest_snapshots_dir), target_path)
        except:
            with open(snapshot, "rb") as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    print(f"✅ 已更新至 {data_path}")

def start_scheduler():
    def loop():
        while True:
            print("🔄 更新中...")
            update_all()
            time.sleep(30)

    threading.Thread(target=loop, daemon=True).start()

def start_http_server(port=80, directory="."):
    def run_server():
        os.chdir(directory)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 HTTP 伺服器已啟動於 http://localhost:{port}")
            httpd.serve_forever()

    threading.Thread(target=run_server, daemon=True).start()

if __name__ == "__main__":
    start_scheduler()
    start_http_server(port=80, directory=".")  # 建議用 8080 避免需 root
    print("✅ 車輛偵測與 Web 伺服器已啟動，按 Ctrl+C 結束")
    while True:
        time.sleep(3600)
