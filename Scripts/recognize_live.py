# """
# recognize_live.py
# ─────────────────────────────────────────────────────────────────
# Live CCTV / webcam face recognition — optimized version of your
# original video-file script, adapted for real-time streams.

# Changes from your original:
#   - FrameGrabber thread keeps the camera buffer empty (no lag)
#   - Recognition runs on the latest frame, not a stale buffered one
#   - Auto-reconnects if the RTSP stream drops
#   - Cooldown prevents the same person being printed 100x per second
#   - Everything else (InsightFace, .pkl loading, matching) is identical
# """

# import cv2
# import pickle
# import numpy as np
# from insightface.app import FaceAnalysis
# import glob
# import os
# import threading
# import queue
# import time

# print("=" * 70)
# print("LIVE CCTV — INSIGHTFACE FACE RECOGNITION")
# print("=" * 70)

# # ──────────────────────────────────────────────
# # CONFIG  ← change these
# # ──────────────────────────────────────────────

# # Camera source:
# #   0 or 1          → USB / laptop webcam (good for testing)
# #   "rtsp://..."    → real CCTV camera
# #
# # Common RTSP formats:
# #   Hikvision : rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
# #   Dahua     : rtsp://admin:password@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0
# #   TP-Link   : rtsp://admin:password@192.168.1.64:554/stream1
# #   Generic   : rtsp://admin:password@192.168.1.64:554/live

# CAMERA_SOURCE    = 0                  # ← replace with your RTSP URL for real CCTV
# STUDENTS_FOLDER  = r"C:\Users\SWISS TECH\Documents\Maaz\attendence"
# THRESHOLD        = 0.55                # lower = stricter matching
# COOLDOWN_SECONDS = 10                  # seconds before same person is printed again
# PROCESS_EVERY_N  = 3                   # run InsightFace every Nth frame (speed vs accuracy)
# SHOW_PREVIEW     = True                # set False on headless server

# # ──────────────────────────────────────────────
# # STEP 1 — Load InsightFace
# # ──────────────────────────────────────────────
# print("\n[1/3] Loading InsightFace model...")
# face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# face_app.prepare(ctx_id=-1, det_size=(640, 640))
# print("✓ InsightFace loaded")

# # ──────────────────────────────────────────────
# # STEP 2 — Load students from .pkl files
# # ──────────────────────────────────────────────
# print("\n[2/3] Loading students...")
# students = []
# pkl_files = glob.glob(os.path.join(STUDENTS_FOLDER, "student_*.pkl"))

# if not pkl_files:
#     print("ERROR: No student .pkl files found!")
#     exit()

# for pkl_file in pkl_files:
#     with open(pkl_file, 'rb') as f:
#         students.append(pickle.load(f))

# print(f"✓ Loaded {len(students)} students")
# for s in students:
#     print(f"  - {s['name']} ({s['student_id']})")

# # ──────────────────────────────────────────────
# # STEP 3 — Matching function (same as yours)
# # ──────────────────────────────────────────────
# def find_match(embedding, threshold=THRESHOLD):
#     embedding = embedding / np.linalg.norm(embedding)
#     best_match = None
#     best_distance = float('inf')

#     for student in students:
#         stored_list = student.get('embeddings', [student.get('embedding')])
#         for stored_emb in stored_list:
#             stored_emb = stored_emb / np.linalg.norm(stored_emb)
#             distance = np.linalg.norm(embedding - stored_emb)
#             if distance < best_distance:
#                 best_distance = distance
#                 best_match = student

#     if best_distance < threshold:
#         confidence = 1 - (best_distance / 2)
#         return best_match, confidence, best_distance

#     return None, 0, best_distance

# # ──────────────────────────────────────────────
# # FRAME GRABBER THREAD
# # Reads frames as fast as possible, keeps only the latest.
# # This is what prevents the "processing old frames" problem.
# # ──────────────────────────────────────────────
# frame_queue = queue.Queue(maxsize=1)   # size=1 → always the freshest frame
# grabber_running = True

# def frame_grabber():
#     global grabber_running
#     cap = cv2.VideoCapture(CAMERA_SOURCE)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize OpenCV's internal buffer

#     if not cap.isOpened():
#         print("ERROR: Cannot open camera source!")
#         grabber_running = False
#         return

#     print(f"\n✓ Camera opened: {CAMERA_SOURCE}")

#     while grabber_running:
#         ret, frame = cap.read()

#         if not ret:
#             print("⚠ Frame read failed — reconnecting in 3s...")
#             cap.release()
#             time.sleep(3)
#             cap = cv2.VideoCapture(CAMERA_SOURCE)
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             continue

#         # Drop the old frame if the queue is full, always keep the latest
#         if frame_queue.full():
#             try:
#                 frame_queue.get_nowait()
#             except queue.Empty:
#                 pass

#         frame_queue.put(frame)

#     cap.release()

# # ──────────────────────────────────────────────
# # MAIN RECOGNITION LOOP
# # ──────────────────────────────────────────────
# print("\n[3/3] Starting camera...")
# grabber_thread = threading.Thread(target=frame_grabber, daemon=True)
# grabber_thread.start()

# # Wait until first frame arrives
# time.sleep(2)
# if not grabber_running:
#     exit()

# print("\n" + "=" * 70)
# print("LIVE RECOGNITION RUNNING  (press Q to quit)")
# print("=" * 70)

# frame_count      = 0
# total_detections = 0
# total_recognized = 0
# last_seen        = {}    # student_id → last print timestamp (for cooldown)

# while True:
#     # Get the latest frame (block up to 2 seconds)
#     try:
#         frame = frame_queue.get(timeout=2.0)
#     except queue.Empty:
#         print("⚠ No frames received for 2s...")
#         continue

#     frame_count += 1

#     # ── Only run InsightFace every Nth frame ──────────────────────────────
#     if frame_count % PROCESS_EVERY_N != 0:
#         if SHOW_PREVIEW:
#             cv2.imshow("Live Attendance", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         continue

#     # ── Detect + embed all faces in the frame ─────────────────────────────
#     try:
#         faces = face_app.get(frame)
#     except Exception as e:
#         print(f"InsightFace error: {e}")
#         continue

#     total_detections += len(faces)

#     for face in faces:
#         if face.det_score < 0.6:      # skip low-confidence detections
#             continue

#         bbox = face.bbox.astype(int)
#         x1, y1, x2, y2 = bbox
#         embedding = face.embedding

#         match, confidence, distance = find_match(embedding)

#         if match:
#             total_recognized += 1
#             color = (0, 255, 0)   # green
#             label = f"{match['name']} ({confidence * 100:.0f}%)"

#             # Cooldown: only print to console once per COOLDOWN_SECONDS
#             sid  = match['student_id']
#             now  = time.time()
#             last = last_seen.get(sid, 0)

#             if now - last >= COOLDOWN_SECONDS:
#                 last_seen[sid] = now
#                 print(f"\n✓ RECOGNIZED: {match['name']}")
#                 print(f"  ID         : {match['student_id']}")
#                 print(f"  Confidence : {confidence * 100:.1f}%")
#                 print(f"  Distance   : {distance:.4f}")
#                 print(f"  Frame      : {frame_count}")
#         else:
#             color = (0, 0, 255)   # red
#             label = f"Unknown ({distance:.2f})"

#         # Draw box and label
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(label) * 11, y1), color, -1)
#         cv2.putText(frame, label, (x1 + 4, y1 - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#     # Stats overlay
#     stats = f"Frame: {frame_count}  |  Detected: {total_detections}  |  Recognized: {total_recognized}"
#     cv2.putText(frame, stats, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

#     if SHOW_PREVIEW:
#         cv2.imshow("Live Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # ── Cleanup ───────────────────────────────────────────────────────────────
# grabber_running = False
# cv2.destroyAllWindows()

# print("\n" + "=" * 70)
# print("SESSION SUMMARY")
# print("=" * 70)
# print(f"Total frames read  : {frame_count}")
# print(f"Frames processed   : {frame_count // PROCESS_EVERY_N}")
# print(f"Total detections   : {total_detections}")
# print(f"Total recognized   : {total_recognized}")
# print("=" * 70)



# import cv2
# import pickle
# import numpy as np
# from insightface.app import FaceAnalysis
# import glob
# import os
# import threading
# from collections import deque
# import time

# print("="*70)
# print("LIVE CCTV FACE RECOGNITION - OPTIMIZED")
# print("="*70)

# # ─────────────────────────────────────────────
# # CONFIG - tune these for your hardware
# # ─────────────────────────────────────────────
# THRESHOLD       = 1.0
# DET_SIZE        = (320, 320)   # ← smaller = faster (was 640x640)
# PROCESS_EVERY_N = 3            # ← process 1 in every N frames
# FRAME_SCALE     = 0.5          # ← resize frame before detection (0.5 = half)
# CAMERA_SOURCE   = r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-19 at 3.26.03 PM.mp4"           # ← 0 for webcam, or RTSP URL for CCTV
# # CAMERA_SOURCE = "rtsp://user:pass@192.168.1.100:554/stream"
# # ─────────────────────────────────────────────

# # 1. LOAD INSIGHTFACE
# print("\n[1/3] Loading InsightFace...")
# face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
# print("✓ InsightFace loaded")

# # 2. LOAD STUDENTS
# print("\n[2/3] Loading students...")
# students = []
# students_folder = r"C:\Users\SWISS TECH\Documents\Maaz\attendence"
# pkl_files = glob.glob(os.path.join(students_folder, "student_*.pkl"))

# if not pkl_files:
#     print("ERROR: No student .pkl files found!")
#     exit()

# for pkl_file in pkl_files:
#     with open(pkl_file, 'rb') as f:
#         students.append(pickle.load(f))

# # Pre-normalize all stored embeddings ONCE at startup (big speedup)
# for student in students:
#     if 'embeddings' in student:
#         student['embeddings'] = [e / np.linalg.norm(e) for e in student['embeddings']]
#         student['embeddings_matrix'] = np.array(student['embeddings'])  # for batch distance
#     else:
#         student['embedding'] = student['embedding'] / np.linalg.norm(student['embedding'])

# print(f"✓ Loaded {len(students)} students")

# # 3. FAST MATCHING using vectorized numpy (no loops over students)
# def find_match(embedding, threshold=THRESHOLD):
#     embedding = embedding / np.linalg.norm(embedding)
#     best_match = None
#     best_distance = float('inf')

#     for student in students:
#         if 'embeddings_matrix' in student:
#             # Vectorized: compute all distances in one shot
#             diffs = student['embeddings_matrix'] - embedding
#             distances = np.linalg.norm(diffs, axis=1)
#             min_dist = distances.min()
#         else:
#             min_dist = np.linalg.norm(student['embedding'] - embedding)

#         if min_dist < best_distance:
#             best_distance = min_dist
#             best_match = student

#     if best_distance < threshold:
#         confidence = 1 - (best_distance / 2)
#         return best_match, confidence, best_distance
#     return None, 0, best_distance


# # ─────────────────────────────────────────────
# # THREADED FRAME READER
# # Reads frames in background so inference never waits on camera I/O
# # ─────────────────────────────────────────────
# class CameraStream:
#     def __init__(self, source):
#         self.cap = cv2.VideoCapture(source)
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
#         self.frame = None
#         self.running = True
#         self.lock = threading.Lock()
#         self.thread = threading.Thread(target=self._reader, daemon=True)
#         self.thread.start()

#     def _reader(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 with self.lock:
#                     self.frame = frame

#     def read(self):
#         with self.lock:
#             return self.frame.copy() if self.frame is not None else None

#     def stop(self):
#         self.running = False
#         self.cap.release()


# # ─────────────────────────────────────────────
# # THREADED INFERENCE WORKER
# # Runs face detection + recognition in a separate thread
# # So display never blocks on heavy inference
# # ─────────────────────────────────────────────
# class InferenceWorker:
#     def __init__(self):
#         self.input_frame = None
#         self.output_faces = []   # list of (bbox, label, color)
#         self.running = True
#         self.lock = threading.Lock()
#         self.new_frame_event = threading.Event()
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def submit(self, frame):
#         with self.lock:
#             self.input_frame = frame
#         self.new_frame_event.set()

#     def get_results(self):
#         with self.lock:
#             return list(self.output_faces)

#     def _worker(self):
#         while self.running:
#             self.new_frame_event.wait()
#             self.new_frame_event.clear()

#             with self.lock:
#                 frame = self.input_frame.copy() if self.input_frame is not None else None

#             if frame is None:
#                 continue

#             # Resize for faster detection
#             small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
#             faces = face_app.get(small)

#             results = []
#             for face in faces:
#                 # Scale bbox back to original size
#                 bbox = (face.bbox / FRAME_SCALE).astype(int)
#                 x1, y1, x2, y2 = bbox

#                 match, confidence, distance = find_match(face.embedding)

#                 if match:
#                     color = (0, 255, 0)
#                     label = f"{match['name']} {confidence*100:.0f}%"
#                 else:
#                     color = (0, 0, 255)
#                     label = f"Unknown {distance:.2f}"

#                 results.append((x1, y1, x2, y2, label, color))

#             with self.lock:
#                 self.output_faces = results

#     def stop(self):
#         self.running = False
#         self.new_frame_event.set()


# # ─────────────────────────────────────────────
# # MAIN LOOP
# # ─────────────────────────────────────────────
# print("\n[3/3] Starting live stream...")
# print("Press Q to quit\n")

# stream   = CameraStream(CAMERA_SOURCE)
# worker   = InferenceWorker()

# frame_count  = 0
# fps_counter  = 0
# fps_display  = 0
# fps_timer    = time.time()

# time.sleep(1)  # Let camera warm up

# while True:
#     frame = stream.read()
#     if frame is None:
#         continue

#     frame_count  += 1
#     fps_counter  += 1

#     # Submit every Nth frame for inference (non-blocking)
#     if frame_count % PROCESS_EVERY_N == 0:
#         worker.submit(frame)

#     # Draw last known results on current frame (always fresh display)
#     faces = worker.get_results()
#     for (x1, y1, x2, y2, label, color) in faces:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         label_w = len(label) * 11
#         cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_w, y1), color, -1)
#         cv2.putText(frame, label, (x1 + 4, y1 - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#     # FPS counter
#     elapsed = time.time() - fps_timer
#     if elapsed >= 1.0:
#         fps_display = fps_counter / elapsed
#         fps_counter = 0
#         fps_timer   = time.time()

#     cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.putText(frame, f"Faces: {len(faces)}", (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     cv2.imshow("CCTV Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# stream.stop()
# worker.stop()
# cv2.destroyAllWindows()
# print("\n✓ Stream closed.")



import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import glob
import os
import threading
import time
import faiss

print("="*70)
print("LIVE CCTV FACE RECOGNITION - OPTIMIZED")
print("="*70)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
THRESHOLD       = 1.0
DET_SIZE        = (320, 320)
PROCESS_EVERY_N = 3
FRAME_SCALE     = 0.5
CAMERA_SOURCE   = r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-26 at 1.53.27 PM.mp4"
IS_VIDEO_FILE   = not (isinstance(CAMERA_SOURCE, int) or CAMERA_SOURCE.startswith("rtsp"))
# ─────────────────────────────────────────────

# 1. LOAD INSIGHTFACE
print("\n[1/3] Loading InsightFace...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
print("✓ InsightFace loaded")



# 2. LOAD FAISS INDEX
print("\n[2/3] Loading FAISS index...")

index = faiss.read_index("face_index.faiss")

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

print(f"✓ FAISS loaded with {index.ntotal} embeddings")



def find_match(embedding):
    embedding = embedding / np.linalg.norm(embedding)
    embedding = np.expand_dims(embedding.astype('float32'), axis=0)

    distances, indices = index.search(embedding, 1)

    best_distance = distances[0][0]
    best_index = indices[0][0]

    if best_distance < THRESHOLD:
        student_id = labels[best_index]
        confidence = 1 - (best_distance / 2)

        return {
            "student_id": student_id,
            "name": student_id   # change later if you store real names
        }, confidence, best_distance

    return None, 0, best_distance

# ─────────────────────────────────────────────
# CAMERA STREAM
# For video files: respects FPS timing so playback is smooth
# For live camera: reads as fast as possible (no buffering lag)
# ─────────────────────────────────────────────
class CameraStream:
    def __init__(self, source, is_file=False):
        self.cap = cv2.VideoCapture(source)
        self.is_file = is_file
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.frame_available = threading.Event()

        if is_file:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1/30
            print(f"✓ Video file detected — playback FPS: {self.fps:.1f}")
        else:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("✓ Live camera detected")

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            t_start = time.time()

            ret, frame = self.cap.read()

            if not ret:
                if self.is_file:
                    print("\n[Stream] End of video.")
                self.running = False
                break

            with self.lock:
                self.frame = frame
            self.frame_available.set()

            # For video files: sleep to match original FPS
            # This prevents the thread racing ahead and dropping frames
            if self.is_file:
                elapsed = time.time() - t_start
                sleep_time = self.frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def read(self):
        # Block until a new frame is actually available
        self.frame_available.wait(timeout=2.0)
        self.frame_available.clear()
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        self.frame_available.set()  # Unblock any waiting read()
        self.cap.release()


# ─────────────────────────────────────────────
# INFERENCE WORKER (unchanged, runs in background thread)
# ─────────────────────────────────────────────
class InferenceWorker:
    def __init__(self):
        self.input_frame = None
        self.output_faces = []
        self.running = True
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.input_frame = frame
        self.new_frame_event.set()

    def get_results(self):
        with self.lock:
            return list(self.output_faces)

    def _worker(self):
        while self.running:
            self.new_frame_event.wait()
            self.new_frame_event.clear()

            with self.lock:
                frame = self.input_frame.copy() if self.input_frame is not None else None

            if frame is None:
                continue

            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            faces = face_app.get(small)

            results = []
            for face in faces:
                bbox = (face.bbox / FRAME_SCALE).astype(int)
                x1, y1, x2, y2 = bbox
                match, confidence, distance = find_match(face.embedding)

                if match:
                    color = (0, 255, 0)
                    label = f"{match['name']} {confidence*100:.0f}%"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown {distance:.2f}"

                results.append((x1, y1, x2, y2, label, color))

            with self.lock:
                self.output_faces = results

    def stop(self):
        self.running = False
        self.new_frame_event.set()


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("\n[3/3] Starting stream...")
print("Press Q to quit\n")

stream = CameraStream(CAMERA_SOURCE, is_file=IS_VIDEO_FILE)
worker = InferenceWorker()

frame_count = 0
fps_counter = 0
fps_display = 0
fps_timer   = time.time()

time.sleep(1)  # Let camera/file warm up

while stream.is_running():
    frame = stream.read()
    if frame is None:
        break

    frame_count += 1
    fps_counter += 1

    # Submit every Nth frame for inference (non-blocking)
    if frame_count % PROCESS_EVERY_N == 0:
        worker.submit(frame)

    # Draw last known face results on current frame
    faces = worker.get_results()
    for (x1, y1, x2, y2, label, color) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_w = len(label) * 11
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # FPS overlay
    elapsed = time.time() - fps_timer
    if elapsed >= 1.0:
        fps_display = fps_counter / elapsed
        fps_counter = 0
        fps_timer   = time.time()

    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("CCTV Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
worker.stop()
cv2.destroyAllWindows()
print("\n✓ Done.")

# import cv2
# import pickle
# import numpy as np
# from insightface.app import FaceAnalysis
# import threading
# import time
# import faiss
# from deep_sort_realtime.deepsort_tracker import DeepSort

# print("=" * 70)
# print("LIVE CCTV FACE RECOGNITION - OPTIMIZED WITH DEEPSORT + ARCFACE EMBEDDINGS")
# print("=" * 70)

# # ─────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────
# THRESHOLD        = 1.0
# DET_SIZE         = (320, 320)
# PROCESS_EVERY_N  = 3
# FRAME_SCALE      = 0.5
# CAMERA_SOURCE    = r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-19 at 3.26.03 PM.mp4"
# IS_VIDEO_FILE    = not (isinstance(CAMERA_SOURCE, int) or CAMERA_SOURCE.startswith("rtsp"))
# RECOG_INTERVAL   = 1.0   # seconds between re-identification per track
# # ─────────────────────────────────────────────

# # 1. LOAD INSIGHTFACE
# print("\n[1/3] Loading InsightFace...")
# face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
# print("✓ InsightFace loaded")

# # 2. LOAD FAISS INDEX
# print("\n[2/3] Loading FAISS index...")
# index = faiss.read_index("face_index.faiss")
# with open("labels.pkl", "rb") as f:
#     labels = pickle.load(f)
# print(f"✓ FAISS loaded with {index.ntotal} embeddings")

# # 3. INIT DEEPSORT
# # embedder=None → disables internal PyTorch embedder (no pkg_resources crash)
# # IMPORTANT: when embedder=None, you MUST ALWAYS pass embeds= in update_tracks()
# #            even when the detections list is empty (pass embeds=[] in that case)
# print("\n[3/3] Initializing DeepSORT tracker...")
# tracker = DeepSort(
#     max_age=30,
#     n_init=3,
#     nms_max_overlap=1.0,
#     embedder=None,    # we supply ArcFace embeddings ourselves
#     half=False,
#     bgr=True,
# )
# print("✓ DeepSORT initialized")

# # ─────────────────────────────────────────────
# # MATCHING FUNCTION
# # ─────────────────────────────────────────────
# def find_match(embedding):
#     """Search FAISS index for the closest face embedding."""
#     norm = np.linalg.norm(embedding)
#     if norm == 0:
#         return None, 0.0, float('inf')
#     embedding = (embedding / norm).astype('float32')
#     query = np.expand_dims(embedding, axis=0)
#     distances, indices = index.search(query, 1)
#     best_distance = distances[0][0]
#     best_index    = indices[0][0]
#     if best_distance < THRESHOLD:
#         student_id = labels[best_index]
#         confidence = 1.0 - (best_distance / 2.0)
#         return {"student_id": student_id, "name": student_id}, confidence, best_distance
#     return None, 0.0, best_distance

# # ─────────────────────────────────────────────
# # CAMERA STREAM
# # ─────────────────────────────────────────────
# class CameraStream:
#     def __init__(self, source, is_file=False):
#         self.cap     = cv2.VideoCapture(source)
#         self.is_file = is_file
#         self.frame   = None
#         self.running = True
#         self.lock    = threading.Lock()
#         self.frame_available = threading.Event()

#         if is_file:
#             self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
#             self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1 / 30
#             print(f"✓ Video file detected — playback FPS: {self.fps:.1f}")
#         else:
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             print("✓ Live camera detected")

#         self.thread = threading.Thread(target=self._reader, daemon=True)
#         self.thread.start()

#     def _reader(self):
#         while self.running:
#             t_start = time.time()
#             ret, frame = self.cap.read()
#             if not ret:
#                 if self.is_file:
#                     print("\n[Stream] End of video.")
#                 self.running = False
#                 break
#             with self.lock:
#                 self.frame = frame
#             self.frame_available.set()
#             if self.is_file:
#                 elapsed    = time.time() - t_start
#                 sleep_time = self.frame_delay - elapsed
#                 if sleep_time > 0:
#                     time.sleep(sleep_time)

#     def read(self):
#         self.frame_available.wait(timeout=2.0)
#         self.frame_available.clear()
#         with self.lock:
#             return self.frame.copy() if self.frame is not None else None

#     def is_running(self):
#         return self.running

#     def stop(self):
#         self.running = False
#         self.frame_available.set()
#         self.cap.release()

# # ─────────────────────────────────────────────
# # INFERENCE WORKER
# # ─────────────────────────────────────────────
# class InferenceWorker:
#     def __init__(self):
#         self.input_frame     = None
#         self.output_faces    = []
#         self.running         = True
#         self.lock            = threading.Lock()
#         self.new_frame_event = threading.Event()
#         # Per-track recognition cache: {track_id: (match, confidence, timestamp)}
#         self.track_cache     = {}
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def submit(self, frame):
#         with self.lock:
#             self.input_frame = frame
#         self.new_frame_event.set()

#     def get_results(self):
#         with self.lock:
#             return list(self.output_faces)

#     def _worker(self):
#         while self.running:
#             self.new_frame_event.wait()
#             self.new_frame_event.clear()

#             with self.lock:
#                 frame = self.input_frame.copy() if self.input_frame is not None else None
#             if frame is None:
#                 continue

#             h_fr, w_fr = frame.shape[:2]

#             # ── Step 1: Detect faces on downscaled frame ──────────────────────
#             small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
#             faces = face_app.get(small)

#             # ── Step 2: Build detection + embedding lists ─────────────────────
#             # DeepSORT format (when embedder=None):
#             #   raw_dets : [([left, top, w, h], confidence, detection_class), ...]
#             #              detection_class is just a label string — NOT the embedding
#             #   embeds   : [np.array(512,), ...]  passed separately via embeds= kwarg
#             #
#             # CRITICAL: len(embeds) MUST equal len(raw_dets) at all times.
#             # CRITICAL: even when raw_dets=[], you must pass embeds=[] explicitly.
#             raw_dets = []
#             embeds   = []

#             for face in faces:
#                 x1, y1, x2, y2 = (face.bbox / FRAME_SCALE).astype(int)
#                 x1 = max(0, x1);    y1 = max(0, y1)
#                 x2 = min(w_fr, x2); y2 = min(h_fr, y2)
#                 w  = max(1, x2 - x1)
#                 h  = max(1, y2 - y1)

#                 conf    = float(face.det_score)
#                 raw_emb = face.embedding
#                 norm    = np.linalg.norm(raw_emb)
#                 emb     = (raw_emb / norm).astype('float32') if norm > 0 else raw_emb.astype('float32')

#                 # 3rd element is detection_class (string), NOT the embedding
#                 raw_dets.append(([x1, y1, w, h], conf, "face"))
#                 embeds.append(emb)

#             # ── Step 3: Update DeepSORT ───────────────────────────────────────
#             # Always pass embeds= (even as [] for empty frames) — required when embedder=None
#             tracks = tracker.update_tracks(raw_dets, embeds=embeds, frame=frame)

#             # ── Step 4: Recognize each confirmed track via FAISS ──────────────
#             results = []
#             now = time.time()

#             for track in tracks:
#                 if not track.is_confirmed():
#                     continue

#                 ltrb = track.to_ltrb()
#                 x1 = max(0, int(ltrb[0]))
#                 y1 = max(0, int(ltrb[1]))
#                 x2 = min(w_fr, int(ltrb[2]))
#                 y2 = min(h_fr, int(ltrb[3]))
#                 track_id = track.track_id

#                 # Pull latest ArcFace embedding stored by DeepSORT
#                 embedding = None
#                 if hasattr(track, 'features') and track.features:
#                     embedding = track.features[-1]

#                 # Re-run FAISS only after RECOG_INTERVAL seconds per track
#                 cached = self.track_cache.get(track_id)
#                 if embedding is not None and (cached is None or now - cached[2] >= RECOG_INTERVAL):
#                     match, confidence, _ = find_match(embedding)
#                     self.track_cache[track_id] = (match, confidence, now)
#                 elif cached is not None:
#                     match, confidence = cached[0], cached[1]
#                 else:
#                     match, confidence = None, 0.0

#                 if match:
#                     color = (0, 255, 0)
#                     label = f"ID{track_id} {match['name']} {confidence * 100:.0f}%"
#                 else:
#                     color = (0, 0, 255)
#                     label = f"ID{track_id} Unknown"

#                 results.append((x1, y1, x2, y2, label, color))

#             # Clean up stale cache entries for dead tracks
#             active_ids = {t.track_id for t in tracks}
#             self.track_cache = {k: v for k, v in self.track_cache.items() if k in active_ids}

#             with self.lock:
#                 self.output_faces = results

#     def stop(self):
#         self.running = False
#         self.new_frame_event.set()

# # ─────────────────────────────────────────────
# # MAIN LOOP
# # ─────────────────────────────────────────────
# print("\n[4/4] Starting stream... Press Q to quit\n")

# stream = CameraStream(CAMERA_SOURCE, is_file=IS_VIDEO_FILE)
# worker = InferenceWorker()

# frame_count = 0
# fps_counter = 0
# fps_display = 0.0
# fps_timer   = time.time()

# time.sleep(1)  # warm-up

# while stream.is_running():
#     frame = stream.read()
#     if frame is None:
#         break

#     frame_count += 1
#     fps_counter += 1

#     if frame_count % PROCESS_EVERY_N == 0:
#         worker.submit(frame)

#     faces = worker.get_results()
#     for (x1, y1, x2, y2, label, color) in faces:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         label_w = len(label) * 11
#         cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_w, y1), color, -1)
#         cv2.putText(frame, label, (x1 + 4, y1 - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#     elapsed = time.time() - fps_timer
#     if elapsed >= 1.0:
#         fps_display = fps_counter / elapsed
#         fps_counter = 0
#         fps_timer   = time.time()

#     cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.putText(frame, f"Faces: {len(faces)}", (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     cv2.imshow("CCTV Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# stream.stop()
# worker.stop()
# cv2.destroyAllWindows()
# print("\n✓ Done.")