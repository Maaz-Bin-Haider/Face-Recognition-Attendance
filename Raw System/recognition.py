# import cv2
# import pickle
# import numpy as np
# from insightface.app import FaceAnalysis
# import threading
# import time
# import faiss
# from deep_sort_realtime.deepsort_tracker import DeepSort

# print("=" * 70)
# print("LIVE CCTV FACE RECOGNITION - FINAL OPTIMIZED v5")
# print("=" * 70)

# # ═══════════════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════════════
# THRESHOLD        = 1.0    # L2 distance — keep at 1.0 to match how your index was built
# DET_SIZE         = (320, 320)
# PROCESS_EVERY_N  = 3      # run inference every N display frames
# FRAME_SCALE      = 0.5    # detection on half-res (fast, no accuracy loss for det)
# RECOG_INTERVAL   = 1.5    # seconds before re-querying FAISS per track
# MIN_DET_CONF     = 0.45   # drop detections below this confidence
# NMS_IOU_THRESH   = 0.35   # IoU for pre-tracker duplicate removal
# MAX_FACES        = 10     # hard cap on faces per frame

# CAMERA_SOURCE    = r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-26 at 1.50.24 PM.mp4"
# IS_VIDEO_FILE    = not (isinstance(CAMERA_SOURCE, int) or CAMERA_SOURCE.startswith("rtsp"))
# # ═══════════════════════════════════════════════════════════════════════

# # ── 1. LOAD INSIGHTFACE ────────────────────────────────────────────────
# print("\n[1/3] Loading InsightFace...")

# # ┌──────────────────────────────────────────────────────────────────────┐
# # │  CRITICAL — why these exact allowed_modules matter:                  │
# # │                                                                      │
# # │  buffalo_l pipeline:                                                 │
# # │    det_10g        → detects faces, produces 5 keypoints (kps)        │
# # │    2d106det       → 106-pt landmarks (NEEDED: provides kps for align)│
# # │    w600k_r50      → ArcFace ResNet50 recognition                     │
# # │    landmark_3d_68 → 3D landmarks (NOT needed — skip for speed)       │
# # │    genderage      → gender/age prediction (NOT needed — skip)        │
# # │                                                                      │
# # │  InsightFace uses kps (5 keypoints) to perform similarity warp       │
# # │  (norm_crop) BEFORE passing the face to the recognition model.       │
# # │  If you skip landmark_2d_106 / det alignment, kps may still come     │
# # │  from the detector (det_10g provides 5-pt kps natively).             │
# # │  But skipping ALL landmark modules risks losing alignment quality.   │
# # │                                                                      │
# # │  Safest fast config: skip ONLY landmark_3d_68 and genderage.        │
# # │  This saves ~25-30% CPU vs full buffalo_l while keeping recognition  │
# # │  100% identical to your original script.                             │
# # └──────────────────────────────────────────────────────────────────────┘
# face_app = FaceAnalysis(
#     name='buffalo_l',
#     providers=['CPUExecutionProvider'],
#     allowed_modules=['detection', 'landmark_2d_106', 'recognition']
#     # Skipped: landmark_3d_68 (3D landmarks — unused in recognition pipeline)
#     #          genderage       (gender/age — unused in recognition pipeline)
# )
# face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
# print("✓ InsightFace loaded  [buffalo_l · det + 2D align + recognition]")

# # ── 2. LOAD FAISS INDEX ────────────────────────────────────────────────
# print("\n[2/3] Loading FAISS index...")
# index = faiss.read_index("face_index.faiss")
# with open("labels.pkl", "rb") as f:
#     labels = pickle.load(f)
# print(f"✓ FAISS loaded  [{index.ntotal} embeddings]")

# # ── 3. INIT DEEPSORT ──────────────────────────────────────────────────
# print("\n[3/3] Initializing DeepSORT tracker...")
# tracker = DeepSort(
#     max_age=15,           # ghost boxes disappear after 15 missed frames
#     n_init=2,             # confirm track after 2 detections (fast box appearance)
#     nms_max_overlap=0.3,  # second-layer NMS inside DeepSORT
#     embedder=None,        # we supply ArcFace embeddings ourselves
#     half=False,
#     bgr=True,
# )
# print("✓ DeepSORT initialized")


# # ═══════════════════════════════════════════════════════════════════════
# # IoU PRE-TRACKER NMS — kills duplicate detections before DeepSORT
# # This is the main fix for "multiple boxes on one person"
# # ═══════════════════════════════════════════════════════════════════════
# def compute_iou(a, b):
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
#     ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
#     inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
#     if inter == 0:
#         return 0.0
#     return inter / float((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

# def nms_faces(faces, thresh=NMS_IOU_THRESH):
#     if not faces:
#         return faces
#     faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
#     kept = []
#     dead = set()
#     for i, fi in enumerate(faces):
#         if i in dead:
#             continue
#         kept.append(fi)
#         for j in range(i + 1, len(faces)):
#             if j not in dead and compute_iou(fi.bbox, faces[j].bbox) > thresh:
#                 dead.add(j)
#     return kept


# # ═══════════════════════════════════════════════════════════════════════
# # FAISS MATCHING
# # ═══════════════════════════════════════════════════════════════════════
# def find_match(embedding):
#     norm = np.linalg.norm(embedding)
#     if norm == 0:
#         return None, 0.0, float('inf')
#     emb = (embedding / norm).astype('float32')
#     dists, idxs = index.search(np.expand_dims(emb, 0), 1)
#     dist = dists[0][0]
#     idx  = idxs[0][0]
#     if dist < THRESHOLD:
#         sid  = labels[idx]
#         conf = 1.0 - (dist / 2.0)
#         return {"student_id": sid, "name": sid}, conf, dist
#     return None, 0.0, dist


# # ═══════════════════════════════════════════════════════════════════════
# # CAMERA STREAM — non-blocking, always returns latest frame immediately
# # ═══════════════════════════════════════════════════════════════════════
# class CameraStream:
#     def __init__(self, source, is_file=False):
#         self.cap     = cv2.VideoCapture(source)
#         self.is_file = is_file
#         self.frame   = None
#         self.running = True
#         self.lock    = threading.Lock()

#         if is_file:
#             self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
#             self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1 / 30
#             print(f"✓ Video file — source FPS: {self.fps:.1f}")
#         else:
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             print("✓ Live camera")

#         self.thread = threading.Thread(target=self._reader, daemon=True)
#         self.thread.start()

#     def _reader(self):
#         while self.running:
#             t0 = time.time()
#             ret, frame = self.cap.read()
#             if not ret:
#                 if self.is_file:
#                     print("\n[Stream] End of video.")
#                 self.running = False
#                 break
#             with self.lock:
#                 self.frame = frame
#             if self.is_file:
#                 sleep = self.frame_delay - (time.time() - t0)
#                 if sleep > 0:
#                     time.sleep(sleep)

#     def read(self):
#         """Non-blocking — returns latest frame instantly."""
#         with self.lock:
#             return self.frame.copy() if self.frame is not None else None

#     def is_running(self):
#         return self.running

#     def stop(self):
#         self.running = False
#         self.cap.release()


# # ═══════════════════════════════════════════════════════════════════════
# # INFERENCE WORKER — drop-if-busy, never processes stale frames
# # ═══════════════════════════════════════════════════════════════════════
# class InferenceWorker:
#     def __init__(self):
#         self.input_frame     = None
#         self.output_faces    = []
#         self.running         = True
#         self.busy            = False
#         self.lock            = threading.Lock()
#         self.new_frame_event = threading.Event()
#         self.track_cache     = {}   # {track_id: (match, confidence, timestamp)}
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def submit(self, frame):
#         """Accept frame only if idle — drop if currently processing."""
#         with self.lock:
#             if self.busy:
#                 return
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
#                 self.busy = True
#             if frame is None:
#                 with self.lock:
#                     self.busy = False
#                 continue

#             try:
#                 h_fr, w_fr = frame.shape[:2]

#                 # ── Step 1: Detect + align on downscaled frame ────────────────
#                 # InsightFace internally: detects face → finds 5 keypoints →
#                 # performs norm_crop (similarity warp) → runs recognition model.
#                 # FRAME_SCALE only affects detection resolution, not alignment.
#                 small     = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
#                 raw_faces = face_app.get(small)

#                 # Filter low-confidence detections
#                 raw_faces = [f for f in raw_faces if float(f.det_score) >= MIN_DET_CONF]
#                 raw_faces = raw_faces[:MAX_FACES]

#                 # Pre-tracker NMS: remove duplicate detections of same person
#                 faces = nms_faces(raw_faces)

#                 # ── Step 2: Build DeepSORT inputs ─────────────────────────────
#                 raw_dets = []
#                 embeds   = []

#                 for face in faces:
#                     # Scale bbox back to original frame coordinates
#                     x1, y1, x2, y2 = (face.bbox / FRAME_SCALE).astype(int)
#                     x1 = max(0, x1);     y1 = max(0, y1)
#                     x2 = min(w_fr, x2);  y2 = min(h_fr, y2)
#                     w  = max(1, x2 - x1)
#                     h  = max(1, y2 - y1)

#                     conf    = float(face.det_score)
#                     raw_emb = face.embedding
#                     norm    = np.linalg.norm(raw_emb)
#                     emb     = (raw_emb / norm).astype('float32') if norm > 0 else raw_emb.astype('float32')

#                     raw_dets.append(([x1, y1, w, h], conf, "face"))
#                     embeds.append(emb)

#                 # ── Step 3: Update tracker ────────────────────────────────────
#                 # Always pass embeds= (even []) — required when embedder=None
#                 tracks = tracker.update_tracks(raw_dets, embeds=embeds, frame=frame)

#                 # ── Step 4: FAISS recognition per confirmed track ─────────────
#                 results = []
#                 now     = time.time()

#                 for track in tracks:
#                     if not track.is_confirmed():
#                         continue

#                     ltrb = track.to_ltrb()
#                     x1 = max(0, int(ltrb[0]));     y1 = max(0, int(ltrb[1]))
#                     x2 = min(w_fr, int(ltrb[2]));  y2 = min(h_fr, int(ltrb[3]))
#                     track_id = track.track_id

#                     # Get latest embedding stored by DeepSORT
#                     embedding = None
#                     if hasattr(track, 'features') and track.features:
#                         embedding = track.features[-1]
#                         # Trim to prevent memory growth on long runs
#                         if len(track.features) > 1:
#                             track.features = [track.features[-1]]

#                     # Only re-query FAISS after RECOG_INTERVAL seconds
#                     cached = self.track_cache.get(track_id)
#                     if embedding is not None and (cached is None or now - cached[2] >= RECOG_INTERVAL):
#                         match, confidence, _ = find_match(embedding)
#                         self.track_cache[track_id] = (match, confidence, now)
#                     elif cached is not None:
#                         match, confidence = cached[0], cached[1]
#                     else:
#                         match, confidence = None, 0.0

#                     if match:
#                         color = (0, 210, 0)
#                         label = f"{match['name']}  {confidence * 100:.0f}%"
#                     else:
#                         color = (0, 0, 210)
#                         label = f"Unknown  ID{track_id}"

#                     results.append((x1, y1, x2, y2, label, color))

#                 # Remove cache entries for dead tracks
#                 active_ids = {t.track_id for t in tracks}
#                 self.track_cache = {k: v for k, v in self.track_cache.items() if k in active_ids}

#                 with self.lock:
#                     self.output_faces = results

#             finally:
#                 with self.lock:
#                     self.busy = False

#     def stop(self):
#         self.running = False
#         self.new_frame_event.set()


# # ═══════════════════════════════════════════════════════════════════════
# # MAIN LOOP — pure display loop, never blocked by inference
# # ═══════════════════════════════════════════════════════════════════════
# print("\n[4/4] Starting stream... Press Q to quit\n")

# stream = CameraStream(CAMERA_SOURCE, is_file=IS_VIDEO_FILE)
# worker = InferenceWorker()

# frame_count = 0
# fps_counter = 0
# fps_display = 0.0
# fps_timer   = time.time()

# time.sleep(0.5)

# while stream.is_running():
#     frame = stream.read()
#     if frame is None:
#         time.sleep(0.005)
#         continue

#     frame_count += 1
#     fps_counter += 1

#     # Submit every N frames, only when worker is idle
#     if frame_count % PROCESS_EVERY_N == 0:
#         worker.submit(frame)

#     # Overlay latest detection results (1-2 frames stale max — imperceptible)
#     faces = worker.get_results()
#     for (x1, y1, x2, y2, label, color) in faces:
#         # Bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         # Label background — sized to text
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#         cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
#         cv2.putText(frame, label, (x1 + 4, y1 - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#     # FPS counter
#     elapsed = time.time() - fps_timer
#     if elapsed >= 1.0:
#         fps_display = fps_counter / elapsed
#         fps_counter = 0
#         fps_timer   = time.time()

#     cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 35),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
#     cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     cv2.imshow("CCTV Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# stream.stop()
# worker.stop()
# cv2.destroyAllWindows()
# print("\n✓ Done.")




# import cv2
# import pickle
# import numpy as np
# from insightface.app import FaceAnalysis
# import threading
# import time
# import faiss
# from deep_sort_realtime.deepsort_tracker import DeepSort

# print("=" * 70)
# print("LIVE CCTV FACE RECOGNITION - FINAL OPTIMIZED v6")
# print("=" * 70)

# # ═══════════════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════════════
# THRESHOLD        = 0.9    # L2 distance threshold (keep at 1.0 — matches index build)
# DET_SIZE         = (320, 320)
# PROCESS_EVERY_N  = 3      # run inference every N display frames
# FRAME_SCALE      = 0.5    # detection on half-res
# RECOG_INTERVAL   = 1.5    # seconds before re-querying FAISS per track
# MIN_DET_CONF     = 0.50   # drop weak detections
# MAX_FACES        = 10     # hard cap on faces per frame

# # ── NMS SETTINGS ─────────────────────────────────────────────────────
# # Layer 1: Pre-detection NMS (InsightFace output → deduplicate raw detections)
# # This is the PRIMARY fix for multiple boxes.
# # Lower value = more aggressive deduplication.
# # 0.3 means: if two boxes share >30% area → keep only the higher-conf one
# PRE_NMS_IOU      = 0.30

# # Layer 2: Post-track NMS (on final track boxes before drawing)
# # Catches any duplicates that survived tracking (same person, 2 confirmed tracks)
# # This is the SAFETY NET — catches edge cases Layer 1 missed
# POST_NMS_IOU     = 0.45

# CAMERA_SOURCE    = r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-26 at 1.53.27 PM.mp4"
# IS_VIDEO_FILE    = not (isinstance(CAMERA_SOURCE, int) or CAMERA_SOURCE.startswith("rtsp"))
# # ═══════════════════════════════════════════════════════════════════════

# # ── 1. LOAD INSIGHTFACE ────────────────────────────────────────────────
# print("\n[1/3] Loading InsightFace...")
# face_app = FaceAnalysis(
#     name='buffalo_l',
#     providers=['CPUExecutionProvider'],
#     allowed_modules=['detection', 'landmark_2d_106', 'recognition']
#     # landmark_2d_106 is REQUIRED — provides 5-point keypoints for face alignment
#     # (norm_crop warp before recognition). Skipping it kills accuracy on
#     # tilted, distant, or fast-moving faces.
#     # landmark_3d_68 and genderage are safely skipped — not used in recognition.
# )
# face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
# print("✓ InsightFace loaded  [buffalo_l · det + 2D align + recognition]")

# # ── 2. LOAD FAISS INDEX ────────────────────────────────────────────────
# print("\n[2/3] Loading FAISS index...")
# index = faiss.read_index("face_index.faiss")
# with open("labels.pkl", "rb") as f:
#     labels = pickle.load(f)
# print(f"✓ FAISS loaded  [{index.ntotal} embeddings]")

# # ── 3. INIT DEEPSORT ──────────────────────────────────────────────────
# print("\n[3/3] Initializing DeepSORT tracker...")
# tracker = DeepSort(
#     max_age=15,
#     n_init=3,             # CHANGED back to 3: require 3 consistent detections before
#                           # confirming a track. This stops jitter detections from
#                           # spawning confirmed duplicate tracks immediately.
#     nms_max_overlap=0.3,
#     embedder=None,
#     half=False,
#     bgr=True,
# )
# print("✓ DeepSORT initialized")


# # ═══════════════════════════════════════════════════════════════════════
# # IoU UTILITY
# # ═══════════════════════════════════════════════════════════════════════
# def compute_iou(a, b):
#     """IoU between two boxes in [x1, y1, x2, y2] format."""
#     ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
#     ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
#     inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
#     if inter == 0:
#         return 0.0
#     aA = (a[2]-a[0]) * (a[3]-a[1])
#     aB = (b[2]-b[0]) * (b[3]-b[1])
#     return inter / float(aA + aB - inter)


# # ═══════════════════════════════════════════════════════════════════════
# # LAYER 1 — PRE-TRACKER NMS
# # Runs on raw InsightFace detections (small-frame coordinates).
# # Eliminates duplicate detections of the same face before they reach
# # DeepSORT and spawn separate track IDs.
# # ═══════════════════════════════════════════════════════════════════════
# def pre_nms(faces, thresh=PRE_NMS_IOU):
#     if len(faces) <= 1:
#         return faces
#     faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
#     kept = []
#     dead = set()
#     for i, fi in enumerate(faces):
#         if i in dead:
#             continue
#         kept.append(fi)
#         for j in range(i + 1, len(faces)):
#             if j not in dead and compute_iou(fi.bbox, faces[j].bbox) > thresh:
#                 dead.add(j)
#     return kept


# # ═══════════════════════════════════════════════════════════════════════
# # LAYER 2 — POST-TRACK NMS
# # Runs on the final confirmed track bounding boxes (original-frame coords).
# # Safety net: if two tracks have heavily overlapping boxes (same person
# # tracked twice despite Layer 1), keep only the one with higher confidence.
# # ═══════════════════════════════════════════════════════════════════════
# def post_nms(results, thresh=POST_NMS_IOU):
#     """
#     results: list of (x1, y1, x2, y2, label, color, confidence)
#     Returns deduplicated list — same format but without 'confidence' field.
#     """
#     if len(results) <= 1:
#         return [(x1,y1,x2,y2,label,color) for x1,y1,x2,y2,label,color,_ in results]

#     # Sort by confidence descending so highest-conf box wins ties
#     results = sorted(results, key=lambda r: r[6], reverse=True)
#     kept = []
#     dead = set()
#     for i, ri in enumerate(results):
#         if i in dead:
#             continue
#         kept.append(ri)
#         box_i = (ri[0], ri[1], ri[2], ri[3])
#         for j in range(i + 1, len(results)):
#             if j not in dead:
#                 rj = results[j]
#                 box_j = (rj[0], rj[1], rj[2], rj[3])
#                 if compute_iou(box_i, box_j) > thresh:
#                     dead.add(j)

#     return [(x1,y1,x2,y2,label,color) for x1,y1,x2,y2,label,color,_ in kept]


# # ═══════════════════════════════════════════════════════════════════════
# # FAISS MATCHING
# # ═══════════════════════════════════════════════════════════════════════
# def find_match(embedding):
#     norm = np.linalg.norm(embedding)
#     if norm == 0:
#         return None, 0.0, float('inf')
#     emb = (embedding / norm).astype('float32')
#     dists, idxs = index.search(np.expand_dims(emb, 0), 1)
#     dist = dists[0][0]
#     idx  = idxs[0][0]
#     if dist < THRESHOLD:
#         sid  = labels[idx]
#         conf = 1.0 - (dist / 2.0)
#         return {"student_id": sid, "name": sid}, conf, dist
#     return None, 0.0, dist


# # ═══════════════════════════════════════════════════════════════════════
# # CAMERA STREAM — non-blocking, always returns latest frame immediately
# # ═══════════════════════════════════════════════════════════════════════
# class CameraStream:
#     def __init__(self, source, is_file=False):
#         self.cap     = cv2.VideoCapture(source)
#         self.is_file = is_file
#         self.frame   = None
#         self.running = True
#         self.lock    = threading.Lock()

#         if is_file:
#             self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
#             self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1 / 30
#             print(f"✓ Video file — source FPS: {self.fps:.1f}")
#         else:
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             print("✓ Live camera")

#         self.thread = threading.Thread(target=self._reader, daemon=True)
#         self.thread.start()

#     def _reader(self):
#         while self.running:
#             t0 = time.time()
#             ret, frame = self.cap.read()
#             if not ret:
#                 if self.is_file:
#                     print("\n[Stream] End of video.")
#                 self.running = False
#                 break
#             with self.lock:
#                 self.frame = frame
#             if self.is_file:
#                 sleep = self.frame_delay - (time.time() - t0)
#                 if sleep > 0:
#                     time.sleep(sleep)

#     def read(self):
#         with self.lock:
#             return self.frame.copy() if self.frame is not None else None

#     def is_running(self):
#         return self.running

#     def stop(self):
#         self.running = False
#         self.cap.release()


# # ═══════════════════════════════════════════════════════════════════════
# # INFERENCE WORKER — drop-if-busy, always processes freshest frame
# # ═══════════════════════════════════════════════════════════════════════
# class InferenceWorker:
#     def __init__(self):
#         self.input_frame     = None
#         self.output_faces    = []
#         self.running         = True
#         self.busy            = False
#         self.lock            = threading.Lock()
#         self.new_frame_event = threading.Event()
#         self.track_cache     = {}   # {track_id: (match, confidence, timestamp)}
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def submit(self, frame):
#         with self.lock:
#             if self.busy:
#                 return
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
#                 self.busy = True
#             if frame is None:
#                 with self.lock:
#                     self.busy = False
#                 continue

#             try:
#                 h_fr, w_fr = frame.shape[:2]

#                 # ── Step 1: Detect on downscaled frame ────────────────────────
#                 small     = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
#                 raw_faces = face_app.get(small)

#                 # Drop weak detections and cap count
#                 raw_faces = [f for f in raw_faces if float(f.det_score) >= MIN_DET_CONF]
#                 raw_faces = raw_faces[:MAX_FACES]

#                 # LAYER 1 NMS: deduplicate on small-frame coords before tracker
#                 faces = pre_nms(raw_faces)

#                 # ── Step 2: Build DeepSORT inputs ─────────────────────────────
#                 raw_dets = []
#                 embeds   = []

#                 for face in faces:
#                     # Scale bbox back to original resolution
#                     x1, y1, x2, y2 = (face.bbox / FRAME_SCALE).astype(int)
#                     x1 = max(0, x1);     y1 = max(0, y1)
#                     x2 = min(w_fr, x2);  y2 = min(h_fr, y2)
#                     w  = max(1, x2 - x1)
#                     h  = max(1, y2 - y1)

#                     conf    = float(face.det_score)
#                     raw_emb = face.embedding
#                     norm    = np.linalg.norm(raw_emb)
#                     emb     = (raw_emb / norm).astype('float32') if norm > 0 else raw_emb.astype('float32')

#                     raw_dets.append(([x1, y1, w, h], conf, "face"))
#                     embeds.append(emb)

#                 # ── Step 3: Update DeepSORT tracker ───────────────────────────
#                 tracks = tracker.update_tracks(raw_dets, embeds=embeds, frame=frame)

#                 # ── Step 4: FAISS recognition per confirmed track ─────────────
#                 # Build results WITH confidence for post-NMS sorting
#                 raw_results = []
#                 now = time.time()

#                 for track in tracks:
#                     if not track.is_confirmed():
#                         continue

#                     ltrb = track.to_ltrb()
#                     x1 = max(0, int(ltrb[0]));     y1 = max(0, int(ltrb[1]))
#                     x2 = min(w_fr, int(ltrb[2]));  y2 = min(h_fr, int(ltrb[3]))
#                     track_id = track.track_id

#                     # Skip degenerate boxes
#                     if x2 <= x1 or y2 <= y1:
#                         continue

#                     # Get embedding from DeepSORT
#                     embedding = None
#                     if hasattr(track, 'features') and track.features:
#                         embedding = track.features[-1]
#                         if len(track.features) > 1:
#                             track.features = [track.features[-1]]

#                     # Re-query FAISS only after RECOG_INTERVAL
#                     cached = self.track_cache.get(track_id)
#                     if embedding is not None and (cached is None or now - cached[2] >= RECOG_INTERVAL):
#                         match, confidence, _ = find_match(embedding)
#                         self.track_cache[track_id] = (match, confidence, now)
#                     elif cached is not None:
#                         match, confidence = cached[0], cached[1]
#                     else:
#                         match, confidence = None, 0.0

#                     if match:
#                         color = (0, 210, 0)
#                         label = f"{match['name']}  {confidence * 100:.0f}%"
#                         sort_conf = confidence
#                     else:
#                         color = (0, 0, 210)
#                         label = f"Unknown  ID{track_id}"
#                         sort_conf = 0.0

#                     # Include confidence as 7th element for post-NMS sorting
#                     raw_results.append((x1, y1, x2, y2, label, color, sort_conf))

#                 # LAYER 2 NMS: eliminate any surviving duplicate track boxes
#                 final_results = post_nms(raw_results)

#                 # Clean up cache for dead tracks
#                 active_ids = {t.track_id for t in tracks}
#                 self.track_cache = {k: v for k, v in self.track_cache.items() if k in active_ids}

#                 with self.lock:
#                     self.output_faces = final_results

#             finally:
#                 with self.lock:
#                     self.busy = False

#     def stop(self):
#         self.running = False
#         self.new_frame_event.set()


# # ═══════════════════════════════════════════════════════════════════════
# # MAIN LOOP
# # ═══════════════════════════════════════════════════════════════════════
# print("\n[4/4] Starting stream... Press Q to quit\n")

# stream = CameraStream(CAMERA_SOURCE, is_file=IS_VIDEO_FILE)
# worker = InferenceWorker()

# frame_count = 0
# fps_counter = 0
# fps_display = 0.0
# fps_timer   = time.time()

# time.sleep(0.5)

# while stream.is_running():
#     frame = stream.read()
#     if frame is None:
#         time.sleep(0.005)
#         continue

#     frame_count += 1
#     fps_counter += 1

#     if frame_count % PROCESS_EVERY_N == 0:
#         worker.submit(frame)

#     faces = worker.get_results()
#     for (x1, y1, x2, y2, label, color) in faces:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#         cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
#         cv2.putText(frame, label, (x1 + 4, y1 - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#     elapsed = time.time() - fps_timer
#     if elapsed >= 1.0:
#         fps_display = fps_counter / elapsed
#         fps_counter = 0
#         fps_timer   = time.time()

#     cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 35),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
#     cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     cv2.imshow("CCTV Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# stream.stop()
# worker.stop()
# cv2.destroyAllWindows()
# print("\n✓ Done.")



# import cv2
# import pickle
# import numpy as np
# from insightface.app import FaceAnalysis
# import threading
# import time
# import faiss
# from deep_sort_realtime.deepsort_tracker import DeepSort

# print("=" * 70)
# print("LIVE CCTV FACE RECOGNITION - FINAL OPTIMIZED v7 (Fast-Pass Fix)")
# print("=" * 70)

# # ═══════════════════════════════════════════════════════════════════════
# # CONFIG
# # ═══════════════════════════════════════════════════════════════════════
# THRESHOLD        = 0.9
# DET_SIZE         = (320, 320)

# # FIX 1: Reduced from 3 → 1 so EVERY display frame is submitted for inference.
# # This eliminates the temporal blind spot where fast-moving students were
# # skipped entirely. The InferenceWorker is already drop-if-busy, so if CPU
# # can't keep up it naturally self-throttles without blocking the display loop.
# PROCESS_EVERY_N  = 1

# FRAME_SCALE      = 0.5

# # FIX 3a: Reduced from 1.5s → 0.5s. For a student in frame for <1s, the old
# # 1.5s interval meant FAISS was queried 0 or 1 times. At 0.5s it gets queried
# # up to 2x in a short appearance, dramatically improving recognition coverage.
# RECOG_INTERVAL   = 0.5

# MIN_DET_CONF     = 0.50
# MAX_FACES        = 10

# PRE_NMS_IOU      = 0.30
# POST_NMS_IOU     = 0.45

# CAMERA_SOURCE    = r"C:\Users\SWISS TECH\Downloads\IMG_7179.MOV"
# IS_VIDEO_FILE    = not (isinstance(CAMERA_SOURCE, int) or CAMERA_SOURCE.startswith("rtsp"))
# # ═══════════════════════════════════════════════════════════════════════

# # ── 1. LOAD INSIGHTFACE ────────────────────────────────────────────────
# print("\n[1/3] Loading InsightFace...")
# face_app = FaceAnalysis(
#     name='buffalo_l',
#     providers=['CPUExecutionProvider'],
#     allowed_modules=['detection', 'landmark_2d_106', 'recognition']
# )
# face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
# print("✓ InsightFace loaded  [buffalo_l · det + 2D align + recognition]")

# # ── 2. LOAD FAISS INDEX ────────────────────────────────────────────────
# print("\n[2/3] Loading FAISS index...")
# index = faiss.read_index("face_index.faiss")
# with open("labels.pkl", "rb") as f:
#     labels = pickle.load(f)
# print(f"✓ FAISS loaded  [{index.ntotal} embeddings]")

# # ── 3. INIT DEEPSORT ──────────────────────────────────────────────────
# print("\n[3/3] Initializing DeepSORT tracker...")
# tracker = DeepSort(
#     max_age=15,
#     # FIX 2: Reduced from 3 → 2. With n_init=3 at PROCESS_EVERY_N=3, a track
#     # needed 9 display frames (~0.3s) to confirm. Now it confirms in 2 detections
#     # (~0.07s at full inference rate), leaving far more time for FAISS to fire
#     # before the student exits frame. Accuracy impact is minimal — the pre-NMS
#     # and IoU matching still prevent phantom tracks.
#     n_init=2,
#     nms_max_overlap=0.3,
#     embedder=None,
#     half=False,
#     bgr=True,
# )
# print("✓ DeepSORT initialized")


# # ═══════════════════════════════════════════════════════════════════════
# # IoU UTILITY
# # ═══════════════════════════════════════════════════════════════════════
# def compute_iou(a, b):
#     ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
#     ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
#     inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
#     if inter == 0:
#         return 0.0
#     aA = (a[2]-a[0]) * (a[3]-a[1])
#     aB = (b[2]-b[0]) * (b[3]-b[1])
#     return inter / float(aA + aB - inter)


# # ═══════════════════════════════════════════════════════════════════════
# # LAYER 1 — PRE-TRACKER NMS
# # ═══════════════════════════════════════════════════════════════════════
# def pre_nms(faces, thresh=PRE_NMS_IOU):
#     if len(faces) <= 1:
#         return faces
#     faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
#     kept = []
#     dead = set()
#     for i, fi in enumerate(faces):
#         if i in dead:
#             continue
#         kept.append(fi)
#         for j in range(i + 1, len(faces)):
#             if j not in dead and compute_iou(fi.bbox, faces[j].bbox) > thresh:
#                 dead.add(j)
#     return kept


# # ═══════════════════════════════════════════════════════════════════════
# # LAYER 2 — POST-TRACK NMS
# # ═══════════════════════════════════════════════════════════════════════
# def post_nms(results, thresh=POST_NMS_IOU):
#     if len(results) <= 1:
#         return [(x1,y1,x2,y2,label,color) for x1,y1,x2,y2,label,color,_ in results]
#     results = sorted(results, key=lambda r: r[6], reverse=True)
#     kept = []
#     dead = set()
#     for i, ri in enumerate(results):
#         if i in dead:
#             continue
#         kept.append(ri)
#         box_i = (ri[0], ri[1], ri[2], ri[3])
#         for j in range(i + 1, len(results)):
#             if j not in dead:
#                 rj = results[j]
#                 box_j = (rj[0], rj[1], rj[2], rj[3])
#                 if compute_iou(box_i, box_j) > thresh:
#                     dead.add(j)
#     return [(x1,y1,x2,y2,label,color) for x1,y1,x2,y2,label,color,_ in kept]


# # ═══════════════════════════════════════════════════════════════════════
# # FAISS MATCHING
# # ═══════════════════════════════════════════════════════════════════════
# def find_match(embedding):
#     norm = np.linalg.norm(embedding)
#     if norm == 0:
#         return None, 0.0, float('inf')
#     emb = (embedding / norm).astype('float32')
#     dists, idxs = index.search(np.expand_dims(emb, 0), 1)
#     dist = dists[0][0]
#     idx  = idxs[0][0]
#     if dist < THRESHOLD:
#         sid  = labels[idx]
#         conf = 1.0 - (dist / 2.0)
#         return {"student_id": sid, "name": sid}, conf, dist
#     return None, 0.0, dist


# # ═══════════════════════════════════════════════════════════════════════
# # CAMERA STREAM
# # ═══════════════════════════════════════════════════════════════════════
# class CameraStream:
#     def __init__(self, source, is_file=False):
#         self.cap     = cv2.VideoCapture(source)
#         self.is_file = is_file
#         self.frame   = None
#         self.running = True
#         self.lock    = threading.Lock()

#         if is_file:
#             self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
#             self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1 / 30
#             print(f"✓ Video file — source FPS: {self.fps:.1f}")
#         else:
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             print("✓ Live camera")

#         self.thread = threading.Thread(target=self._reader, daemon=True)
#         self.thread.start()

#     def _reader(self):
#         while self.running:
#             t0 = time.time()
#             ret, frame = self.cap.read()
#             if not ret:
#                 if self.is_file:
#                     print("\n[Stream] End of video.")
#                 self.running = False
#                 break
#             with self.lock:
#                 self.frame = frame
#             if self.is_file:
#                 sleep = self.frame_delay - (time.time() - t0)
#                 if sleep > 0:
#                     time.sleep(sleep)

#     def read(self):
#         with self.lock:
#             return self.frame.copy() if self.frame is not None else None

#     def is_running(self):
#         return self.running

#     def stop(self):
#         self.running = False
#         self.cap.release()


# # ═══════════════════════════════════════════════════════════════════════
# # INFERENCE WORKER
# # ═══════════════════════════════════════════════════════════════════════
# class InferenceWorker:
#     def __init__(self):
#         self.input_frame     = None
#         self.output_faces    = []
#         self.running         = True
#         self.busy            = False
#         self.lock            = threading.Lock()
#         self.new_frame_event = threading.Event()
#         self.track_cache     = {}
#         # FIX 3b: Track which IDs have NEVER been queried yet.
#         # These get an immediate FAISS call regardless of RECOG_INTERVAL,
#         # so even a student who appears for just 2 frames gets recognized.
#         self.never_queried   = set()
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def submit(self, frame):
#         with self.lock:
#             if self.busy:
#                 return
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
#                 self.busy = True
#             if frame is None:
#                 with self.lock:
#                     self.busy = False
#                 continue

#             try:
#                 h_fr, w_fr = frame.shape[:2]

#                 small     = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
#                 raw_faces = face_app.get(small)

#                 raw_faces = [f for f in raw_faces if float(f.det_score) >= MIN_DET_CONF]
#                 raw_faces = raw_faces[:MAX_FACES]

#                 faces = pre_nms(raw_faces)

#                 raw_dets = []
#                 embeds   = []

#                 for face in faces:
#                     x1, y1, x2, y2 = (face.bbox / FRAME_SCALE).astype(int)
#                     x1 = max(0, x1);     y1 = max(0, y1)
#                     x2 = min(w_fr, x2);  y2 = min(h_fr, y2)
#                     w  = max(1, x2 - x1)
#                     h  = max(1, y2 - y1)

#                     conf    = float(face.det_score)
#                     raw_emb = face.embedding
#                     norm    = np.linalg.norm(raw_emb)
#                     emb     = (raw_emb / norm).astype('float32') if norm > 0 else raw_emb.astype('float32')

#                     raw_dets.append(([x1, y1, w, h], conf, "face"))
#                     embeds.append(emb)

#                 tracks = tracker.update_tracks(raw_dets, embeds=embeds, frame=frame)

#                 raw_results = []
#                 now = time.time()

#                 for track in tracks:
#                     if not track.is_confirmed():
#                         continue

#                     ltrb = track.to_ltrb()
#                     x1 = max(0, int(ltrb[0]));     y1 = max(0, int(ltrb[1]))
#                     x2 = min(w_fr, int(ltrb[2]));  y2 = min(h_fr, int(ltrb[3]))
#                     track_id = track.track_id

#                     if x2 <= x1 or y2 <= y1:
#                         continue

#                     embedding = None
#                     if hasattr(track, 'features') and track.features:
#                         embedding = track.features[-1]
#                         if len(track.features) > 1:
#                             track.features = [track.features[-1]]

#                     cached = self.track_cache.get(track_id)

#                     # FIX 3b: First-seen tracks bypass the interval entirely.
#                     # A brand-new confirmed track fires FAISS immediately so
#                     # fast-pass students are recognized on their very first
#                     # confirmed frame rather than waiting up to 1.5 seconds.
#                     is_first_time = track_id in self.never_queried or cached is None

#                     should_query = (
#                         embedding is not None and
#                         (is_first_time or now - cached[2] >= RECOG_INTERVAL)
#                     )

#                     if should_query:
#                         match, confidence, _ = find_match(embedding)
#                         self.track_cache[track_id] = (match, confidence, now)
#                         # Remove from never_queried once first query is done
#                         self.never_queried.discard(track_id)
#                     elif cached is not None:
#                         match, confidence = cached[0], cached[1]
#                     else:
#                         match, confidence = None, 0.0

#                     if match:
#                         color     = (0, 210, 0)
#                         label     = f"{match['name']}  {confidence * 100:.0f}%"
#                         sort_conf = confidence
#                     else:
#                         color     = (0, 0, 210)
#                         label     = f"Unknown  ID{track_id}"
#                         sort_conf = 0.0

#                     raw_results.append((x1, y1, x2, y2, label, color, sort_conf))

#                 final_results = post_nms(raw_results)

#                 # Mark new track IDs that just appeared as never-queried
#                 active_ids = {t.track_id for t in tracks if t.is_confirmed()}
#                 for tid in active_ids:
#                     if tid not in self.track_cache:
#                         self.never_queried.add(tid)

#                 # Clean up cache for dead tracks
#                 all_active = {t.track_id for t in tracks}
#                 self.track_cache  = {k: v for k, v in self.track_cache.items() if k in all_active}
#                 self.never_queried = {k for k in self.never_queried if k in all_active}

#                 with self.lock:
#                     self.output_faces = final_results

#             finally:
#                 with self.lock:
#                     self.busy = False

#     def stop(self):
#         self.running = False
#         self.new_frame_event.set()


# # ═══════════════════════════════════════════════════════════════════════
# # MAIN LOOP
# # ═══════════════════════════════════════════════════════════════════════
# print("\n[4/4] Starting stream... Press Q to quit\n")

# stream = CameraStream(CAMERA_SOURCE, is_file=IS_VIDEO_FILE)
# worker = InferenceWorker()

# frame_count = 0
# fps_counter = 0
# fps_display = 0.0
# fps_timer   = time.time()

# time.sleep(0.5)

# while stream.is_running():
#     frame = stream.read()
#     if frame is None:
#         time.sleep(0.005)
#         continue

#     frame_count += 1
#     fps_counter += 1

#     # PROCESS_EVERY_N is now 1 — every frame is submitted.
#     # The worker's drop-if-busy guard means if inference is still running
#     # from the previous frame, this submission is silently skipped and
#     # the display loop continues unblocked — no stutter, no queue buildup.
#     if frame_count % PROCESS_EVERY_N == 0:
#         worker.submit(frame)

#     faces = worker.get_results()
#     for (x1, y1, x2, y2, label, color) in faces:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#         cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
#         cv2.putText(frame, label, (x1 + 4, y1 - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#     elapsed = time.time() - fps_timer
#     if elapsed >= 1.0:
#         fps_display = fps_counter / elapsed
#         fps_counter = 0
#         fps_timer   = time.time()

#     cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 35),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
#     cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     cv2.imshow("CCTV Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# stream.stop()
# worker.stop()
# cv2.destroyAllWindows()
# print("\n✓ Done.")




import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import threading
import time
import faiss
from deep_sort_realtime.deepsort_tracker import DeepSort

print("=" * 70)
print("LIVE CCTV FACE RECOGNITION - FINAL OPTIMIZED v8 (Ghost Box Fix)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
THRESHOLD        = 0.9
DET_SIZE         = (320, 320)
PROCESS_EVERY_N  = 1
FRAME_SCALE      = 0.5
RECOG_INTERVAL   = 0.5
MIN_DET_CONF     = 0.50
MAX_FACES        = 10

PRE_NMS_IOU      = 0.30
POST_NMS_IOU     = 0.45

# FIX 3: Center-distance threshold for post-NMS ghost suppression.
# If two confirmed track boxes have centers within this many pixels of
# each other, the lower-confidence one is treated as a ghost and removed.
# Set relative to your typical face-box size at FRAME_SCALE=0.5.
# 80px on original resolution ≈ a face width — safe default.
CENTER_DIST_PX   = 80

CAMERA_SOURCE    = r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-27 at 12.57.32 PM.mp4"
IS_VIDEO_FILE    = not (isinstance(CAMERA_SOURCE, int) or CAMERA_SOURCE.startswith("rtsp"))
# ═══════════════════════════════════════════════════════════════════════


# ── 1. LOAD INSIGHTFACE ────────────────────────────────────────────────
print("\n[1/3] Loading InsightFace...")
face_app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'landmark_2d_106', 'recognition']
)
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
print("✓ InsightFace loaded")

# ── 2. LOAD FAISS INDEX ────────────────────────────────────────────────
print("\n[2/3] Loading FAISS index...")
index = faiss.read_index("face_index.faiss")
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)
print(f"✓ FAISS loaded  [{index.ntotal} embeddings]")

# ── 3. INIT DEEPSORT ──────────────────────────────────────────────────
print("\n[3/3] Initializing DeepSORT tracker...")
tracker = DeepSort(
    # FIX 2: Reduced from 15 → 5. Ghost tracks from the previous position
    # now die in ~5 frames (~0.17s at 30fps) instead of lingering for 0.5s.
    # This is the primary reason multiple boxes persist during fast movement.
    max_age=5,

    n_init=2,

    # FIX 1: max_iou_distance controls how far DeepSORT searches for a match.
    # Default is 0.7. Raising it to 0.95 means even boxes with very low IoU
    # (fast-moving face that jumped far) can still be matched to the existing
    # track, preventing a new ghost track from being spawned in the first place.
    # This is the root-cause fix — stops duplicates from being created at all.
    max_iou_distance=0.95,

    nms_max_overlap=0.3,
    embedder=None,
    half=False,
    bgr=True,
)
print("✓ DeepSORT initialized")


# ═══════════════════════════════════════════════════════════════════════
# IoU UTILITY
# ═══════════════════════════════════════════════════════════════════════
def compute_iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aA = (a[2]-a[0]) * (a[3]-a[1])
    aB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(aA + aB - inter)


# ═══════════════════════════════════════════════════════════════════════
# FIX 3: ENHANCED POST-NMS WITH CENTER-DISTANCE SUPPRESSION
# Standard IoU NMS only catches overlapping boxes. When a ghost track's
# predicted box has drifted away from the real face position, the two boxes
# don't overlap — IoU=0 — and NMS keeps both. Center-distance suppression
# catches these separated ghosts by checking proximity of box centers.
# ═══════════════════════════════════════════════════════════════════════
def box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def center_dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def post_nms(results, iou_thresh=POST_NMS_IOU, dist_thresh=CENTER_DIST_PX):
    """
    Dual-criterion NMS:
      - Suppresses boxes with IoU > iou_thresh (standard overlap NMS)
      - ALSO suppresses boxes whose centers are within dist_thresh pixels
        (catches ghost tracks that have drifted away from the real face)
    Higher-confidence box always wins when suppressing.
    """
    if len(results) <= 1:
        return [(x1,y1,x2,y2,label,color) for x1,y1,x2,y2,label,color,_ in results]

    results = sorted(results, key=lambda r: r[6], reverse=True)
    kept = []
    dead = set()

    for i, ri in enumerate(results):
        if i in dead:
            continue
        kept.append(ri)
        box_i    = (ri[0], ri[1], ri[2], ri[3])
        center_i = box_center(*box_i)

        for j in range(i + 1, len(results)):
            if j in dead:
                continue
            rj       = results[j]
            box_j    = (rj[0], rj[1], rj[2], rj[3])
            center_j = box_center(*box_j)

            iou_hit  = compute_iou(box_i, box_j) > iou_thresh
            dist_hit = center_dist(center_i, center_j) < dist_thresh

            if iou_hit or dist_hit:
                dead.add(j)

    return [(x1,y1,x2,y2,label,color) for x1,y1,x2,y2,label,color,_ in kept]


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1 — PRE-TRACKER NMS (unchanged)
# ═══════════════════════════════════════════════════════════════════════
def pre_nms(faces, thresh=PRE_NMS_IOU):
    if len(faces) <= 1:
        return faces
    faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
    kept = []
    dead = set()
    for i, fi in enumerate(faces):
        if i in dead:
            continue
        kept.append(fi)
        for j in range(i + 1, len(faces)):
            if j not in dead and compute_iou(fi.bbox, faces[j].bbox) > thresh:
                dead.add(j)
    return kept


# ═══════════════════════════════════════════════════════════════════════
# FAISS MATCHING
# ═══════════════════════════════════════════════════════════════════════
def find_match(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None, 0.0, float('inf')
    emb = (embedding / norm).astype('float32')
    dists, idxs = index.search(np.expand_dims(emb, 0), 1)
    dist = dists[0][0]
    idx  = idxs[0][0]
    if dist < THRESHOLD:
        sid  = labels[idx]
        conf = 1.0 - (dist / 2.0)
        return {"student_id": sid, "name": sid}, conf, dist
    return None, 0.0, dist


# ═══════════════════════════════════════════════════════════════════════
# CAMERA STREAM
# ═══════════════════════════════════════════════════════════════════════
class CameraStream:
    def __init__(self, source, is_file=False):
        self.cap     = cv2.VideoCapture(source)
        self.is_file = is_file
        self.frame   = None
        self.running = True
        self.lock    = threading.Lock()

        if is_file:
            self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = 1.0 / self.fps if self.fps > 0 else 1 / 30
            print(f"✓ Video file — source FPS: {self.fps:.1f}")
        else:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("✓ Live camera")

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    print("\n[Stream] End of video.")
                self.running = False
                break
            with self.lock:
                self.frame = frame
            if self.is_file:
                sleep = self.frame_delay - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE WORKER
# ═══════════════════════════════════════════════════════════════════════
class InferenceWorker:
    def __init__(self):
        self.input_frame     = None
        self.output_faces    = []
        self.running         = True
        self.busy            = False
        self.lock            = threading.Lock()
        self.new_frame_event = threading.Event()
        self.track_cache     = {}
        self.never_queried   = set()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            if self.busy:
                return
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
                self.busy = True
            if frame is None:
                with self.lock:
                    self.busy = False
                continue

            try:
                h_fr, w_fr = frame.shape[:2]

                small     = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                raw_faces = face_app.get(small)
                raw_faces = [f for f in raw_faces if float(f.det_score) >= MIN_DET_CONF]
                raw_faces = raw_faces[:MAX_FACES]
                faces     = pre_nms(raw_faces)

                raw_dets = []
                embeds   = []

                for face in faces:
                    x1, y1, x2, y2 = (face.bbox / FRAME_SCALE).astype(int)
                    x1 = max(0, x1);     y1 = max(0, y1)
                    x2 = min(w_fr, x2);  y2 = min(h_fr, y2)
                    w  = max(1, x2 - x1)
                    h  = max(1, y2 - y1)

                    conf    = float(face.det_score)
                    raw_emb = face.embedding
                    norm    = np.linalg.norm(raw_emb)
                    emb     = (raw_emb / norm).astype('float32') if norm > 0 else raw_emb.astype('float32')

                    raw_dets.append(([x1, y1, w, h], conf, "face"))
                    embeds.append(emb)

                tracks = tracker.update_tracks(raw_dets, embeds=embeds, frame=frame)

                raw_results = []
                now = time.time()

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    ltrb = track.to_ltrb()
                    x1 = max(0, int(ltrb[0]));     y1 = max(0, int(ltrb[1]))
                    x2 = min(w_fr, int(ltrb[2]));  y2 = min(h_fr, int(ltrb[3]))
                    track_id = track.track_id

                    if x2 <= x1 or y2 <= y1:
                        continue

                    embedding = None
                    if hasattr(track, 'features') and track.features:
                        embedding = track.features[-1]
                        if len(track.features) > 1:
                            track.features = [track.features[-1]]

                    cached       = self.track_cache.get(track_id)
                    is_first     = track_id in self.never_queried or cached is None
                    should_query = embedding is not None and (
                        is_first or now - cached[2] >= RECOG_INTERVAL
                    )

                    if should_query:
                        match, confidence, _ = find_match(embedding)
                        self.track_cache[track_id] = (match, confidence, now)
                        self.never_queried.discard(track_id)
                    elif cached is not None:
                        match, confidence = cached[0], cached[1]
                    else:
                        match, confidence = None, 0.0

                    if match:
                        color     = (0, 210, 0)
                        label     = f"{match['name']}  {confidence * 100:.0f}%"
                        sort_conf = confidence
                    else:
                        color     = (0, 0, 210)
                        label     = f"Unknown  ID{track_id}"
                        sort_conf = 0.0

                    raw_results.append((x1, y1, x2, y2, label, color, sort_conf))

                # Enhanced dual-criterion NMS
                final_results = post_nms(raw_results)

                active_ids = {t.track_id for t in tracks if t.is_confirmed()}
                for tid in active_ids:
                    if tid not in self.track_cache:
                        self.never_queried.add(tid)

                all_active        = {t.track_id for t in tracks}
                self.track_cache  = {k: v for k, v in self.track_cache.items() if k in all_active}
                self.never_queried = {k for k in self.never_queried if k in all_active}

                with self.lock:
                    self.output_faces = final_results

            finally:
                with self.lock:
                    self.busy = False

    def stop(self):
        self.running = False
        self.new_frame_event.set()


# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
print("\n[4/4] Starting stream... Press Q to quit\n")

stream = CameraStream(CAMERA_SOURCE, is_file=IS_VIDEO_FILE)
worker = InferenceWorker()

frame_count = 0
fps_counter = 0
fps_display = 0.0
fps_timer   = time.time()

time.sleep(0.5)

while stream.is_running():
    frame = stream.read()
    if frame is None:
        time.sleep(0.005)
        continue

    frame_count += 1
    fps_counter += 1

    if frame_count % PROCESS_EVERY_N == 0:
        worker.submit(frame)

    faces = worker.get_results()
    for (x1, y1, x2, y2, label, color) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    elapsed = time.time() - fps_timer
    if elapsed >= 1.0:
        fps_display = fps_counter / elapsed
        fps_counter = 0
        fps_timer   = time.time()

    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("CCTV Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
worker.stop()
cv2.destroyAllWindows()
print("\n✓ Done.")


