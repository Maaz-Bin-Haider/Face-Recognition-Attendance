# 📹 Live CCTV Face Recognition Script — Technical Documentation

> **File:** `recognition.py`  
> **Purpose:** Real-time face detection, tracking, and recognition from live CCTV or recorded video footage.  
> **Version:** v8 — Final Optimized (Ghost Box Fix)  
> **Author:** SWISS TECH Team

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies & Libraries](#dependencies--libraries)
3. [Configuration Block](#configuration-block)
4. [Model & Index Loading](#model--index-loading)
5. [DeepSORT Tracker Initialization](#deepsort-tracker-initialization)
6. [Utility Functions](#utility-functions)
7. [Pre-Tracker NMS — Layer 1](#pre-tracker-nms--layer-1)
8. [Post-Tracker NMS — Layer 2](#post-tracker-nms--layer-2)
9. [FAISS Matching](#faiss-matching)
10. [CameraStream Class](#camerastream-class)
11. [InferenceWorker Class](#inferenceworker-class)
12. [Main Display Loop](#main-display-loop)
13. [Threading Architecture](#threading-architecture)
14. [NMS Dual-Layer Strategy](#nms-dual-layer-strategy)
15. [Performance Tuning Reference](#performance-tuning-reference)

---

## Overview

This script is the **runtime engine** of the face recognition attendance system. It continuously reads video frames, detects faces, tracks them across frames using DeepSORT, and identifies each person by comparing their face embedding against the FAISS database built during enrollment.

The architecture is built around three concurrent threads:

- **Stream Thread** — reads frames from camera/file as fast as possible
- **Inference Thread** — runs face detection + tracking + recognition (CPU-heavy)
- **Display Thread** (main thread) — draws boxes on frames and shows the window at full FPS

This separation ensures the UI never freezes while inference is running, and that inference always works on the freshest available frame.

---

## Dependencies & Libraries

```python
import cv2                        # Frame capture, drawing, display
import pickle                     # Loading labels.pkl
import numpy as np                # Embedding math
from insightface.app import FaceAnalysis   # Face detection + embedding
import threading                  # Multi-thread architecture
import time                       # Timing for FPS, recognition intervals
import faiss                      # Nearest-neighbor search
from deep_sort_realtime.deepsort_tracker import DeepSort  # Multi-object tracker
```

| Library | Role |
|---|---|
| `insightface` (buffalo_l) | Detects faces and generates 512-dim ArcFace embeddings |
| `faiss` | Sub-millisecond nearest-neighbor search in embedding space |
| `deep_sort_realtime` | Assigns persistent track IDs across frames using Kalman filter + appearance features |
| `opencv-cv2` | Video decoding, resizing, drawing bounding boxes and labels |
| `threading` | Non-blocking parallel execution of stream, inference, and display |

---

## Configuration Block

```python
THRESHOLD       = 0.9    # L2 distance threshold for FAISS matching
DET_SIZE        = (320, 320)   # Face detection input size
PROCESS_EVERY_N = 1      # Submit frame to inference every N display frames
FRAME_SCALE     = 0.5    # Downscale factor before detection
RECOG_INTERVAL  = 0.5    # Seconds before re-querying FAISS for same track
MIN_DET_CONF    = 0.50   # Minimum detection confidence to accept a face
MAX_FACES       = 10     # Hard cap on simultaneous face detections

PRE_NMS_IOU     = 0.30   # IoU threshold for pre-tracker deduplication
POST_NMS_IOU    = 0.45   # IoU threshold for post-tracker deduplication
CENTER_DIST_PX  = 80     # Pixel distance threshold for ghost box suppression
```

### Threshold Explained

`THRESHOLD = 0.9` is the **L2 distance cutoff** for recognition decisions. After normalizing embeddings to unit length:
- Distance `0.0` = identical face
- Distance `~0.6–0.8` = same person, different conditions
- Distance `~1.0+` = different people
- Distance `2.0` = maximum (opposite unit vectors)

Setting `THRESHOLD = 0.9` means: only declare a match if the nearest neighbor in FAISS is within 0.9 L2 distance. Faces beyond this distance are labeled "Unknown."

### FRAME_SCALE Explained

Detection runs on a **50% scaled-down copy** of the frame. A 1080p frame (1920×1080) becomes 960×540 before being passed to InsightFace. This nearly halves the detection time with minimal accuracy loss for faces that are large enough to be detected reliably. Detected bounding boxes are then scaled back up by dividing by `FRAME_SCALE`.

### RECOG_INTERVAL Explained

Once a track is recognized, its identity is cached and not re-queried from FAISS for 0.5 seconds. This prevents redundant FAISS calls on every frame for a stationary face. For fast-moving faces, the cache is bypassed on the very first confirmed frame (see `never_queried` logic in InferenceWorker).

---

## Model & Index Loading

### InsightFace

```python
face_app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'landmark_2d_106', 'recognition']
)
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
```

Only three modules are loaded from the buffalo_l bundle:
- `detection` — RetinaFace bounding box detector
- `landmark_2d_106` — 5-point keypoint detector needed for face alignment (crop + warp) before recognition. **Required** — without alignment, accuracy on tilted or fast-moving faces drops significantly.
- `recognition` — ArcFace embedding generator

Excluded modules (`landmark_3d_68`, `genderage`) save loading time and memory without affecting recognition accuracy.

### FAISS Index

```python
index = faiss.read_index("face_index.faiss")
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)
```

The FAISS index and label list built by the enrollment script are loaded directly into memory. `index.ntotal` shows the total number of stored embeddings across all enrolled students.

---

## DeepSORT Tracker Initialization

```python
tracker = DeepSort(
    max_age=5,
    n_init=2,
    max_iou_distance=0.95,
    nms_max_overlap=0.3,
    embedder=None,
    half=False,
    bgr=True,
)
```

DeepSORT is a **multi-object tracker** that assigns persistent IDs to detected faces across frames. It uses a Kalman filter to predict where each tracked face will be in the next frame, and the Hungarian algorithm to match new detections to existing tracks.

| Parameter | Value | Reason |
|---|---|---|
| `max_age=5` | 5 frames | A track with no matching detection dies after 5 frames. Reduced from 15 to prevent ghost boxes from lingering during fast movement. |
| `n_init=2` | 2 frames | A new track is "confirmed" after matching detections in 2 consecutive frames. Lower value = faster confirmation = more time for FAISS to recognize fast-pass subjects. |
| `max_iou_distance=0.95` | 0.95 | How far apart (in IoU terms) a detection can be from a track's predicted position and still be considered a match. High value = handles fast-moving faces without spawning ghost tracks. |
| `nms_max_overlap=0.3` | 0.3 | Internal DeepSORT NMS overlap threshold |
| `embedder=None` | None | We supply our own ArcFace embeddings — DeepSORT's built-in embedder is disabled |

---

## Utility Functions

### `compute_iou(a, b)`

```python
def compute_iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aA = (a[2]-a[0]) * (a[3]-a[1])
    aB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(aA + aB - inter)
```

Computes the **Intersection over Union** between two bounding boxes in `[x1, y1, x2, y2]` format. IoU ranges from 0.0 (no overlap) to 1.0 (identical boxes). Used by both NMS layers to identify duplicate boxes.

### `box_center(x1, y1, x2, y2)`

Returns the pixel coordinates of the center of a bounding box. Used by the enhanced post-NMS for ghost box suppression.

### `center_dist(a, b)`

Computes Euclidean pixel distance between two center points. Used to detect ghost boxes that have drifted away from the real face — these have near-zero IoU but close center proximity.

---

## Pre-Tracker NMS — Layer 1

```python
def pre_nms(faces, thresh=PRE_NMS_IOU):
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
```

**When it runs:** Immediately after InsightFace detection, before any detections are passed to DeepSORT.

**What it does:** Sorts detections by confidence (highest first), then greedily removes any lower-confidence detection that overlaps with an already-kept detection by more than 30% IoU.

**Why it's needed:** InsightFace can occasionally output two overlapping boxes for the same face (especially near frame edges or at oblique angles). If these duplicates reach DeepSORT, they spawn two separate track IDs for the same physical face, causing persistent double-box artifacts.

---

## Post-Tracker NMS — Layer 2

```python
def post_nms(results, iou_thresh=POST_NMS_IOU, dist_thresh=CENTER_DIST_PX):
```

**When it runs:** After DeepSORT tracking, on the final list of confirmed track bounding boxes.

**What it does:** Dual-criterion suppression:

1. **IoU criterion:** Removes any box that overlaps an already-kept box by more than 45% — catches duplicate tracks with overlapping boxes.

2. **Center-distance criterion:** Removes any box whose center is within 80 pixels of an already-kept box — catches **ghost tracks** that have drifted away from the real face. When a face moves quickly, DeepSORT may briefly maintain both the old track (with its Kalman-predicted position) and a new track at the current face position. These two boxes don't overlap (IoU=0), so standard NMS misses them. Center-distance suppression catches them.

**Why two layers?** Layer 1 prevents duplicates from being created. Layer 2 is the safety net for any duplicates that survive Layer 1 through DeepSORT's internal matching.

---

## FAISS Matching

```python
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
```

**Input:** A 512-dimensional face embedding from a DeepSORT track.

**Process:**
1. L2-normalize the embedding (ensures fair comparison with normalized database embeddings)
2. Call `index.search()` to find the single nearest neighbor in the entire FAISS database
3. Check if the distance is below `THRESHOLD = 0.9`
4. Convert distance to a confidence score: `conf = 1.0 - (dist / 2.0)`. At distance 0.0, confidence = 100%. At distance 0.9, confidence = 55%.

**Output:** Returns the matched student's info dict + confidence, or `None` if no match found.

---

## CameraStream Class

```python
class CameraStream:
    def __init__(self, source, is_file=False): ...
    def _reader(self): ...       # Runs in background thread
    def read(self): ...          # Returns latest frame instantly
    def is_running(self): ...
    def stop(self): ...
```

A **non-blocking video reader** that runs in its own background thread.

### Why a Separate Thread?

`cap.read()` (OpenCV's frame read) is a **blocking call** — it waits until the next frame is available from the camera or file. If this ran in the main loop, the display would freeze waiting for each frame. By putting it in a thread:
- The stream thread always has the freshest frame ready
- The main thread calls `stream.read()` and immediately gets the latest frame with no wait
- Frames the display loop doesn't consume are silently overwritten — no queue buildup

### Video File Pacing

```python
if is_file:
    sleep = self.frame_delay - (time.time() - t0)
    if sleep > 0:
        time.sleep(sleep)
```

For video files, the reader sleeps between frames to respect the original video's FPS. Without this, a video file would be read at maximum speed (potentially 500+ FPS), flooding the inference worker with frames.

---

## InferenceWorker Class

This is the most complex component. It runs the heavy CPU work (detection, tracking, recognition) in a dedicated background thread without blocking the display.

### Architecture: Drop-If-Busy

```python
def submit(self, frame):
    with self.lock:
        if self.busy:
            return          # ← Drop the frame silently
        self.input_frame = frame
    self.new_frame_event.set()
```

If inference is still running when a new frame arrives, the new frame is **silently dropped**. This is intentional — it means the display loop and stream thread never block waiting for inference. The worker always processes the most recently submitted frame (not a queue of old frames).

### Internal Processing Pipeline

Each time the worker wakes up, it runs this full pipeline on the latest frame:

#### Step 1 — Downscale

```python
small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
```

Frame scaled to 50% for faster detection.

#### Step 2 — InsightFace Detection

```python
raw_faces = face_app.get(small)
raw_faces = [f for f in raw_faces if float(f.det_score) >= MIN_DET_CONF]
raw_faces = raw_faces[:MAX_FACES]
faces = pre_nms(raw_faces)
```

Detect all faces, drop weak ones, cap at 10, apply Layer 1 NMS.

#### Step 3 — Build DeepSORT Inputs

```python
for face in faces:
    x1, y1, x2, y2 = (face.bbox / FRAME_SCALE).astype(int)  # Scale back up
    ...
    raw_dets.append(([x1, y1, w, h], conf, "face"))
    embeds.append(emb)
```

Bounding boxes are scaled back to original resolution. Embeddings are L2-normalized and attached to each detection so DeepSORT uses them for appearance-based track matching.

#### Step 4 — DeepSORT Update

```python
tracks = tracker.update_tracks(raw_dets, embeds=embeds, frame=frame)
```

DeepSORT returns the current state of all tracks — new, confirmed, and aged. Only **confirmed** tracks (that survived `n_init=2` frames) are processed further.

#### Step 5 — FAISS Recognition Per Track

```python
is_first     = track_id in self.never_queried or cached is None
should_query = embedding is not None and (
    is_first or now - cached[2] >= RECOG_INTERVAL
)
```

**Two cases trigger a FAISS query:**

1. **First-ever query** (`is_first=True`) — brand new track, never been identified. Fires immediately on the first confirmed frame, regardless of `RECOG_INTERVAL`. This is the fast-pass fix: even a student in frame for 0.5 seconds gets recognized immediately.

2. **Interval elapsed** — the track was previously recognized but `RECOG_INTERVAL` seconds have passed. Allows re-identification in case of initial mismatch.

**If neither condition is met:** The cached identity from the last FAISS query is reused. This avoids querying FAISS on every frame for faces that are already identified.

#### Step 6 — Post-Track NMS

```python
final_results = post_nms(raw_results)
```

Layer 2 NMS: IoU + center-distance suppression on all confirmed track boxes.

### Track Cache Management

```python
self.track_cache = {k: v for k, v in self.track_cache.items() if k in all_active}
self.never_queried = {k for k in self.never_queried if k in all_active}
```

Dead tracks (aged out of DeepSORT) are removed from the cache to prevent unbounded memory growth. The `never_queried` set tracks which confirmed track IDs have never been queried — new IDs are added here, and removed once their first FAISS query completes.

---

## Main Display Loop

```python
while stream.is_running():
    frame = stream.read()
    frame_count += 1

    if frame_count % PROCESS_EVERY_N == 0:
        worker.submit(frame)

    faces = worker.get_results()
    for (x1, y1, x2, y2, label, color) in faces:
        cv2.rectangle(...)
        cv2.putText(...)

    cv2.imshow("CCTV Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

The main thread runs as fast as possible. It:
1. Reads the latest frame from `CameraStream`
2. Submits every frame to the inference worker (PROCESS_EVERY_N=1)
3. Gets the **most recently computed** face results (may be from a previous frame's inference — this is fine, boxes are stable due to tracking)
4. Draws boxes and labels
5. Shows the frame
6. Checks for 'Q' keypress to quit

**Color coding:**
- 🟢 Green box = recognized student
- 🔵 Blue box = unknown person (no FAISS match within threshold)

---

## Threading Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MAIN THREAD (Display)                  │
│  stream.read() → worker.submit() → worker.get_results()  │
│  → Draw boxes → imshow() → waitKey()                     │
│  Runs at full display FPS (30+)                           │
└──────────────────────┬───────────────────────────────────┘
                       │ submits frame
                       ▼
┌──────────────────────────────────────────────────────────┐
│              INFERENCE THREAD (InferenceWorker)           │
│  InsightFace detect → Pre-NMS → DeepSORT → FAISS         │
│  → Post-NMS → write output_faces                         │
│  Runs at inference FPS (5–15 FPS depending on CPU)       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│              STREAM THREAD (CameraStream)                 │
│  cap.read() → store latest frame                         │
│  Runs at source FPS (30 FPS or file speed)               │
└──────────────────────────────────────────────────────────┘
```

All shared state is protected by `threading.Lock()`. The inference worker uses a `threading.Event` to sleep when idle and wake immediately when a new frame is submitted.

---

## NMS Dual-Layer Strategy

| Layer | When | What It Catches | Method |
|---|---|---|---|
| **Layer 1 Pre-NMS** | Before tracker, on raw detections | InsightFace double-detections of same face | IoU > 0.30 |
| **Layer 2 Post-NMS** | After tracker, on confirmed tracks | Ghost tracks from fast movement | IoU > 0.45 OR center distance < 80px |

---

## Performance Tuning Reference

| Setting | Current | Lower Value Effect | Higher Value Effect |
|---|---|---|---|
| `THRESHOLD` | 0.9 | Fewer false matches (stricter) | More matches, higher false positive risk |
| `DET_SIZE` | 320×320 | Faster detection, misses small faces | Slower, catches smaller faces |
| `FRAME_SCALE` | 0.5 | Faster detection | More accurate on large frames |
| `RECOG_INTERVAL` | 0.5s | More FAISS queries, higher CPU | Fewer queries, stale identity longer |
| `MIN_DET_CONF` | 0.50 | Accepts more faces (risk of false det) | Only high-confidence faces accepted |
| `max_age` | 5 | Ghost tracks die faster | Longer persistence, more ghost risk |
| `n_init` | 2 | Tracks confirmed faster | Slower confirmation, more robust |
| `max_iou_distance` | 0.95 | Stricter matching, more new tracks | Handles faster motion |
| `CENTER_DIST_PX` | 80px | More aggressive ghost suppression | Risk of suppressing real nearby faces |
