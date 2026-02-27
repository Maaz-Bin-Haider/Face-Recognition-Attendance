# 📋 Face Enrollment Script — Technical Documentation

> **File:** `enrollment.py`  
> **Purpose:** Register new students into the FAISS face recognition database by extracting embeddings from video footage.  
> **Version:** 1.0  
> **Author:** SWISS TECH Team

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies & Libraries](#dependencies--libraries)
3. [Configuration Block](#configuration-block)
4. [Model Initialization](#model-initialization)
5. [User Input Collection](#user-input-collection)
6. [Embedding Collection Loop](#embedding-collection-loop)
7. [Student Backup File](#student-backup-file)
8. [FAISS Index Management](#faiss-index-management)
9. [Data Flow Summary](#data-flow-summary)
10. [Key Design Decisions](#key-design-decisions)
11. [Output Files](#output-files)
12. [How to Run](#how-to-run)

---

## Overview

This script is the **first step** of the face recognition attendance system. Before any student can be recognized in live CCTV footage, they must be enrolled here. The script:

1. Takes a video file of the student as input
2. Extracts high-quality face embeddings from video frames using InsightFace (buffalo_l model)
3. Deduplicates embeddings to ensure diversity (avoids storing near-identical frames)
4. Saves embeddings to a per-student `.pkl` backup file
5. Appends all new embeddings to the shared FAISS index for fast nearest-neighbor lookup at recognition time

The design philosophy is **quality over quantity** — embeddings are filtered by detection confidence and diversity distance to ensure the index only contains meaningful, varied representations of each face.

---

## Dependencies & Libraries

```python
import cv2          # Video frame reading and display
import pickle       # Serializing Python objects to disk (.pkl files)
import numpy as np  # Numerical operations on embedding vectors
import os           # File path operations
import faiss        # Facebook AI Similarity Search — vector index
from insightface.app import FaceAnalysis  # Face detection + embedding model
```

| Library | Role | Why This One |
|---|---|---|
| `opencv-cv2` | Video decoding, frame extraction | Industry standard, fast, reliable |
| `insightface` (buffalo_l) | Face detection + 512-dim embedding | State-of-art accuracy, runs on CPU |
| `faiss` | Vector similarity search index | Millisecond-speed search even with thousands of embeddings |
| `numpy` | L2 normalization, distance math | Required for cosine-like comparisons |
| `pickle` | Saving Python dicts/lists to disk | Simple, no external format dependency |

---

## Configuration Block

```python
STUDENTS_FOLDER = r"C:\Users\SWISS TECH\Documents\Maaz\Face recognition attendance\Scripts"
INDEX_FILE  = "face_index.faiss"
LABELS_FILE = "labels.pkl"
```

| Variable | Purpose |
|---|---|
| `STUDENTS_FOLDER` | Directory where individual student `.pkl` backup files are saved |
| `INDEX_FILE` | Shared FAISS binary index file — all students' embeddings live here |
| `LABELS_FILE` | Parallel list of student metadata (ID + name) matching each embedding in the FAISS index |

> ⚠️ **Important:** `INDEX_FILE` and `LABELS_FILE` must always stay in sync. Every time you add N embeddings to the FAISS index, you must also append N matching label entries to `labels.pkl`. This script handles this automatically.

---

## Model Initialization

```python
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
```

**`buffalo_l`** is InsightFace's largest, most accurate pre-trained model bundle. It includes:
- **RetinaFace** detector — finds face bounding boxes
- **2D landmark detector** — finds 5 facial keypoints for alignment
- **ArcFace** recognizer — produces the 512-dimensional embedding

`det_size=(640, 640)` is set **larger here than in the recognition script (320×320)**. This is intentional — during enrollment we are not running in real time, so we can afford the slower but more precise detection to get the cleanest possible embeddings for the database.

`ctx_id=-1` means CPU inference (no GPU required).

---

## User Input Collection

```python
student_id    = input("Enter Student ID (e.g., S001): ").strip()
student_name  = input("Enter Student Name: ").strip()
target_count  = int(input("How many embeddings to collect? (recommended 100-200): ").strip())
```

The script is **interactive** — it collects three pieces of information before processing:

| Input | Type | Used For |
|---|---|---|
| `student_id` | String (e.g., `S001`) | Unique key in FAISS labels, backup filename |
| `student_name` | String (e.g., `Ali Hassan`) | Display label in recognition output |
| `target_count` | Integer (100–200 recommended) | How many diverse embeddings to collect before stopping |

**Why 100–200 embeddings?** This range gives the FAISS index enough coverage of the student's face across different angles, lighting conditions, and expressions from the video — without being so large that it bloats the index or introduces noise.

---

## Embedding Collection Loop

This is the core processing section. It iterates over video frames and builds `all_embeddings`.

### Frame Sampling

```python
frame_count += 1
if frame_count % 5 != 0:
    continue
```

Every **5th frame** is processed. This is a deliberate skip — consecutive frames are nearly identical and would produce duplicate embeddings. Sampling every 5th frame at 30 FPS gives ~6 candidate frames per second.

### Detection & Quality Gate

```python
faces = app.get(frame)
if len(faces) > 0:
    face = faces[0]
    if face.det_score > 0.6:
```

- `app.get(frame)` runs the full detection + alignment + embedding pipeline
- `faces[0]` takes only the **first detected face** — assumes one person per enrollment video
- `det_score > 0.6` is a confidence threshold — only high-quality, clearly-detected faces are accepted. Blurry, partially visible, or low-light detections are discarded here.

### Embedding Normalization

```python
emb = face.embedding / np.linalg.norm(face.embedding)
```

The raw 512-dimensional ArcFace embedding is **L2-normalized** to unit length. This converts the L2 distance metric used by FAISS into a cosine-similarity-equivalent comparison. After normalization, a distance of `0.0` means identical faces, and `2.0` means completely opposite (maximum distance on a unit hypersphere).

### Diversity Filter

```python
if len(all_embeddings) == 0:
    all_embeddings.append(emb)
else:
    dists = [np.linalg.norm(emb - e) for e in all_embeddings]
    if min(dists) > 0.1:
        all_embeddings.append(emb)
```

This is a **critical quality-control step**. Before adding any new embedding, it computes the L2 distance between the candidate and every already-stored embedding. The embedding is only accepted if its minimum distance to all existing embeddings exceeds **0.1**.

This threshold (0.1) means: "only store this embedding if it represents a meaningfully different view of the face." It prevents the collection from being dominated by frames where the person held still — ensuring the database covers diverse poses, expressions, and lighting.

---

## Student Backup File

```python
student_data = {
    'student_id': student_id,
    'name': student_name,
    'embeddings': all_embeddings,
    'num_samples': len(all_embeddings)
}
filename = os.path.join(STUDENTS_FOLDER, f"student_{student_id}.pkl")
with open(filename, 'wb') as f:
    pickle.dump(student_data, f)
```

Each student gets their own `.pkl` backup file (e.g., `student_S001.pkl`). This serves two purposes:

1. **Recovery:** If the FAISS index is corrupted or accidentally deleted, you can rebuild it from scratch using all the individual student `.pkl` files without re-running enrollment videos.
2. **Audit:** You can inspect any student's stored embeddings count and verify enrollment quality.

---

## FAISS Index Management

### Load or Create

```python
dimension = 512

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(LABELS_FILE, "rb") as f:
        labels = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dimension)
    labels = []
```

The script checks if a FAISS index already exists:
- **Exists:** Load the existing index and its labels — ready to append new student data
- **Does not exist:** Create a new `IndexFlatL2(512)` index from scratch

`IndexFlatL2` performs exact (brute-force) L2 distance search across all 512-dimensional embeddings. It guarantees finding the true nearest neighbor, at the cost of O(N) search time. For typical school-scale databases (a few thousand embeddings), this is perfectly fast.

### Adding Embeddings

```python
new_embeddings = np.array(all_embeddings).astype('float32')
index.add(new_embeddings)

for _ in range(len(new_embeddings)):
    labels.append({"student_id": student_id, "name": student_name})
```

All collected embeddings are added to the FAISS index in one batch call. For each embedding added, one matching label dict is appended to the `labels` list. The position of each label in the list **exactly matches** the row index in the FAISS index — this is the lookup mechanism used during recognition.

### Saving

```python
faiss.write_index(index, INDEX_FILE)
with open(LABELS_FILE, "wb") as f:
    pickle.dump(labels, f)
```

Both files are saved atomically at the end. The FAISS index is written as a binary file, and labels are pickled.

---

## Data Flow Summary

```
Video File (.MOV/.MP4)
        │
        ▼
  Frame Sampling (every 5th frame)
        │
        ▼
  InsightFace Detection + Quality Gate (det_score > 0.6)
        │
        ▼
  ArcFace Embedding (512-dim) → L2 Normalize
        │
        ▼
  Diversity Filter (min distance > 0.1 from all existing)
        │
        ▼
  Collect until target_count reached
        │
        ├──► student_S001.pkl  (backup)
        │
        └──► face_index.faiss  +  labels.pkl  (shared database)
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `det_size=(640,640)` for enrollment | Slower but maximally accurate — no real-time constraint |
| Skip every 5 frames | Avoids near-duplicate embeddings from consecutive frames |
| Diversity filter (>0.1 distance) | Ensures embeddings cover different poses/lighting, not the same face repeated |
| Separate `.pkl` per student | Enables index rebuild without re-enrollment |
| `IndexFlatL2` (exact search) | Guarantees correct nearest-neighbor at school-scale sizes |
| L2 normalization before storage | Converts L2 distance into cosine-equivalent metric for ArcFace embeddings |

---

## Output Files

| File | Location | Contents |
|---|---|---|
| `student_<ID>.pkl` | `STUDENTS_FOLDER` | Dict with student ID, name, embeddings list, count |
| `face_index.faiss` | Working directory | Binary FAISS index — all embeddings from all students |
| `labels.pkl` | Working directory | Python list of `{"student_id": ..., "name": ...}` dicts, one per embedding |

---

## How to Run

```bash
python enrollment.py
```

Then follow the prompts:
```
Enter Student ID (e.g., S001): S012
Enter Student Name: Sara Ahmed
How many embeddings to collect? (recommended 100-200): 150
```

The script will scan the video, print progress, and confirm when done:
```
✓ Student backup saved: ...student_S012.pkl
✓ FAISS index updated successfully
✓ Registered Sara Ahmed (S012) with 147 embeddings
```

> **Note:** Change `video_paths` in the script to point to the student's enrollment video before running.
