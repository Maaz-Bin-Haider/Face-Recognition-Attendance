<div align="center">

# 🎯 Face Recognition Attendance System
### *Automated Attendance Marking using InsightFace & Deep Learning*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![InsightFace](https://img.shields.io/badge/InsightFace-Buffalo__L-FF6B35?style=for-the-badge&logo=opencv&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

> An automated attendance system that identifies registered students from **live video or recorded footage** using state-of-the-art face recognition. Built with **InsightFace (buffalo_l)** for face detection and 512-dimensional embedding extraction, and **YOLOv8** for high-accuracy face detection — with cosine-distance matching against a pre-registered student database.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Pipeline](#-pipeline)
- [Student Registration](#-student-registration)
- [Recognition & Attendance](#-recognition--attendance)
- [Matching Algorithm](#-matching-algorithm)
- [Key Design Decisions](#-key-design-decisions)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)

---

## 🌟 Overview

Traditional attendance systems are slow and prone to proxy fraud. This system solves both problems by:

- **Automatically detecting** every face in a video frame
- **Extracting a 512-dimensional face embedding** for each detected face
- **Matching embeddings** against a pre-registered student database using L2 distance
- **Marking attendance** for recognized individuals in real-time

The system is built to be **robust across variations** in lighting, angle, and appearance by storing **multiple embeddings per student** during registration — so recognition does not rely on a single reference image.

---

## 🔍 How It Works

```
VIDEO FRAME
     │
     ▼
┌─────────────────────────┐
│  InsightFace (buffalo_l)│  ← Detects ALL faces in the frame
│  Face Detection         │     Returns bounding boxes + 5-point landmarks
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Face Alignment         │  ← Landmarks used to normalize face orientation
│  (Built into buffalo_l) │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Embedding Extraction   │  ← 512-dimensional face vector generated per face
│  (ArcFace backbone)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  L2 Distance Matching   │  ← Compare against all stored student embeddings
│  + Threshold Filter     │     Normalized vectors → Euclidean distance
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
RECOGNIZED          UNKNOWN
(Green Box)         (Red Box)
Name + Confidence   Distance shown
```

---

## 🏗️ System Architecture

### Two-Phase System

| Phase | Script | Purpose |
|-------|--------|---------|
| **Registration** | `register_multi_embedding.py` | Capture student faces → extract multiple embeddings → save to `.pkl` |
| **Recognition** | `recognition.py` | Load student database → process video → identify faces → mark attendance |

### Data Storage

Students are stored as **serialized `.pkl` files**, one per student:

```python
# Student pickle file structure
{
    "name":       "Ahmed Khan",
    "student_id": "STU-001",
    "embeddings": [  # Multiple embeddings for robustness
        np.array([...]),   # 512-dim vector — frontal
        np.array([...]),   # 512-dim vector — slight left angle
        np.array([...]),   # 512-dim vector — slight right angle
    ]
}
```

> Backwards compatible with older single-embedding format (`student['embedding']`)

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Face Detection** | InsightFace `buffalo_l` | Detect faces + extract landmarks in full frames |
| **Face Detection (Alt)** | YOLOv8 (custom `model.pt`) | High-speed face detection (explored in development) |
| **Face Alignment** | SCRFD via InsightFace | 5-point landmark alignment before embedding |
| **Embedding Model** | ArcFace (via `buffalo_l`) | 512-dim face representation |
| **Matching** | NumPy L2 Distance | Cosine-normalized Euclidean distance |
| **Video Processing** | OpenCV (cv2) | Frame capture, annotation, output writing |
| **Model Storage** | Python Pickle | Per-student `.pkl` embedding files |

---

## 🔄 Pipeline

### Registration Phase

```
Student sits in front of camera
          │
          ▼
Capture multiple photos from different angles
          │
          ▼
InsightFace detects face in each image
          │
          ▼
Extract 512-dim ArcFace embedding per image
          │
          ▼
Store all embeddings in student_<name>.pkl
```

### Recognition Phase

```
Load all student_*.pkl files from folder
          │
          ▼
Open video file (or webcam stream)
          │
          ▼
Process every 5th frame (performance optimization)
│   └── Reuse last annotated frame for skipped frames
          │
          ▼
InsightFace detects all faces in full frame
          │
          ▼
For each detected face:
    ├── Extract 512-dim embedding
    ├── Normalize embedding (L2 norm)
    ├── Compare against ALL stored embeddings of ALL students
    ├── Find minimum L2 distance
    └── If distance < threshold (1.0): RECOGNIZED ✓
        Else: UNKNOWN ✗
          │
          ▼
Annotate frame:
    ├── Green box + Name + Confidence → Recognized
    └── Red box + "Unknown" + Distance → Unknown
          │
          ▼
Write annotated frame to output video
```

---

## 👤 Student Registration

The `register_multi_embedding.py` script:

1. Takes **multiple face images** per student (from different angles/lighting)
2. Runs each image through InsightFace to extract a 512-dimensional embedding
3. Stores **all embeddings together** in one `.pkl` file per student
4. Supports re-registration (adding more embeddings to existing students)

**Why multiple embeddings?**
A single reference image fails when the student's face angle or lighting differs at attendance time. Multiple embeddings capture the natural variation in a person's appearance and dramatically improve recognition accuracy.

---

## 🎯 Recognition & Attendance

The `recognition.py` script processes video with the following behavior:

### Frame Processing Strategy
- **Every 5th frame** is fully processed (detection + recognition)
- **Skipped frames** reuse the last annotated result — significantly reduces CPU/GPU load without visible quality loss

### Output
- Annotated output video saved as `output_recognition.mp4`
- Real-time console logs showing per-frame recognition results:
  ```
  Frame 45: ✓ Ahmed Khan | Conf: 87.3% | Dist: 0.4821
  Frame 50: ✗ Unknown | Dist: 1.2341
  [Progress: 25.0% | Detections: 34 | Recognized: 28]
  ```
- Final summary with total detections and recognition count

---

## 🔢 Matching Algorithm

```python
def find_match(embedding, threshold=1.0):
    # Step 1: Normalize incoming embedding
    embedding = embedding / np.linalg.norm(embedding)

    best_match = None
    best_distance = float('inf')

    for student in students:
        # Step 2: Support both single and multi-embedding formats
        stored_list = student.get('embeddings', [student['embedding']])

        for stored_emb in stored_list:
            # Step 3: Normalize stored embedding
            stored_emb = stored_emb / np.linalg.norm(stored_emb)

            # Step 4: L2 distance on normalized vectors
            distance = np.linalg.norm(embedding - stored_emb)

            if distance < best_distance:
                best_distance = distance
                best_match = student

    # Step 5: Apply threshold
    if best_distance < threshold:
        confidence = 1 - (best_distance / 2)   # Map to 0–1 confidence
        return best_match, confidence, best_distance

    return None, 0, best_distance
```

**Threshold = 1.0** on normalized vectors is equivalent to a cosine similarity check. Lower distance = more similar faces.

---

## 🧠 Key Design Decisions

| Decision | Reason |
|----------|--------|
| **InsightFace buffalo_l** | State-of-the-art accuracy; handles face detection + alignment + embedding in one pipeline |
| **YOLOv8 explored** | Faster detection; explored as a two-stage pipeline (YOLO detect → InsightFace embed) |
| **Every 5th frame processing** | Balances performance vs. accuracy; real-time feel without maxing out CPU |
| **Multiple embeddings per student** | Handles pose/lighting variation far better than a single reference image |
| **L2 on normalized vectors** | Equivalent to cosine distance; scale-invariant and more reliable for face embeddings |
| **Per-student `.pkl` files** | Simple, portable, no database dependency; easy to add/remove students |
| **Backwards-compatible loader** | Supports both old (`embedding`) and new (`embeddings`) pkl formats |

---

## 📁 Project Structure

```
Face-Recognition-Attendance/
│
├── register_multi_embedding.py   # Student registration script
├── recognition.py                # Main recognition + attendance script
├── model.pt                      # YOLOv8 custom face detection weights
│
├── students/                     # Student embedding database
│   ├── student_Ahmed_Khan.pkl
│   ├── student_Ali_Hassan.pkl
│   └── student_*.pkl
│
└── outputs/
    └── output_recognition.mp4   # Annotated output video
```

---

## 🚀 Setup & Usage

### Prerequisites

```bash
pip install opencv-python insightface ultralytics numpy onnxruntime
```

### Register a Student

```bash
python register_multi_embedding.py
# Follow prompts: enter student name, ID, and provide face images
```

### Run Recognition on Video

```python
# Edit recognition.py — update these paths:
students_folder = r"path/to/your/students/folder"
video_path = r"path/to/your/video.mp4"

# Then run:
python recognition.py
```

### Output

- Annotated video saved to `students_folder/output_recognition.mp4`
- Console shows live recognition results per frame
- Final summary: total detections, total recognized

---

## 🔮 Planned Features

- [ ] Live webcam stream support
- [ ] Django web dashboard for attendance records
- [ ] Export attendance to Excel/CSV
- [ ] Date and time stamping per recognized student
- [ ] Multi-camera support
- [ ] GPU acceleration (CUDA) for faster processing

---

<div align="center">

**Built with ❤️ using InsightFace, OpenCV & YOLOv8**

</div>
