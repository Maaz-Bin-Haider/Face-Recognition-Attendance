# # register_multi_embedding.py
# import cv2
# from insightface.app import FaceAnalysis
# import pickle
# import numpy as np
# import os

# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=-1, det_size=(640, 640))

# student_id = input("Enter Student ID (e.g., S001): ").strip()
# student_name = input("Enter Student Name: ").strip()
# target_count = int(input("How many embeddings to collect? (recommended 100-200): ").strip())

# video_paths = [r"C:\Users\SWISS TECH\Downloads\IMG_7178.MOV"]

# all_embeddings = []

# for video_path in video_paths:
#     if len(all_embeddings) >= target_count:
#         break

#     cap = cv2.VideoCapture(video_path)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_count = 0
#     print(f"\nScanning: {os.path.basename(video_path)}")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % 5 != 0:
#             continue

#         faces = app.get(frame)
#         if len(faces) > 0:
#             face = faces[0]
#             if face.det_score > 0.6:
#                 # Normalize embedding before storing
#                 emb = face.embedding / np.linalg.norm(face.embedding)

#                 # Avoid storing near-duplicate embeddings
#                 # Only add if it's different enough from existing ones
#                 if len(all_embeddings) == 0:
#                     all_embeddings.append(emb)
#                 else:
#                     dists = [np.linalg.norm(emb - e) for e in all_embeddings]
#                     if min(dists) > 0.1:  # Only add if sufficiently different
#                         all_embeddings.append(emb)
#                         print(f"  Collected {len(all_embeddings)}/{target_count} embeddings", end='\r')

#                 if len(all_embeddings) >= target_count:
#                     break

#     cap.release()

# print(f"\n\nTotal embeddings collected: {len(all_embeddings)}")

# if len(all_embeddings) == 0:
#     print("No embeddings collected!")
#     exit()

# student_data = {
#     'student_id': student_id,
#     'name': student_name,
#     'embeddings': all_embeddings,       # List of many embeddings
#     'num_samples': len(all_embeddings)
# }

# students_folder = r"C:\Users\SWISS TECH\Documents\Maaz\attendence"
# filename = os.path.join(students_folder, f"student_{student_id}.pkl")
# with open(filename, 'wb') as f:
#     pickle.dump(student_data, f)

# print(f"✓ Registered {student_name} with {len(all_embeddings)} embeddings")
# print(f"  Saved to: {filename}")




import cv2
from insightface.app import FaceAnalysis
import pickle
import numpy as np
import os
import faiss

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
STUDENTS_FOLDER = r"C:\Users\SWISS TECH\Documents\Maaz\Face recognition attendance\Scripts"
INDEX_FILE = "face_index.faiss"
LABELS_FILE = "labels.pkl"
# ─────────────────────────────────────────────

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

student_id = input("Enter Student ID (e.g., S001): ").strip()
student_name = input("Enter Student Name: ").strip()
target_count = int(input("How many embeddings to collect? (recommended 100-200): ").strip())

video_paths = [r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-26 at 1.40.50 PM.mp4"]

all_embeddings = []

# ─────────────────────────────────────────────
# COLLECT EMBEDDINGS
# ─────────────────────────────────────────────
for video_path in video_paths:
    if len(all_embeddings) >= target_count:
        break

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    print(f"\nScanning: {os.path.basename(video_path)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        faces = app.get(frame)
        if len(faces) > 0:
            face = faces[0]

            if face.det_score > 0.6:
                emb = face.embedding / np.linalg.norm(face.embedding)

                if len(all_embeddings) == 0:
                    all_embeddings.append(emb)
                else:
                    dists = [np.linalg.norm(emb - e) for e in all_embeddings]
                    if min(dists) > 0.1:
                        all_embeddings.append(emb)
                        print(f"  Collected {len(all_embeddings)}/{target_count}", end='\r')

                if len(all_embeddings) >= target_count:
                    break

    cap.release()

print(f"\nTotal embeddings collected: {len(all_embeddings)}")

if len(all_embeddings) == 0:
    print("No embeddings collected!")
    exit()

# Convert to float32 matrix
new_embeddings = np.array(all_embeddings).astype('float32')

# ─────────────────────────────────────────────
# SAVE STUDENT BACKUP FILE (optional but good)
# ─────────────────────────────────────────────
student_data = {
    'student_id': student_id,
    'name': student_name,
    'embeddings': all_embeddings,
    'num_samples': len(all_embeddings)
}

filename = os.path.join(STUDENTS_FOLDER, f"student_{student_id}.pkl")
with open(filename, 'wb') as f:
    pickle.dump(student_data, f)

print(f"✓ Student backup saved: {filename}")

# ─────────────────────────────────────────────
# UPDATE FAISS INDEX
# ─────────────────────────────────────────────
dimension = 512

# If index exists → load
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)

    with open(LABELS_FILE, "rb") as f:
        labels = pickle.load(f)

    print("✓ Existing FAISS index loaded")

else:
    index = faiss.IndexFlatL2(dimension)
    labels = []
    print("✓ Creating new FAISS index")

# Add new embeddings
index.add(new_embeddings)

# Add labels (store full student info)
for _ in range(len(new_embeddings)):
    labels.append({
        "student_id": student_id,
        "name": student_name
    })

# Save updated index + labels
faiss.write_index(index, INDEX_FILE)

with open(LABELS_FILE, "wb") as f:
    pickle.dump(labels, f)

print("✓ FAISS index updated successfully")
print(f"✓ Registered {student_name} ({student_id}) with {len(all_embeddings)} embeddings")