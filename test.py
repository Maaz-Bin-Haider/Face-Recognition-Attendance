import cv2

# Replace with your phone's IP webcam URL


cap = cv2.VideoCapture(r"C:\Users\SWISS TECH\Downloads\WhatsApp Video 2026-02-16 at 4.15.02 PM.mp4")


if not cap.isOpened():
    print("Cannot open IP camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    cv2.imshow('Phone Camera', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
