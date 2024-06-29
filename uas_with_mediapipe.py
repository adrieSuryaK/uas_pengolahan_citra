from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch
import mediapipe as mp

# Check CUDA availability
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. You are using GPU.")
    else:
        print("CUDA is not available. You are using CPU.")

check_cuda()

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# For Webcam
cap = cv2.VideoCapture(1)
# For Video
# cap = cv2.VideoCapture("deteksi.mp4")
cap.set(3, 720)
cap.set(4, 480)

model = YOLO("best.pt")
classNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Switch', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

detected_text = ""
last_detected_time = {class_name: 0 for class_name in classNames}
DETECTION_COOLDOWN = 5  # seconds

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_mediapipe = hands.process(img_rgb)

    results_yolo = model(img, stream=True)

    # Process YOLO results
    for r in results_yolo:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if conf >= 0.8:
                current_time = time.time()
                if current_time - last_detected_time[class_name] > DETECTION_COOLDOWN:
                    detected_text += class_name + " "
                    last_detected_time[class_name] = current_time
                    # Reset detected text if it reaches 27 characters
                    if len(detected_text.split()) >= 27:
                        detected_text = ""
                cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=3, thickness=1)

    # Process Mediapipe results
    if results_mediapipe.multi_hand_landmarks:
        for hand_landmarks in results_mediapipe.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Add detected text to frame
    cv2.putText(img, detected_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("SIBI Translator", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
