import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np

st.set_page_config(page_title="Blink Verification 👁️", layout="centered")
st.title("Blink Verification Demo 👁️")

FRAME_WINDOW = st.image([])
status_placeholder = st.empty()

# Mediapipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# EAR function (eye aspect ratio)
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # EAR calculation: (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Eye indices for mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

# Countdown before challenge
status_placeholder.markdown("### ⏳ Blink challenge starts in 3 seconds...")
for i in range(3, 0, -1):
    status_placeholder.markdown(f"### ⏳ Blink in {i}...")
    time.sleep(1)

status_placeholder.markdown("### 👀 Please BLINK within 5 seconds!")

challenge_start_time = time.time()
challenge_duration = 5
blink_detected = False
eyes_closed_once = False

EAR_THRESH = 0.21  # threshold for closed eyes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            leftEAR = eye_aspect_ratio(landmarks.landmark, LEFT_EYE, w, h)
            rightEAR = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, w, h)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESH:  # eyes closed
                eyes_closed_once = True
            elif eyes_closed_once and ear >= EAR_THRESH:  # reopened → blink
                blink_detected = True

            circle_color = (0, 255, 0) if blink_detected else (0, 0, 255)
            cv2.circle(frame, (w // 2, h // 2), 100, circle_color, 3)

    FRAME_WINDOW.image(frame, channels="BGR")

    if time.time() - challenge_start_time > challenge_duration:
        break

cap.release()

# Final result
if blink_detected:
    status_placeholder.markdown("### ✅ Verification Successful!")
else:
    status_placeholder.markdown("### ❌ Verification Failed. Please try again.")
