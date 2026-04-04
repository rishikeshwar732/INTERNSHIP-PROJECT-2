"""
Face Tracker - Interview Proctor Version
✔ Stable eye tracking (EMA smoothing)
✔ Face max = 5
✔ Eye max = 8
✔ Cooldown-based counting
✔ NO FACE = violation
✔ INTERVIEW CANCELS if any limit exceeded
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# ---------------- CONFIG ----------------
SMOOTHING = 0.8
EYE_DEADZONE = 2
EYE_THRESHOLD = 4

MAX_FACE_COUNT = 5
MAX_EYE_COUNT = 8

FACE_COOLDOWN = 1.0
EYE_COOLDOWN = 0.6

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP = 1

# ---------------- GLOBALS ----------------
prev_left_iris = None
prev_right_iris = None

# ---------------- FACE DIRECTION ----------------
def get_head_direction(face_landmarks, frame_shape):
    h, w = frame_shape[:2]

    nose = face_landmarks.landmark[NOSE_TIP]
    left_face = face_landmarks.landmark[234]
    right_face = face_landmarks.landmark[454]

    offset = (nose.x - (left_face.x + right_face.x) / 2) * w
    if offset > w * 0.03:
        return "Face: RIGHT"
    elif offset < -w * 0.03:
        return "Face: LEFT"

    nose_y = nose.y * h
    center_y = (face_landmarks.landmark[152].y +
                face_landmarks.landmark[10].y) * h / 2

    v_offset = nose_y - center_y
    if v_offset > h * 0.02:
        return "Face: DOWN"
    elif v_offset < -h * 0.02:
        return "Face: UP"

    return "Face: CENTER"

# ---------------- EYE DIRECTION ----------------
def get_eye_direction(iris_center, eye_points):
    eye_center = np.mean(eye_points, axis=0)
    dx, dy = iris_center - eye_center

    if abs(dx) < EYE_DEADZONE and abs(dy) < EYE_DEADZONE:
        return "Eyes: CENTER"

    if abs(dx) > abs(dy):
        if dx > EYE_THRESHOLD:
            return "Eyes: RIGHT"
        elif dx < -EYE_THRESHOLD:
            return "Eyes: LEFT"
    else:
        if dy > EYE_THRESHOLD:
            return "Eyes: DOWN"
        elif dy < -EYE_THRESHOLD:
            return "Eyes: UP"

    return "Eyes: CENTER"

# ---------------- LANDMARK EXTRACTION ----------------
def extract_landmarks(face_landmarks, frame_shape):
    h, w = frame_shape[:2]

    def pts(idxs):
        return np.array([
            [int(face_landmarks.landmark[i].x * w),
             int(face_landmarks.landmark[i].y * h)]
            for i in idxs
        ])

    return {
        "left_eye": pts(LEFT_EYE),
        "right_eye": pts(RIGHT_EYE),
        "left_iris": pts(LEFT_IRIS),
        "right_iris": pts(RIGHT_IRIS)
    }

# ---------------- DRAW INFO ----------------
def draw_info(frame, face_dir, eye_dir, face_count, eye_count):
    y = 35
    for text, color in [
        (face_dir, (0, 255, 255)),
        (eye_dir, (0, 255, 0)),
        (f"Face Count: {face_count}/{MAX_FACE_COUNT}", (0, 200, 255)),
        (f"Eye Count: {eye_count}/{MAX_EYE_COUNT}", (0, 200, 0))
    ]:
        cv2.putText(frame, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        y += 30
    return frame

# ---------------- INTERVIEW CANCEL SCREEN ----------------
def show_interview_cancelled(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "INTERVIEW CANCELLED", (60, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    cv2.putText(frame, "Interview Rules Violated", (90, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Interview Proctor", frame)
    cv2.waitKey(1)
    time.sleep(2)

# ---------------- MAIN ----------------
def main():
    global prev_left_iris, prev_right_iris

    cap = cv2.VideoCapture(0)

    face_count = 0
    eye_count = 0

    prev_face = "Face: CENTER"
    prev_eye = "Eyes: CENTER"

    last_face_time = 0
    last_eye_time = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = face_mesh.process(rgb)
            now = time.time()

            # -------- NO FACE --------
            if not res.multi_face_landmarks:
                face_count += 1
                cv2.putText(frame, "NO FACE DETECTED", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                if face_count > MAX_FACE_COUNT:
                    show_interview_cancelled(frame)
                    break

            else:
                fl = res.multi_face_landmarks[0]
                lm = extract_landmarks(fl, frame.shape)

                face_dir = get_head_direction(fl, frame.shape)

                raw_left = np.mean(lm["left_iris"], axis=0)
                raw_right = np.mean(lm["right_iris"], axis=0)

                if prev_left_iris is None:
                    prev_left_iris = raw_left
                    prev_right_iris = raw_right

                left_iris = SMOOTHING * prev_left_iris + (1 - SMOOTHING) * raw_left
                right_iris = SMOOTHING * prev_right_iris + (1 - SMOOTHING) * raw_right

                prev_left_iris = left_iris
                prev_right_iris = right_iris

                left_eye_dir = get_eye_direction(left_iris, lm["left_eye"])
                right_eye_dir = get_eye_direction(right_iris, lm["right_eye"])
                eye_dir = left_eye_dir if left_eye_dir == right_eye_dir else "Eyes: CENTER"

                # -------- COUNTERS --------
                if face_dir != "Face: CENTER" and face_dir != prev_face:
                    if now - last_face_time > FACE_COOLDOWN:
                        face_count += 1
                        last_face_time = now

                if eye_dir != "Eyes: CENTER" and eye_dir != prev_eye:
                    if now - last_eye_time > EYE_COOLDOWN:
                        eye_count += 1
                        last_eye_time = now

                prev_face = face_dir
                prev_eye = eye_dir

                if face_count > MAX_FACE_COUNT or eye_count > MAX_EYE_COUNT:
                    show_interview_cancelled(frame)
                    break

                frame = draw_info(frame, face_dir, eye_dir, face_count, eye_count)

            cv2.imshow("Interview Proctor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()