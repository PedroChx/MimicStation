"""
Mimic – Dual Hand Demo
Reconoce dos manos, asigna región izquierda/derecha y muestra el gesto detectado.
Requisitos:
    pip install mediapipe opencv-python
Coloca gesture_recognizer.task en el mismo directorio.
"""

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_task
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ---------- Configuración ----------
MODEL_PATH = "../Mimic/gesture_recognizer.task"
GESTURES   = {"Open_Palm", "Closed_Fist", "Pointing_Up", "Thumb_Up", "Victory"}

BaseOptions      = mp_task.BaseOptions
GestureRecOpt    = vision.GestureRecognizerOptions
GestureRecognizer= vision.GestureRecognizer
VisionRunningMode= vision.RunningMode
drawing_utils    = mp.solutions.drawing_utils
hands_connections= mp.solutions.hands.HAND_CONNECTIONS

# ---------- Inicializar recognizer ----------
options = GestureRecOpt(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
recognizer = GestureRecognizer.create_from_options(options)

# ---------- Captura de cámara ----------
cap   = cv2.VideoCapture(0)
start = time.time()

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break

    # 1. Convertir a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    ts_ms     = int((time.time() - start) * 1000)

    # 2. Inferencia
    result = recognizer.recognize_for_video(mp_image, ts_ms)

    h, w, _ = frame_bgr.shape
    mid_x   = w // 2

    # 3. Guía visual: línea divisoria y etiquetas
    cv2.line(frame_bgr, (mid_x, 0), (mid_x, h), (0, 255, 255), 2)
    cv2.putText(frame_bgr, "LEFT",  (mid_x // 2 - 40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame_bgr, "RIGHT", (mid_x + mid_x // 2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 4. Procesar cada mano detectada
    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        # --- 4.1 Dibujar landmarks si la mano tiene los 21 puntos ---
        if len(hand_landmarks) == 21:
            proto = landmark_pb2.NormalizedLandmarkList()
            for lm in hand_landmarks:
                proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
            drawing_utils.draw_landmarks(frame_bgr, proto, hands_connections)

        # --- 4.2 Determinar región (LEFT / RIGHT) ---
        cx = int(sum(p.x for p in hand_landmarks) / len(hand_landmarks) * w)
        cy = int(sum(p.y for p in hand_landmarks) / len(hand_landmarks) * h)
        region = "LEFT" if cx < mid_x else "RIGHT"

        # --- 4.3 Etiquetar gesto principal ---
        if result.gestures and idx < len(result.gestures) and result.gestures[idx]:
            g = result.gestures[idx][0]
            if g.category_name in GESTURES and g.score > 0.5:
                label = f"{g.category_name} ({g.score:.2f})"
                color = (0, 150, 255) if region == "LEFT" else (255, 0, 150)
                cv2.putText(frame_bgr, label, (cx - 80, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 5. Mostrar resultado
    cv2.imshow("Mimic – Dual Hand Demo", frame_bgr)
    if cv2.waitKey(1) & 0xFF == 27:   # Esc para salir
        break

# ---------- Limpieza ----------
cap.release()
cv2.destroyAllWindows()