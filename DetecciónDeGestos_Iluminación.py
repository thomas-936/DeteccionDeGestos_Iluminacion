import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from math import dist, acos, degrees
import time
import serial
import os
import urllib.request

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Inicializar el puerto serie
arduino = serial.Serial('COM4', 9600)  # Cambiar 'COM4' por el puerto adecuado

# Margen de tolerancia para evitar que un dedo cambie de estado con pequeños
# temblores de la mano. Sube este valor si sigue siendo muy sensible.
MARGEN_DEDO = 0.03  # en unidades normalizadas (0 a 1)
UMBRAL_ANGULO_PULGAR = 60  # grados

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Descargar el modelo automáticamente si no existe todavía
if not os.path.exists(MODEL_PATH):
    print("Descargando el modelo hand_landmarker.task (solo la primera vez)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo descargado.")

# Inicializar el detector de manos con la nueva Tasks API 
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.85,
    min_tracking_confidence=0.85,
)
detector = vision.HandLandmarker.create_from_options(options)

# Anotamos la coordenada de cada dedo según la imagen proporcionada por MediaPipe. Cada dedo tiene dos puntos de interés: el punto medio y la punta.
dedos = {
    "pulgar": [2, 4],
    "indice": [6, 8],
    "anular": [10, 12],
    "mayor": [14, 16],
    "menique": [18, 20],
}

# ============================================================
# FUNCIONES
# ============================================================

def coord_x(landmarks, marcador):
    return landmarks[marcador].x


def coord_y(landmarks, marcador):
    return landmarks[marcador].y


def calcular_angulo(x1, y1, x2, y2, x3, y3):
    """
    Calcula el ángulo entre tres puntos.
    """
    angulo = acos(
        ((x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2))
        / (dist([x1, y1], [x2, y2]) * dist([x2, y2], [x3, y3]))
    )
    return degrees(angulo)


def detectarDedo(landmarks):
    try:
        x_palma = coord_x(landmarks, 0)
        y_palma = coord_y(landmarks, 0)
        cerrados = []

        for dedo, (medio, punta) in dedos.items():
            x_medio = coord_x(landmarks, medio)
            y_medio = coord_y(landmarks, medio)
            x_punta = coord_x(landmarks, punta)
            y_punta = coord_y(landmarks, punta)

            if dedo != "pulgar":
                d_medio = dist([x_palma, y_palma], [x_medio, y_medio])
                d_punta = dist([x_palma, y_palma], [x_punta, y_punta])
                cerrados.append(1 if d_medio < d_punta - MARGEN_DEDO else 0)
            else:
                x_ip = coord_x(landmarks, 3)
                y_ip = coord_y(landmarks, 3)
                x_base_menique = coord_x(landmarks, 17)
                y_base_menique = coord_y(landmarks, 17)

                d_ip = dist([x_ip, y_ip], [x_base_menique, y_base_menique])
                d_punta = dist([x_punta, y_punta], [x_base_menique, y_base_menique])

                cerrados.append(1 if d_punta > d_ip + MARGEN_DEDO else 0)
        if len(cerrados) > 5:
            cerrados = cerrados[:5]

        return cerrados
    except Exception as e:
        print(f"Error en la detección del dedo: {e}")
        return [0] * 5


def dibujar_landmarks(frame, landmarks, ancho, alto):
    """Dibuja los puntos de la mano manualmente (reemplaza a mpDraw)."""
    for lm in landmarks:
        cx, cy = int(lm.x * ancho), int(lm.y * alto)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        resultado = detector.detect(mp_image)

        if resultado.hand_landmarks:
            landmarks = resultado.hand_landmarks[0]  # primera mano detectada
            alto, ancho, _ = frame.shape
            dibujar_landmarks(frame, landmarks, ancho, alto)

            deteccion = detectarDedo(landmarks)

            cv2.putText(
                frame,
                f"dedos detectados={sum(deteccion)}",
                (20, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
            )
            print(f"Detección: {deteccion}")

            datos = "".join([f"{i}{estado}" for i, estado in enumerate(deteccion)])
            arduino.write(bytes(datos + "\n", "utf-8"))

        cv2.imshow("Detección de mano", frame)
        time.sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        print(f"Error en el bucle principal: {e}")

cap.release()
cv2.destroyAllWindows()
arduino.close()