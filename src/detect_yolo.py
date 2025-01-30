from ultralytics import YOLO
import cv2
import os
import uuid
from datetime import datetime
import time
import base64
import requests

EVIDENCE_DIR = "evidence/"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

API_LOGIN_ENDPOINT = "http://127.0.0.1:8000/api/authentication/login/"
API_MONITORING_ENDPOINT = "http://127.0.0.1:8000/api/monitoring/"
USERNAME = "admin"
PASSWORD = "Senai@2023"


def authenticate():
    payload = {"username": USERNAME, "password": PASSWORD}
    try:
        response = requests.post(API_LOGIN_ENDPOINT, json=payload)
        if response.status_code == 200:
            access_token = response.json().get("token_access")
            print("Login bem-sucedido!")
            return access_token
        else:
            print(f"Erro ao fazer login: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Erro na conexão de login: {e}")
        return None


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def send_to_system_b(access_token, mac, timestamp, object_class, evidence_path):
    base64_image = encode_image_to_base64(evidence_path)

    payload = {
        "mac": mac,
        "date": timestamp,
        "object_classobject_class": object_class,
        "evidence": f"data:image/jpeg;base64,{base64_image}",
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_MONITORING_ENDPOINT, json=payload, headers=headers)
        if response.status_code == 201:
            print("Registro enviado com sucesso ao Sistema B!")
        else:
            print(f"Erro ao enviar registro: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Erro na conexão: {e}")


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    return ":".join(mac[i : i + 2] for i in range(0, 12, 2))


model = YOLO("models/yolov11/yolo11m.pt")


def detect_yolo(video_path=None):
    access_token = authenticate()
    if not access_token:
        print("Não foi possível autenticar no Sistema B. Saindo.")
        return

    mac_address = get_mac_address()
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    last_saved_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6]
            class_name = model.names[int(cls)]

            if conf < 0.9:
                continue

            if class_name not in ["mouse", "mouse-1"]:
                continue

            current_time = time.time()
            if current_time - last_saved_time < 1:
                continue

            last_saved_time = current_time

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            evidence_filename = f"{EVIDENCE_DIR}evidence_{uuid.uuid4().hex[:8]}.jpg"

            cropped_img = frame[int(y1) : int(y2), int(x1) : int(x2)]
            cv2.imwrite(evidence_filename, cropped_img)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            print(f"MAC: {mac_address}")
            print(f"DATE: {timestamp}")
            print(f"CLASS: {class_name}")
            print(f"EVIDENCE: {evidence_filename}")
            print("=" * 50)

            send_to_system_b(
                access_token, mac_address, timestamp, class_name, evidence_filename
            )

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_yolo()
