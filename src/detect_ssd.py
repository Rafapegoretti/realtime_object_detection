import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/ssd/ssd_model.h5")


def detect_ssd(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_img = cv2.resize(frame, (300, 300)) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        bbox_pred, class_pred = model.predict(input_img)

        for i in range(len(bbox_pred[0])):
            x_min, y_min, x_max, y_max = bbox_pred[0][i] * frame.shape[1]
            class_id = np.argmax(class_pred[0][i])
            label = f"Class {class_id}: {class_pred[0][i].max():.2f}"

            cv2.rectangle(
                frame,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                label,
                (int(x_min), int(y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        cv2.imshow("SSD Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_ssd()
