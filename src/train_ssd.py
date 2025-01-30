import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

DATASET_PATH = "dataset/SSD/"
CLASS_MAPPING = {"mouse-1": 0, "mouse-2": 1}

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 10
MAX_BOXES = 10
NUM_CLASSES = len(CLASS_MAPPING)


def parse_voc_annotation(images_dir, class_mapping):
    images = []
    labels = []

    for file in tqdm(os.listdir(images_dir), desc=f"Carregando {images_dir}"):
        if file.endswith(".xml"):
            xml_path = os.path.join(images_dir, file)
            img_path = xml_path.replace(".xml", ".jpg")

            if not os.path.exists(img_path):
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

            objects = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                class_id = class_mapping.get(class_name, -1)

                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                objects.append(
                    [
                        xmin / IMG_SIZE,
                        ymin / IMG_SIZE,
                        xmax / IMG_SIZE,
                        ymax / IMG_SIZE,
                        class_id,
                    ]
                )

            while len(objects) < MAX_BOXES:
                objects.append([0, 0, 0, 0, -1])

            images.append(img)
            labels.append(objects[:MAX_BOXES])

    return np.array(images), np.array(labels, dtype=np.float32)


print("Carregando os dados...")

X_train, Y_train = parse_voc_annotation(
    os.path.join(DATASET_PATH, "train"), CLASS_MAPPING
)
X_valid, Y_valid = parse_voc_annotation(
    os.path.join(DATASET_PATH, "valid"), CLASS_MAPPING
)

print(f"Treino: {len(X_train)} imagens, Validação: {len(X_valid)} imagens")


print("Criando modelo SSD...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False
)

bbox_head = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
bbox_head = tf.keras.layers.Dense(512, activation="relu")(bbox_head)
bbox_head = tf.keras.layers.Dense(MAX_BOXES * 4, activation="sigmoid")(bbox_head)
bbox_head = tf.keras.layers.Reshape((MAX_BOXES, 4), name="bbox_output")(bbox_head)

class_head = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
class_head = tf.keras.layers.Dense(512, activation="relu")(class_head)
class_head = tf.keras.layers.Dense(MAX_BOXES * NUM_CLASSES, activation="softmax")(
    class_head
)
class_head = tf.keras.layers.Reshape((MAX_BOXES, NUM_CLASSES), name="class_output")(
    class_head
)

model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_head, class_head])

model.compile(
    optimizer="adam",
    loss={
        "bbox_output": "mean_squared_error",
        "class_output": "categorical_crossentropy",
    },
    metrics={
        "bbox_output": "mse",
        "class_output": "accuracy",
    },
)

print("Iniciando treinamento...")

model.fit(
    X_train,
    {
        "bbox_output": Y_train[:, :, :4],
        "class_output": tf.keras.utils.to_categorical(
            Y_train[:, :, 4], num_classes=NUM_CLASSES
        ),
    },
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(
        X_valid,
        {
            "bbox_output": Y_valid[:, :, :4],
            "class_output": tf.keras.utils.to_categorical(
                Y_valid[:, :, 4], num_classes=NUM_CLASSES
            ),
        },
    ),
)

model.save("models/ssd/ssd_model.h5")
print("Treinamento concluído. Modelo salvo em 'models/ssd/ssd_model.h5'.")
