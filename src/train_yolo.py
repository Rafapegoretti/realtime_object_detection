from ultralytics import YOLO
import torch


def train_yolo():
    # Carregar o modelo YOLOv11 pré-treinado
    model = YOLO("models/yolov11/yolo11m.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Treinar o modelo
    model.train(
        data="dataset/YOLO/data.yaml",  # Caminho para o arquivo de configuração do dataset
        epochs=5,  # Número de épocas de treinamento
        imgsz=640,  # Tamanho das imagens de entrada
        batch=16,  # Tamanho do batch
        device=device,  # Índice da GPU (use 'cpu' se não houver GPU)
    )


if __name__ == "__main__":
    train_yolo()
