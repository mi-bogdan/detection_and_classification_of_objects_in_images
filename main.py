import os
import time

import numpy as np
import PIL.Image as Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Загрузка предобученной модели ResNet-152
model = models.resnet152(pretrained=True)
model.eval()

# Индексы для классов кошек и собак в ImageNet
CAT_INDICES = list(range(281, 286))  # Кошки
DOG_INDICES = list(range(151, 269))  # Собаки

# Функция для предобработки изображения


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = transform(image)  # Прямая обработка PIL-изображения
    image = image.unsqueeze(0)  # Добавление размерности батча
    return image


def process_folder(folder_path):
    start_time = time.time()  # Замер времени
    probabilities_list = []
    images_processed = 0

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image = Image.open(image_path)
        input_image = preprocess_image(image)  # Предобработка изображения
        images_processed += 1

        with torch.no_grad():
            output = model(input_image)  # Получаем вывод сети для изображения

        probabilities = torch.nn.functional.softmax(
            output[0], dim=0
        )  # Применяем softmax

        # Суммируем вероятности для индексов кошек и собак
        cat_prob = probabilities[CAT_INDICES].sum().item()
        dog_prob = probabilities[DOG_INDICES].sum().item()

        probabilities_list.append(
            (cat_prob, dog_prob)
        )  # Добавление результатов в список

    end_time = time.time()
    processing_time = end_time - start_time  # Вычисление общего времени выполнения
    if images_processed > 0:
        fps = images_processed / processing_time  # Вычисление FPS (кадров в секунду)
    else:
        fps = 0

    # Вычисление и вывод средних вероятностей обнаружения кошек и собак
    mean_probabilities = np.mean(probabilities_list, axis=0)
    print(f"Обработка папки {folder_path} завершена.")
    print(
        f"Обработано изображений: {images_processed}",
        f"Заняло времени: {processing_time: .3f} секунды",
        f"FPS: {fps: .2f}",
    )
    print(
        f"Средние вероятности:Кошки={mean_probabilities[0]: .3f}",
        f"Собаки={mean_probabilities[1]: .3f}",
    )


# Обработка папок с изображениями кошек и собак
print("Обработка изображений кошек...")
process_folder("train/cats")
print("Обработка изображений собак...")
process_folder("train/dogs")
