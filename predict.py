"""
Скрипт предикта через YOLOv11
"""

from ultralytics import YOLO
import wandb
import argparse
import os
import glob
import cv2
from pathlib import Path
from utils import load_config, setup_wandb


def predict_on_images(model_path, source_path, config_path='config.yaml', save_results=True):
    """Предсказание на изображениях или директории с изображениями"""

    # Загружаем конфигурацию если нужно для W&B
    if os.path.exists(config_path):
        config = load_config(config_path)
        wandb_enabled = setup_wandb(config, "predict")
    else:
        print(f"Конфиг файл {config_path} не найден, W&B логирование отключено")
        wandb_enabled = False

    # Проверяем существование модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    # Инициализация модели
    model = YOLO(model_path)
    print(f"Загружена модель: {model_path}")

    # Проверяем источник данных
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Источник данных не найден: {source_path}")

    # Определяем количество изображений
    if os.path.isfile(source_path):
        image_count = 1
        print(f"Обработка одного изображения: {source_path}")
    else:
        # Подсчитываем изображения в директории
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(source_path, ext)))
            image_files.extend(glob.glob(os.path.join(source_path, ext.upper())))
        image_count = len(image_files)
        print(f"Найдено {image_count} изображений в директории: {source_path}")

    if image_count == 0:
        print("Изображения не найдены!")
        return None

    # Настройки предикта
    prediction_params = {
        'source': source_path,
        'save': save_results,
        'save_txt': save_results,
        'save_conf': save_results,
        'project': 'runs/predict',
        'name': 'dish_predictions',
        'exist_ok': True,
        'verbose': True
    }

    print("Параметры предикта:")
    for key, value in prediction_params.items():
        print(f"  {key}: {value}")

    # Запуск предикта
    results = model.predict(**prediction_params)

    # Статистика обнаружений
    total_detections = 0
    images_with_detections = 0

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            detections_count = len(result.boxes)
            total_detections += detections_count
            images_with_detections += 1

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ПРЕДИКТА:")
    print("="*50)
    print(f"Обработано изображений: {len(results)}")
    print(f"Изображений с объектами: {images_with_detections}")
    print(f"Общее количество объектов: {total_detections}")
    if len(results) > 0:
        print(f"Среднее объектов на изображение: {total_detections/len(results):.2f}")
    print("="*50)

    if save_results:
        output_dir = Path('runs/predict/dish_predictions')
        print(f"Результаты сохранены в: {output_dir}")

    # Логирование в W&B
    if wandb.run and wandb_enabled and results:
        # Логируем статистику
        wandb.log({
            "total_images": len(results),
            "images_with_detections": images_with_detections,
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections/len(results) if len(results) > 0 else 0
        })

        # Логируем примеры предиктов (первые 5)
        sample_images = []
        for i, result in enumerate(results[:5]):
            if result.path and result.boxes is not None:
                # Создаем изображение с аннотациями
                img_with_boxes = result.plot()
                sample_images.append(wandb.Image(
                    img_with_boxes,
                    caption=f"Prediction {i+1}: {len(result.boxes)} detections"
                ))

        if sample_images:
            wandb.log({"sample_predictions": sample_images})
            print("Примеры предиктов залогированы в W&B")

        print("Результаты предикта залогированы в W&B")

    return results


def predict_on_video(model_path, video_path, save_results=True, output_path=None):
    """Предсказание на видео"""

    # Проверяем существование модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    # Проверяем существование видео
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео файл не найден: {video_path}")

    # Инициализация модели
    model = YOLO(model_path)
    print(f"Загружена модель: {model_path}")
    print(f"Обработка видео: {video_path}")

    # Получаем информацию о видео
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Информация о видео:")
    print(f"  Кадров: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Разрешение: {width}x{height}")

    # Настройка выходного пути
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"runs/predict{video_name}_predicted.mp4"

    # Создаем директорию для результатов
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Настройки предикта
    prediction_params = {
        'source': video_path,
        'save': save_results,
        'project': 'runs/predict',
        'name': 'predictions',
        'exist_ok': True,
        'verbose': True,
        'stream': True,
        'show': False
    }

    print("Параметры предикта:")
    for key, value in prediction_params.items():
        print(f"  {key}: {value}")

    # Запуск предикта
    print("\nОбработка видео...")
    results = model.predict(**prediction_params)

    # Статистика обнаружений
    total_detections = 0
    frames_with_detections = 0
    processed_frames = 0

    # Инициализация видео writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for result in results:
            processed_frames += 1

            if result.boxes is not None and len(result.boxes) > 0:
                detections_count = len(result.boxes)
                total_detections += detections_count
                frames_with_detections += 1

            # Получаем изображение с аннотациями
            annotated_frame = result.plot()

            # Записываем кадр в выходное видео
            if save_results:
                out.write(annotated_frame)

            # Прогресс каждые 100 кадров
            if processed_frames % 100 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"Обработано кадров: {processed_frames}/{total_frames} ({progress:.1f}%)")

    finally:
        out.release()

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ПРЕДИКТА НА ВИДЕО:")
    print("="*50)
    print(f"Обработано кадров: {processed_frames}")
    print(f"Кадров с объектами: {frames_with_detections}")
    print(f"Общее количество объектов: {total_detections}")
    if processed_frames > 0:
        print(f"Среднее объектов на кадр: {total_detections/processed_frames:.2f}")
        print(f"Процент кадров с объектами: {(frames_with_detections/processed_frames)*100:.1f}%")
    print("="*50)

    if save_results:
        print(f"Результирующее видео сохранено в: {output_path}")
        print(f"Размер файла: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

    return output_path if save_results else None


def main():
    """Главная функция с аргументами командной строки"""
    parser = argparse.ArgumentParser(description="Предсказание с помощью модели YOLOv11")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Путь к файлу модели (.pt)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Путь к изображению, директории с изображениями или видео файлу"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["image", "video", "auto"],
        default="auto",
        help="Тип источника данных: image, video или auto (автоопределение)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Путь для сохранения результирующего видео (только для видео)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Путь к файлу конфигурации (по умолчанию: config.yaml)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять результаты предиктов"
    )

    args = parser.parse_args()
    print("=" * 60)

    try:
        # Определяем тип источника данных
        source_type = args.type
        if source_type == "auto":
            # Автоопределение по расширению файла
            source_path = Path(args.source)
            if source_path.is_file():
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
                if source_path.suffix.lower() in video_extensions:
                    source_type = "video"
                else:
                    source_type = "image"
            else:
                source_type = "image"  # Директория с изображениями

        print(f"Тип источника: {source_type}")

        if source_type == "video":
            # Предикт на видео
            result_path = predict_on_video(
                model_path=args.model,
                video_path=args.source,
                save_results=not args.no_save,
                output_path=args.output
            )

            if result_path:
                print(f"\nВидео с предиктами сохранено: {result_path}")
        else:
            # Предикт на изображениях
            predict_on_images(
                model_path=args.model,
                source_path=args.source,
                config_path=args.config,
                save_results=not args.no_save
            )

        if wandb.run:
            wandb.finish()
            print("W&B сессия завершена")

        print("\nПредсказание завершено успешно!")

    except Exception as e:
        print(f"Ошибка во время предикта: {e}")
        if wandb.run:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
