"""
Скрипт оценки модели YOLOv11
"""

from ultralytics import YOLO
import wandb
import argparse
import os
from utils import load_config, setup_wandb


def evaluate_model(model_path, config_path='config.yaml', data_path=None):
    """Оценка обученной модели"""

    # Загружаем конфигурацию если нужно для W&B
    if os.path.exists(config_path):
        config = load_config(config_path)
        wandb_enabled = setup_wandb(config, "evaluate")
    else:
        print(f"Конфиг файл {config_path} не найден, W&B логирование отключено")
        wandb_enabled = False
        config = {}

    # Инициализация модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = YOLO(model_path)
    print(f"Загружена модель: {model_path}")

    # Определяем путь к данным
    if data_path is None:
        data_path = config.get('model', {}).get('data', 'dataset/data.yaml')

    print(f"Начинаем оценку модели на данных: {data_path}")

    # Валидация на test сете
    results = model.val(
        data=data_path,
        split='test',
        save_json=True,
        plots=True,
        verbose=True
    )

    # Извлекаем основные метрики
    metrics = {
        "test_mAP50": results.box.map50,
        "test_mAP50-95": results.box.map,
        "test_precision": results.box.mp,
        "test_recall": results.box.mr
    }

    # Логирование результатов валидации в wandb
    if wandb.run and wandb_enabled:
        wandb.log(metrics)
        print("Результаты тестирования залогированы в W&B")
    else:
        print("W&B сессия не активна, результаты оценки не залогированы.")

    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ:")
    print("="*50)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print("="*50)

    return results


def main():
    """Главная функция с аргументами командной строки"""
    parser = argparse.ArgumentParser(description="Оценка модели YOLOv11")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Путь к файлу модели (.pt)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Путь к файлу конфигурации (по умолчанию: config.yaml)"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Путь к файлу конфига датасета (.yaml). Если не указан, берется из конфига"
    )

    args = parser.parse_args()

    print("Оценка модели")
    print("=" * 60)

    try:
        # Оценка модели
        evaluate_model(args.model, args.config, args.data)

        if wandb.run:
            wandb.finish()
            print("W&B сессия завершена")

        print("\nОценка завершена успешно!")
    except Exception as e:
        print(f"Ошибка во время оценки: {e}")
        if wandb.run:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
