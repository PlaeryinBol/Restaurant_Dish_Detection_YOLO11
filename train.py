"""
Скрипт обучения YOLOv11
"""

from ultralytics import YOLO
import wandb
from utils import load_config, setup_wandb, get_device, print_train_params


def train(config_path='config.yaml'):
    """Обучение модели с использованием конфигурации из YAML файла"""
    config = load_config(config_path)
    wandb_enabled = setup_wandb(config, "train")
    device = get_device()

    # Инициализация модели YOLO11
    model = YOLO(config['model']['weights'])

    # Извлекаем параметры для W&B
    wandb_config = config.get('wandb', {})

    # Объединяем все параметры тренировки из конфига
    train_params = {
        # Основные параметры
        'data': config['model']['data'],
        'device': device,

        # Логирование в W&B (Ultralytics WandbLogger)
        'project': wandb_config.get('project_name', 'yolo11_dishes'),
        'name': wandb_config.get('experiment_name', 'yolo11_dishes_experiment'),

        # Параметры тренировки
        **config['training'],
        **config['augmentation'],
        **config['optimization'],
        **config['early_stopping']
    }

    print_train_params(train_params)

    # Запуск обучения
    results = model.train(**train_params)

    # Логирование лучших результатов в wandb
    if wandb.run and wandb_enabled:
        # Получаем метрики из результатов
        metrics_dict = {}
        if hasattr(results, 'results_dict'):
            metrics_dict.update({
                "best_mAP50": results.results_dict.get('metrics/mAP50(B)', 0),
                "best_mAP50-95": results.results_dict.get('metrics/mAP50-95(B)', 0),
                "best_precision": results.results_dict.get('metrics/precision(B)', 0),
                "best_recall": results.results_dict.get('metrics/recall(B)', 0),
                "final_loss": (results.results_dict.get('train/box_loss', 0) +
                               results.results_dict.get('train/cls_loss', 0) +
                               results.results_dict.get('train/dfl_loss', 0))
            })

        # Логируем метрики если они есть
        if metrics_dict:
            wandb.log(metrics_dict)
            print("Финальные метрики залогированы в W&B")

        # Сохранение лучшей модели как артефакт
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            artifact = wandb.Artifact(
                name="yolo11_dishes_best_model",
                type="model",
                description="Лучшая модель"
            )
            artifact.add_file(str(best_model_path))
            wandb.log_artifact(artifact)
            print("Модель сохранена как артефакт в W&B")

    print("Обучение завершено!")
    print(f"Лучшая модель сохранена в: {results.save_dir}")
    return results


if __name__ == "__main__":
    print("Обучение YOLOv11")
    print("=" * 60)

    try:
        # Обучение модели
        print("Запуск обучения...")
        train_results = train()

        print(f"Результаты сохранены в: {train_results.save_dir}")

        if wandb.run:
            wandb.finish()
            print("W&B сессия завершена")

        print("\nОбучение завершено успешно!")

    except Exception as e:
        print(f"Ошибка во время обучения: {e}")
        if wandb.run:
            wandb.finish(exit_code=1)
        raise
