"""
Утилиты для проекта YOLOv11
"""

import yaml
import wandb
import torch
from ultralytics import settings

# Включаем поддержку WandB в Ultralytics
settings.update({"wandb": True})


def load_config(config_path='config.yaml'):
    """Загрузка конфига из YAML файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"Конфиг загружен из: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Конфиг не найден: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Ошибка при парсинге YAML: {e}")
        raise


def setup_wandb(config, script_name="experiment"):
    """Настройка WandB для логирования"""
    wandb_config = config.get('wandb', {})

    if wandb_config.get('enable', True):
        try:
            if 'api_key' in wandb_config:
                wandb.login(key=wandb_config['api_key'])

            wandb.init(
                project=wandb_config.get('project_name', 'yolo11_dishes'),
                entity=wandb_config.get('entity', 'plaeryinbol-everypixel'),
                name=f"{wandb_config.get('experiment_name', 'yolo11_dishes_experiment')}_{script_name}",
                config=config,
                tags=wandb_config.get('tags', ["yolo11", "object_detection"])
            )
            project_name = wandb_config.get('project_name', 'yolo11_dishes')
            entity_name = wandb_config.get('entity', 'plaeryinbol-everypixel')
            print(f"W&B логирование инициализировано. Проект: {entity_name}/{project_name}")
            return True
        except Exception as e:
            print(f"Ошибка инициализации W&B: {e}")
            print("Продолжение без логирования в W&B.")
            return False
    else:
        print("W&B отключен в конфигурации")
        return False


def get_device():
    """Получение GPU/CPU"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    return device


def print_train_params(train_params):
    """Красивый вывод параметров тренировки"""
    print("Параметры тренировки:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
