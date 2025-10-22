# Restaurant_Dish_Detection_YOLO11

A project for object detection in restaurant environments using the YOLO11 model. The system can recognize 16 different object classes: dishes, food, staff, and restaurant visitors.

## Project Description

This project implements a system for automatic detection and classification of objects in restaurant environments. It uses the YOLO11 architecture for real-time object detection.

## Features

- Training on a custom dataset with temporal splitting
- Integration with Weights & Biases for experiment monitoring
- Configurable parameters via YAML files
- Advanced data augmentation for video scenes
- Prediction on images and videos
- Model evaluation with detailed metrics
- GPU and CPU support

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/PlaeryinBol/Restaurant_Dish_Detection_YOLO11.git
cd Restaurant_Dish_Detection_YOLO11
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup Weights & Biases (optional)**
```bash
wandb login
```

## Dataset Structure

The dataset is organized using **temporal splitting** to prevent data leakage (stored [here](https://drive.google.com/drive/folders/1LqMm9-j2ABd46gG7BUPc5ia3wQNvMUYX?usp=sharing) along with the best-performing model *best.pt*).
Annotation was done via [CVAT](https://cvat.ai/).

```
dataset/
├── images/
│   ├── train/    # 217 images (62.5%)
│   ├── val/      # 90 images (26.0%)
│   └── test/     # 40 images (11.5%)
├── labels/       # YOLO format annotations
└── data.yaml     # Dataset configuration
```

**Splitting principle**: Frames from the same time period do not appear in different splits, ensuring realistic evaluation of the model's generalization ability.

## Model Training

### Parameter Configuration
Edit `config.yaml` to change training parameters:

### Train
```bash
python train.py
```

## Model Evaluation

```bash
python evaluate.py --model model.pt
```

## Prediction

### On a folder with images
```bash
python predict.py --model model.pt --source path/to/images/
```

### On video
```bash
python predict.py --model model.pt --source video.mov -o video_output.mp4
```

## Configuration

### Data Augmentation
A complex set of augmentations is applied, taking into account the specifics of video data:
- Geometric transformations (rotation, shift, scale)
- Color transformations (HSV)
- Mosaic and copy-paste techniques
- Label smoothing for "soft" boundaries during motion

## Results

The model is trained on restaurant scenes considering:
- Object movement in the frame
- Partial occlusions
- Various lighting conditions
- Diverse camera angles

### Key Metrics
Charts and metrics are available [here](https://wandb.ai/plaeryinbol-everypixel/yolo11_dishes/reports/Restaurant_Dish_Detection_YOLO11--VmlldzoxMzQ1OTk3OQ).
- **mAP@0.5**: 0.887
- **mAP@0.5:0.95**: 0.81
- **Precision**: 0.938
- **Recall**: 0.894

                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all         40        600      0.938      0.895      0.887       0.81
           table knife         35         54      0.984      0.296      0.379      0.342
             empty cup         40         80          1      0.915      0.994       0.92
          glass teapot         40         40       0.99          1      0.995      0.953
               visitor         21         25       0.57        0.6      0.504      0.248
         bowl of salad          5         10      0.973          1      0.995      0.995
      lavash flatbread         21         21      0.986          1      0.995      0.995
            sauce boat         21         42      0.994          1      0.995       0.99
          roasted ribs         21         21       0.98          1      0.995      0.985
          bowl of soup          5         10      0.974          1      0.995      0.795
            tablespoon         21         47      0.951      0.872      0.915      0.723
          paper napkin         40         65      0.717      0.742      0.563      0.467
            table fork         21         42      0.995          1      0.995      0.969
            shot glass         40         40      0.994          1      0.995      0.893
            smartphone         20         20      0.964          1      0.995      0.911
           empty plate         35         83          1      0.995      0.995      0.964
                waiter          8         15       0.92      0.896      0.888       0.80

Speed: 2.8ms preprocess, 22.7ms inference, 0.0ms loss, 4.4ms postprocess per image

### Example Result
![Model working example on video](output.gif)
