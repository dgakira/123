from ultralytics import YOLO


def train_model(model_checkpoint, dataset_yaml, export_format='onnx', **training_args):
    try:
        model = YOLO(model_checkpoint)
        model.train(data=dataset_yaml, **training_args)
        if export_format:
            export_args = {'format': export_format}
            model.export(**export_args)
    except Exception as e:
        print(f"发生错误：{e}")
if __name__ == '__main__':
    MODEL_PATH = 'yolov8m.pt'
    DATASET_PATH = 'dataset.yaml'
    TRAINING_PARAMETERS = {
        'epochs': 100,  # 训练轮数
        'batch': 16,  # 批大小
        'imgsz': 640,  # 输入图片大小
        'lr0': 0.005,  # 初始学习率
        'optimizer': 'AdamW',  # 使用 AdamW 优化器
        'weight_decay': 0.0001,  # 权重衰减
        'momentum': 0.9,  # 动量
        'device': 0,  # GPU 设备编号
        'pretrained': True,  # 使用预训练权重
        'mosaic': 1.0,  # 启用 Mosaic 数据增强
        'mixup': 0.2,  # 启用 Mixup 数据增强
    }
    train_model(MODEL_PATH, DATASET_PATH, export_format='onnx', **TRAINING_PARAMETERS)
