from ultralytics import YOLO

def evaluate_model(weight_path, data_config, img_size=640, batch_size=16, split='test'):
    try:
        print(f"加载模型权重文件：{weight_path}...")
        model = YOLO(weight_path)
        print(f"在 {split} 数据集上验证模型性能...")
        results = model.val(data=data_config, split=split, imgsz=img_size, batch=batch_size)
        print("\n验证结果：")
        print(f"  mAP@50: {results.metrics.mAP50:.4f}")
        print(f"  mAP@50-95: {results.metrics.mAP50_95:.4f}")
        print(f"  Precision: {results.metrics.precision:.4f}")
        print(f"  Recall: {results.metrics.recall:.4f}")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == '__main__':
    WEIGHT_PATH = 'D:/dataset/best.pt'
    DATASET_CONFIG = 'dataset.yaml'
    IMG_SIZE = 640
    BATCH_SIZE = 16
    SPLIT = 'test'
    evaluate_model(WEIGHT_PATH, DATASET_CONFIG, img_size=IMG_SIZE, batch_size=BATCH_SIZE, split=SPLIT)