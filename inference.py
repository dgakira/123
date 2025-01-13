from ultralytics import YOLO
import os
def run_inference(model_checkpoint, input_dir, output_dir, confidence_threshold=0.25, image_size=640):
    model = YOLO(model_checkpoint)
    results = model.predict(
        source=input_dir,  # 输入目录
        conf=confidence_threshold,  # 置信度阈值
        imgsz=image_size,  # 输入图像尺寸
        save=True,  # 保存结果
        save_txt=True,  # 保存标签文件
        save_conf=True,  # 保存置信度值
        project=output_dir,  # 输出路径
        name="predictions"  # 子目录名称
    )
    result_path = os.path.join(output_dir, 'predictions')
    print(f"推理完成！结果已保存到: {result_path}")
if __name__ == '__main__':
    MODEL_PATH = "D:/best.pt"
    TEST_IMAGE_DIR = "D:\dataset\daset-195-2-color"
    OUTPUT_DIR = "D:/dataset"
    run_inference(MODEL_PATH, TEST_IMAGE_DIR, OUTPUT_DIR)
