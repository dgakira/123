import os

IMG_WIDTH = 640
IMG_HEIGHT = 480

LABELS_PATH = "D:\dataset\daset-196-1-out"

for label_file in os.listdir(LABELS_PATH):
    if label_file.endswith(".txt"):
        label_file_path = os.path.join(LABELS_PATH, label_file)
        with open(label_file_path, 'r') as file:
            content = file.readlines()
        new_content = []
        for line in content:
            columns = line.strip().split(",")
            if len(columns) == 5:
                label, x, y, w, h = map(float, columns[0:])
                x, y, w, h = x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT
                new_line = f"{int(label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                new_content.append(new_line)
        with open(label_file_path, 'w') as file:
            file.writelines(new_content)
print(".txt文件处理完毕！")
