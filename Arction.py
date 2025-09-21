import sys
import torch
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torchvision.transforms as transforms

# -------YOLO裁剪相关--------
def crop_with_yolo(img_path, yolo_model_path='yolov5x.pt', conf_thres=0.25):
    """
    使用YOLOv5进行目标检测并裁剪最大目标，返回PIL.Image对象
    要求已安装ultralytics/yolov5: pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics 包 (pip install ultralytics)")

    # 加载YOLO模型
    model = YOLO(yolo_model_path)
    results = model(img_path, conf=conf_thres)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
    im = Image.open(img_path).convert('RGB')
    if len(boxes) > 0:
        # 找最大面积的目标
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        max_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, boxes[max_idx])
        cropped = im.crop((x1, y1, x2, y2))
        return cropped
    else:
        # 没检测到目标，返回原图
        return im

# -------你的 CBAM 和模型结构--------
class CBAM(torch.nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // reduction, channels)
        )
        self.sigmoid_channel = torch.nn.Sigmoid()
        self.conv_spatial = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid_spatial = torch.nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att.expand_as(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att))
        x = x * spatial_att.expand_as(x)
        return x

class SEEfficientNetB3_CBAM(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b3')
        self.cbam = CBAM(1536)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1536, num_classes)
        params = list(self.backbone.parameters())
        for param in params[:-10]:
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------类别列表（自动读取类别名文件，或手动填写）--------
def load_classes(classes_txt='classes.txt', data_dir=r"Arction Data Set"):
    """
    加载类别列表，优先从 classes.txt 文件读取；
    如果文件不存在且 data_dir 给定，则从ImageFolder自动获取类别并保存到 classes.txt；
    否则返回一个默认列表。
    """
    import os
    from torchvision import datasets

    # 优先尝试读取 classes.txt
    if os.path.exists(classes_txt):
        with open(classes_txt, encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    # 如果指定了数据目录，自动获得类别并保存
    if data_dir is not None and os.path.exists(data_dir):
        dataset = datasets.ImageFolder(root=data_dir)
        classes = dataset.classes
        with open(classes_txt, 'w', encoding='utf-8') as f:
            for cls in classes:
                f.write(cls + '\n')
        return classes

    # 否则返回默认示例类别
    return ['class_0', 'class_1', 'class_2', 'class_3']

# -------模型加载--------
def load_model(model_path, num_classes, device):
    model = SEEfficientNetB3_CBAM(num_classes=31)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# -------TTA+Soft Voting预测，返回Top3类别和概率--------
def tta_predict_top3(model, pil_img, class_names, device, n_votes=6):
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((400, 1200)),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((400, 1200)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((400, 1200)),
            transforms.RandomRotation([0,10]),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.Resize((400, 1200)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((400, 1200)),
            transforms.ColorJitter(contrast=0.2),
            transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize((400, 1200)),
            transforms.ColorJitter(saturation=0.2),
            transforms.ToTensor(),
        ]),
    ]

    probs_sum = np.zeros(len(class_names), dtype=np.float32)
    with torch.no_grad():
        for t in tta_transforms[:n_votes]:
            img_tensor = t(pil_img).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
            probs_sum += probs
    probs_mean = probs_sum / n_votes
    top3_indices = probs_mean.argsort()[-3:][::-1]
    top3 = [(class_names[i], 100*probs_mean[i]) for i in top3_indices]
    return top3

# -------PyQt5界面--------
class ClassifierApp(QWidget):
    def __init__(self, model, classes, device, yolo_path='yolov5x.pt'):
        super().__init__()
        self.model = model
        self.classes = classes
        self.device = device
        self.img_path = None
        self.yolo_path = yolo_path
        self.pil_img = None  # 用于缓存裁剪后的PIL图像
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Arction:AirCraft Recognition')
        self.setAcceptDrops(True)
        layout = QVBoxLayout()

        self.img_label = QLabel('拖拽图片或点击选择图片（侧面图片识别率高）')
        self.img_label.setFixedSize(300, 300)
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label)

        btns = QHBoxLayout()
        self.btn_select = QPushButton('选择图片')
        self.btn_select.clicked.connect(self.openFileDialog)
        btns.addWidget(self.btn_select)

        self.btn_recognize = QPushButton('开始识别')
        self.btn_recognize.clicked.connect(self.recognize)
        btns.addWidget(self.btn_recognize)

        layout.addLayout(btns)

        self.result_label = QLabel('识别结果将在这里显示')
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def openFileDialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files (*.jpg *.png)')
        if fname:
            self.showImage(fname)

    def showImage(self, img_path):
        self.img_path = img_path
        self.pil_img = None
        pixmap = QPixmap(img_path).scaled(300, 300, Qt.KeepAspectRatio)
        self.img_label.setPixmap(pixmap)
        self.result_label.setText('图片已加载，请点击“开始识别”')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            img_path = urls[0].toLocalFile()
            self.showImage(img_path)

    def recognize(self):
        if not self.img_path:
            self.result_label.setText('请先加载图片！')
            return
        self.result_label.setText('正在裁剪...')
        QApplication.processEvents()
        try:
            self.pil_img = crop_with_yolo(self.img_path, yolo_model_path=self.yolo_path)
        except Exception as e:
            self.result_label.setText(f'裁剪失败: {str(e)}')
            return
        self.result_label.setText('正在识别，请稍候...')
        QApplication.processEvents()
        top3 = tta_predict_top3(self.model, self.pil_img, self.classes, self.device)
        res_text = "Top 3 预测结果：\n"
        for i, (cls, prob) in enumerate(top3):
            res_text += f"{i+1}. {cls}: {prob:.2f}%\n"
        self.result_label.setText(res_text)

if __name__ == '__main__':
    model_path = 'seefficientnetb3_60%31.pth' # 改成你的模型文件名
    classes = load_classes('classes.txt')    # 推荐用classes.txt保存你的类别名
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=len(classes), device=device)


    yolo_path = 'yolov5x.pt'
    app = QApplication(sys.argv)
    ex = ClassifierApp(model, classes, device, yolo_path=yolo_path)
    ex.show()
    sys.exit(app.exec_())