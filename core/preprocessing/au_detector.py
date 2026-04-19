import cv2
from torchvision import transforms

class AUPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        au_tensor = self.transform(frame_rgb).unsqueeze(0)
        return au_tensor