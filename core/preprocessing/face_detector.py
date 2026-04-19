import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms

class FacePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process(self, frame_bgr):
        # Chuyển BGR sang RGB cho MTCNN
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(frame_rgb)
        
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0])
            # Tránh tọa độ âm
            x1, y1 = max(0, x1), max(0, y1) 
            face_crop = frame_rgb[y1:y2, x1:x2]
            
            if face_crop.size == 0: 
                return None
                
            # Ép về kích thước chuẩn và thêm chiều Batch
            face_tensor = self.transform(face_crop).unsqueeze(0)
            return face_tensor
            
        return None