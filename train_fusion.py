import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from core.preprocessing.au_detector import AUPreprocessor
from core.feature_extraction.dan_network import DANFeatureExtractor
from core.feature_extraction.vgg19_network import AUFeatureExtractor
from core.fusion.multi_modal import MultiModalFusion

class CroppedPilotDataset(Dataset):
    def __init__(self, data_dir, au_prep):
        self.image_paths = []
        self.labels = []
        self.au_prep = au_prep
        
        self.transform_face = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        valid_folders = [f for f in sorted(os.listdir(data_dir)) 
                         if os.path.isdir(os.path.join(data_dir, f)) and not f.startswith('.')]
        self.num_classes = len(valid_folders)
        
        for label_idx, folder_name in enumerate(valid_folders):
            folder_path = os.path.join(data_dir, folder_name)
            for img_name in os.listdir(folder_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')) and not img_name.startswith('.'):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(label_idx)
                    
        print(f"[INFO] Load thành công {len(self.labels)} ảnh / {self.num_classes} nhãn.")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        frame_bgr = cv2.imread(self.image_paths[idx])
        face_tensor = self.transform_face(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        au_tensor = self.au_prep.process(frame_bgr).squeeze(0)
        return face_tensor, au_tensor, self.labels[idx]

def train_model():
    # Tự động nhận diện phần cứng: NVIDIA (cuda) -> Apple Silicon (mps) -> Intel/AMD (cpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"[INFO] Đang chạy AI trên thiết bị: {device}")

    dataset_path = 'data/train/' 
    au_prep = AUPreprocessor(target_size=(224, 224)) 
    dataset = CroppedPilotDataset(dataset_path, au_prep)
    
    # drop_last=True: Sửa lỗi toán học khi lô cuối bị lẻ 1 tấm ảnh
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True) 

    dan = DANFeatureExtractor(num_heads=4, out_features=128).to(device)
    vgg = AUFeatureExtractor(out_features=128).to(device)
    fusion = MultiModalFusion(num_classes=dataset.num_classes).to(device)

    try:
        sd = torch.load('models/weights/rafdb_dan.pth', map_location=device)
        dan.load_state_dict(sd.get('model_state_dict', sd.get('model', sd)), strict=False)
    except: pass

    # Transfer Learning: Đóng băng mạng DAN
    for param in dan.parameters(): param.requires_grad = False
    dan.eval() 

    vgg.train()
    fusion.train()

    optimizer = optim.Adam(list(vgg.parameters()) + list(fusion.parameters()), lr=0.001)
    criterion = nn.NLLLoss()
    epochs = 100 

    for epoch in range(epochs):
        total_loss = 0.0
        for f_face, f_au, labels in dataloader:
            f_face, f_au, labels = f_face.to(device), f_au.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feat_face = dan(f_face) 
                
            feat_au = vgg(f_au)
            probs, _ = fusion(feat_face, feat_au)
            loss = criterion(torch.log(probs + 1e-7), labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs('models/weights', exist_ok=True)
    torch.save({'vgg_state_dict': vgg.state_dict(), 'fusion_state_dict': fusion.state_dict()}, 
               'models/weights/trained_fusion_module.pth')
    print(f"\n[INFO] HUẤN LUYỆN HOÀN TẤT!")

if __name__ == '__main__':
    train_model()