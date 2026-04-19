import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class AUFeatureExtractor(nn.Module):
    def __init__(self, out_features=128):
        super(AUFeatureExtractor, self).__init__()
        self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)
        
        # Tái tạo lại chính xác cấu trúc cũ để khớp với file weights của bạn
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # Đưa lớp BatchNorm gộp chung vào cụm layer 6 như cũ
            nn.Sequential(
                nn.Linear(4096, out_features),
                nn.BatchNorm1d(out_features)
            )
        )

    def forward(self, x):
        # Vì BatchNorm đã nằm sẵn bên trong vgg19, 
        # chúng ta chỉ cần đẩy dữ liệu đi qua một lần là xong.
        return self.vgg19(x)