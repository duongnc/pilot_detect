import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DANFeatureExtractor(nn.Module):
    def __init__(self, num_heads=4, out_features=128):
        super(DANFeatureExtractor, self).__init__()
        # Load mô hình pre-trained
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Bỏ lớp Fully Connected cuối cùng của ResNet18
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        # Đưa về vector 128D
        self.fc = nn.Linear(512, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Chỉ chạy BatchNorm nếu có nhiều hơn 1 phần tử (Tránh lỗi toán học)
        if x.size(0) > 1 or not self.training:
            x = self.bn(x)
        return x