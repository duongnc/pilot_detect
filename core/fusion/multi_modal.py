import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    def __init__(self, num_classes=7):
        super(MultiModalFusion, self).__init__()
        # Trọng số thích nghi (Mô hình sẽ tự học xem nên tin vào Mặt hay Cơ mặt hơn)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        # Mạng nơ-ron phân loại dựa trên 2 vector ghép lại (128 + 128 = 256)
        self.classifier_concat = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f_face, f_au):
        f_face_weighted = f_face * self.alpha
        f_au_weighted = f_au * self.beta
        
        f_fused = torch.cat((f_face_weighted, f_au_weighted), dim=1)
        logits = self.classifier_concat(f_fused)
        probs = self.softmax(logits)
        
        return probs, torch.argmax(probs, dim=1)