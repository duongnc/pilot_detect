import cv2
import torch
import time
import numpy as np
from collections import deque

from core.camera_manager import CameraManager
from core.preprocessing.face_detector import FacePreprocessor
from core.preprocessing.au_detector import AUPreprocessor
from core.feature_extraction.dan_network import DANFeatureExtractor
from core.feature_extraction.vgg19_network import AUFeatureExtractor
from core.fusion.multi_modal import MultiModalFusion

# Nhãn khớp tuyệt đối với 7 thư mục
EMOTION_LABELS = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
NUM_CLASSES = len(EMOTION_LABELS)

def main():
    print("==================================================")
    print("[INFO] HỆ THỐNG GCS PILOT MONITORING SẴN SÀNG")
    print("==================================================")
    
    # Tự động nhận diện phần cứng: NVIDIA (cuda) -> Apple Silicon (mps) -> Intel/AMD (cpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"[INFO] Đang chạy AI trên thiết bị: {device}")

    face_prep = FacePreprocessor(target_size=(224, 224)) 
    au_prep = AUPreprocessor(target_size=(224, 224))
    
    dan = DANFeatureExtractor(num_heads=4, out_features=128).to(device)
    vgg = AUFeatureExtractor(out_features=128).to(device)
    fusion = MultiModalFusion(num_classes=NUM_CLASSES).to(device)

    # Nạp weights RAF-DB
    try:
        sd = torch.load('models/weights/rafdb_dan.pth', map_location=device)
        dan.load_state_dict(sd.get('model_state_dict', sd.get('model', sd)), strict=False)
    except: pass

    # Nạp weights tự Train (7 classes)
    try:
        cp = torch.load('models/weights/trained_fusion_module.pth', map_location=device)
        vgg.load_state_dict(cp['vgg_state_dict'])
        fusion.load_state_dict(cp['fusion_state_dict'])
        print("[INFO] Đã nạp Model 7 cảm xúc thành công.")
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi nạp trọng số: {e}")

    dan.eval(); vgg.eval(); fusion.eval()

    cam_manager = CameraManager(camera_source=0, width=640, height=480)
    cam_manager.start()
    
    # Bộ lọc trung bình động (10 khung hình)
    emotion_history = deque(maxlen=10) 
    time.sleep(2)

    try:
        with torch.no_grad():
            while True:
                start_time = time.time()
                frame = cam_manager.get_latest_frame()
                if frame is None: continue
                
                display_frame = frame.copy()
                face_tensor = face_prep.process(frame)
                
                if face_tensor is not None:
                    au_tensor = au_prep.process(frame)
                    face_tensor, au_tensor = face_tensor.to(device), au_tensor.to(device)

                    f_face = dan(face_tensor)
                    f_au = vgg(au_tensor)
                    probs, _ = fusion(f_face, f_au)
                    
                    # Cập nhật lịch sử để làm mượt kết quả
                    emotion_history.append(probs[0].cpu().numpy())
                    avg_probs = np.mean(emotion_history, axis=0)
                    
                    class_id = np.argmax(avg_probs)
                    confidence = avg_probs[class_id] * 100
                    
                    text = f"{EMOTION_LABELS[class_id]} ({confidence:.1f}%)"
                    
                    # Tiêu cực (Angry, Contempt, Disgust, Fear, Sadness) -> Đỏ
                    color = (0, 0, 255) if class_id in [0, 1, 2, 3, 5] else (0, 255, 0)
                    
                    cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                else:
                    emotion_history.clear()
                    cv2.putText(display_frame, "Khong tim thay phi cong!", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                fps = 1.0 / (time.time() - start_time)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("GCS Pilot Monitoring", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        cam_manager.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()