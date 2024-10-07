# YOLOv5 모델 로드 및 추론 예시
import torch

# YOLOv5 모델 로드 (best.pt)
# 고라니
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='./animals/gorani/best.pt')
# 족제비
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./animals/jokjebi/best.pt')

# 이미지에 대한 추론 (고라니 판별)
results = model('./Dataset/images/jok.jpg')

# 결과 출력
results.print()  # 콘솔에 결과 출력
results.show()   # 이미지에 박스를 그려서 보여줌

# 결과에서 감지된 객체 정보를 가져오기
predictions = results.xyxy[0].cpu().numpy()  # 예측된 객체들의 bbox 좌표, 확률, 클래스 ID 등

# YOLO 모델의 클래스 ID를 확인 (고라니 클래스의 ID는 데이터셋에 따라 다를 수 있음)
gorani_class_id = 0  # 고라니 클래스 ID (이 ID는 데이터셋에 따라 다를 수 있으니 확인 필요)

# 감지된 객체 중 고라니 여부 판단
gorani_detected = False
for *box, conf, cls in predictions:
    if int(cls) == gorani_class_id and conf > 0.9:  # 클래스가 고라니이고, 확률이 0.5 이상인 경우
        gorani_detected = True
        break

# 고라니 여부 출력
if gorani_detected:
    print("이 이미지는 고라니입니다: Y")
else:
    print("이 이미지는 고라니가 아닙니다: N")
