import torch
from torchvision import transforms
from PIL import Image

# 모델 경로와 동물 레이블 매핑
model_info = [
    {'animal': '고라니', 'path': './animals/gorani/best.pt'},
    {'animal': '족제비', 'path': './animals/jokjebi/best.pt'},
    {'animal': '너구리', 'path': './animals/neoguri/best.pt'},
    {'animal': '반달가슴곰', 'path': './animals/bandalgaseumgom/best.pt'},
    {'animal': '청설모', 'path': './animals/cheongseolmo/best.pt'},
    {'animal': '다람쥐', 'path': './animals/daramjwi/best.pt'},
    {'animal': '중대백로', 'path': './animals/joongdaebaekro/best.pt'},
    {'animal': '멧토끼', 'path': './animals/maettokki/best.pt'},
    {'animal': '멧돼지', 'path': './animals/maetdaeji/best.pt'},
    {'animal': '노루', 'path': './animals/noru/best.pt'},
    {'animal': '왜가리', 'path': './animals/waegari/best.pt'},
]

# 모델 로드 함수
def load_models(model_info):
    models = []
    for info in model_info:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=info['path'], force_reload=True)
        models.append((info['animal'], model))  # 동물 레이블과 모델 객체를 함께 저장
    return models

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")  # 색상 모드 변환
    return img

# 예측 함수
def ensemble_predict(models, image_path):
    img = preprocess_image(image_path)
    
    predictions = []
    for animal, model in models:
        results = model(img)  # 이제 모델이 객체이므로 호출 가능
        predictions.append((animal, results.xyxy[0]))  # 동물 레이블과 예측 결과 저장

    return predictions

# 모델 로드
models = load_models(model_info)

# 예측하고 싶은 이미지 파일 경로
image_path = './Dataset/images/neoguri_train.jpg'

# 앙상블 예측
predictions = ensemble_predict(models, image_path)

# 각 동물별로 최고 신뢰도를 저장할 딕셔너리
best_confidences = {info['animal']: 0.0 for info in model_info}

# 예측 결과를 출력하고 각 동물별 최고 신뢰도를 업데이트합니다.
for animal, model_predictions in predictions:
    for *box, conf, cls in model_predictions:
        conf = conf.item()  # tensor에서 값을 가져옵니다
        
        # 현재 동물의 최고 신뢰도 업데이트
        if conf > best_confidences[animal]:
            best_confidences[animal] = conf

# 전체 중 가장 높은 신뢰도를 가진 동물 찾기
best_animal = max(best_confidences, key=best_confidences.get)
best_overall_confidence = best_confidences[best_animal]

# 결과 출력
for info in model_info:
    animal = info['animal']
    if animal == best_animal and best_confidences[animal] > 0.5:  # 신뢰도 기준
        print(f"이 이미지는 {animal}입니다 (신뢰도: {best_confidences[animal]:.2f}): Y")
    else:
        print(f"이 이미지는 {animal}가 아닙니다 (신뢰도: {best_confidences[animal]:.2f}): N")