from ultralytics import YOLO

# AI 모델 불러오기
model = YOLO('best.pt')  # AI 위치 (상대 위치) 넣어줘야 함

# 이미지 판단
results = model.predict(source="1.jpg", save=True, conf=0.40, classes=[0,1])
# .predict가 없어도 되는데 명시적으로 기능을 알도록 넣음.
# 순서대로 판단할 대상 사진 위치 (상대 위치), 저장 여부, 확률 몇 이상이면 맞다고 판단할 것인지 1 = 100%, 분류 필터링 (이건 건들지 말 것)

# 결과 보여주기
for result in results:
    result.show()