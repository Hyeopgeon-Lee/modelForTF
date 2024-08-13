import tensorflow as tf  # TensorFlow 라이브러리를 임포트하여 딥러닝 모델을 구축하고 학습에 사용
from tensorflow.keras import layers, models  # Keras의 레이어 및 모델 클래스를 임포트하여 신경망을 구성
import matplotlib.pyplot as plt  # 학습 결과 시각화를 위해 Matplotlib의 pyplot 모듈을 임포트
import pathlib  # 파일 경로를 쉽게 다루기 위해 pathlib 모듈을 임포트

# 학습시킬 이미지 파일이 저장된 디렉토리 경로를 지정
data_dir = pathlib.Path("images")

# 이미지 파일의 총 개수를 계산하여 출력 (학습에 사용할 데이터셋 크기 확인)
image_count = len(list(data_dir.glob("*/*.jpg")))
print("image_count:", image_count)

# 배치 크기와 이미지 크기(높이와 너비)를 설정
batch_size = 128  # 한 번에 처리할 이미지의 개수 (배치 크기)
img_height = 180  # 입력 이미지의 높이
img_width = 180  # 입력 이미지의 너비

# 학습용 데이터셋을 생성 (전체 데이터의 80%를 학습용으로 사용)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,  # 이미지 파일이 저장된 디렉토리 경로
    validation_split=0.2,  # 데이터셋의 20%를 검증용으로 분할
    subset="training",  # 학습용 데이터셋으로 사용
    seed=123,  # 데이터 분할 시 랜덤성을 유지하기 위한 시드값
    image_size=(img_height, img_width),  # 모든 이미지를 180x180 크기로 리사이즈
    batch_size=batch_size  # 한 번에 처리할 이미지의 개수 (배치 크기)
)

# 검증용 데이터셋을 생성 (전체 데이터의 20%를 검증용으로 사용)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,  # 이미지 파일이 저장된 디렉토리 경로
    validation_split=0.2,  # 데이터셋의 20%를 검증용으로 분할
    subset="validation",  # 검증용 데이터셋으로 사용
    seed=123,  # 데이터 분할 시 랜덤성을 유지하기 위한 시드값
    image_size=(img_height, img_width),  # 모든 이미지를 180x180 크기로 리사이즈
    batch_size=batch_size  # 한 번에 처리할 이미지의 개수 (배치 크기)
)

# 데이터 증강을 위한 레이어를 추가 (학습 데이터의 다양성을 높여 과적합 방지)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),  # 이미지를 수평으로 랜덤하게 뒤집음
    layers.RandomRotation(0.1),  # 이미지를 최대 10%까지 랜덤하게 회전시킴
    layers.RandomZoom(0.1),  # 이미지를 최대 10%까지 랜덤하게 확대/축소
])

# 데이터셋을 효율적으로 처리하기 위해 캐싱 및 프리패칭을 적용
AUTOTUNE = tf.data.AUTOTUNE  # TensorFlow가 자동으로 최적의 프리패칭 수를 선택하도록 설정
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # 데이터셋을 캐시하고, 셔플링한 후 프리패칭을 적용
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # 검증 데이터셋에도 프리패칭을 적용하여 학습 속도 향상

# 모델 구성 (순차적으로 레이어를 쌓아 신경망을 구성)
model = models.Sequential([
    layers.Rescaling(1. / 255),  # 모든 픽셀 값을 0-1 범위로 정규화
    layers.Conv2D(16, 3, padding="same", activation="relu"),  # 16개의 필터를 사용한 2D 합성곱 레이어, ReLU 활성화 함수 사용
    layers.MaxPooling2D(),  # 최대 풀링 레이어로 차원 축소 (특성 맵 크기 감소)
    layers.Conv2D(32, 3, padding="same", activation="relu"),  # 32개의 필터를 사용한 2D 합성곱 레이어, ReLU 활성화 함수 사용
    layers.MaxPooling2D(),  # 최대 풀링 레이어로 차원 축소 (특성 맵 크기 감소)
    layers.Conv2D(64, 3, padding="same", activation="relu"),  # 64개의 필터를 사용한 2D 합성곱 레이어, ReLU 활성화 함수 사용
    layers.MaxPooling2D(),  # 최대 풀링 레이어로 차원 축소 (특성 맵 크기 감소)

    layers.Flatten(),  # 다차원 배열을 1차원으로 펼침 (Dense 레이어에 입력하기 위해)
    layers.Dense(128, activation="relu"),  # 128개의 뉴런을 가진 완전 연결(Dense) 레이어, ReLU 활성화 함수 사용
    layers.Dropout(0.5),  # 과적합을 방지하기 위해 50%의 뉴런을 무작위로 비활성화
    layers.Dense(5)  # 출력 레이어, 클래스 수만큼 뉴런을 가짐 (클래스 수가 5이므로 5개의 뉴런)
])

# 모델 컴파일 (학습 과정에서 사용할 손실 함수, 최적화 알고리즘, 평가 지표 설정)
model.compile(optimizer="adam",  # Adam 최적화 알고리즘 사용
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 다중 클래스 분류를 위한 손실 함수
              metrics=["accuracy"])  # 모델의 정확도를 평가 지표로 설정

# 조기 종료 콜백 추가 (검증 손실이 개선되지 않으면 학습 조기 종료)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# 모델 체크포인트 콜백 추가 (가장 좋은 성능의 모델을 저장)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("model_checkpoint.keras", save_best_only=True)

# 모델 학습 (학습 데이터셋으로 모델을 학습시키고, 검증 데이터셋으로 성능을 평가)
history = model.fit(
    train_ds,  # 학습 데이터셋
    validation_data=val_ds,  # 검증 데이터셋
    epochs=10,  # 학습 에포크 수 (전체 데이터셋을 몇 번 반복 학습할지)
    callbacks=[early_stopping, model_checkpoint]  # 조기 종료와 체크포인트 콜백 사용
)

# 학습이 완료된 모델을 파일로 저장 (Keras 포맷으로 저장)
model.save("model/myFlowerModel2.keras")

# 학습 결과 시각화 (정확도와 손실의 변화를 그래프로 출력)
acc = history.history["accuracy"]  # 학습 데이터의 정확도 값들을 저장
val_acc = history.history["val_accuracy"]  # 검증 데이터의 정확도 값들을 저장
loss = history.history["loss"]  # 학습 데이터의 손실 값들을 저장
val_loss = history.history["val_loss"]  # 검증 데이터의 손실 값들을 저장

# 에포크 범위를 설정하여 그래프에서 x축으로 사용
epochs_range = range(len(acc))

# 정확도와 손실 그래프 그리기
plt.figure(figsize=(8, 8))  # 그래프의 크기를 설정
plt.subplot(1, 2, 1)  # 첫 번째 서브플롯: 학습 및 검증 정확도 그래프
plt.plot(epochs_range, acc, label="Training Accuracy")  # 학습 정확도 그래프
plt.plot(epochs_range, val_acc, label="Validation Accuracy")  # 검증 정확도 그래프
plt.legend(loc="lower right")  # 범례 위치 설정
plt.title("Training and Validation Accuracy")  # 그래프 제목

plt.subplot(1, 2, 2)  # 두 번째 서브플롯: 학습 및 검증 손실 그래프
plt.plot(epochs_range, loss, label="Training Loss")  # 학습 손실 그래프
plt.plot(epochs_range, val_loss, label="Validation Loss")  # 검증 손실 그래프
plt.legend(loc="upper right")  # 범례 위치 설정
plt.title("Training and Validation Loss")  # 그래프 제목

plt.show()  # 그래프 출력
