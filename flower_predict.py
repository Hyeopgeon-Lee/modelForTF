import matplotlib.pyplot as plt  # 이미지 시각화를 위해 Matplotlib 라이브러리를 임포트
import numpy as np  # 수학적 계산과 배열 조작을 위한 NumPy 라이브러리 임포트
import tensorflow as tf  # 딥러닝 프레임워크인 TensorFlow 임포트
from tensorflow import keras  # TensorFlow의 고수준 API인 Keras를 임포트


# 이미지 전처리 함수 정의
def preprocess_image(image_path, img_height, img_width):
    """이미지를 로드하고 모델 입력에 맞게 전처리하는 함수."""

    # 주어진 경로에서 이미지를 로드하고, 지정된 크기(img_height, img_width)로 리사이즈
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))

    # 이미지를 NumPy 배열로 변환 (이때 픽셀 값은 [0, 255] 범위를 가짐)
    img_array = keras.preprocessing.image.img_to_array(img)

    # 모델이 배치(batch) 처리를 기대하므로, 차원을 추가하여 (1, height, width, 3) 형식의 4D 텐서로 변환
    img_array = tf.expand_dims(img_array, 0)

    # 변환된 4D 텐서를 반환
    return img_array


# 예측 함수 정의
def predict_image(model, img_array, class_names):
    """모델을 사용하여 이미지의 클래스를 예측하는 함수."""

    # 모델에 입력 이미지를 전달하여 예측을 수행 (로짓(logit) 값을 반환)
    predictions = model.predict(img_array)

    # 로짓 값을 소프트맥스 함수를 사용하여 확률 값으로 변환
    score = tf.nn.softmax(predictions[0])

    # 가장 높은 확률을 가진 클래스의 인덱스를 찾아 해당 클래스 이름과 확률을 계산
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # 예측된 클래스 이름과 그 확률을 반환
    return predicted_class, confidence


# 결과 시각화 함수 정의
def display_prediction(img, predicted_class, confidence):
    """예측 결과를 시각적으로 표시하는 함수."""

    # 원본 이미지를 화면에 표시 (img는 PIL 이미지 객체여야 함)
    plt.imshow(img)

    # 이미지 하단에 예측된 클래스와 확률을 텍스트로 표시
    plt.xlabel(f"This image most likely belongs to '{predicted_class}' with a '{confidence:.2f}' percent confidence.",
               color="black")  # 텍스트 색상을 검정으로 지정

    # 그리드를 비활성화하여 깔끔한 이미지 표시
    plt.grid(False)

    # x축의 눈금을 제거하여 이미지만 표시되도록 함
    plt.xticks([])

    # y축의 눈금을 제거하여 이미지만 표시되도록 함
    plt.yticks([])

    # 설정된 내용을 화면에 출력
    plt.show()


# 메인 실행 코드 블록
if __name__ == "__main__":
    # 이미지 전처리 시 사용할 이미지의 높이와 너비를 정의
    img_height = 180
    img_width = 180

    # 예측할 이미지 파일의 경로를 지정
    image_path = "test_images/sunflower.jpg"

    # 사전 학습된 모델 파일의 경로를 지정
    model_path = "model/myFlower2.keras"

    # 예측할 클래스 이름을 리스트로 정의 (모델이 학습한 클래스 순서에 맞게 나열)
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    # 이미지 전처리: 이미지 파일을 로드하고 모델 입력 형식에 맞게 변환
    img_array = preprocess_image(image_path, img_height, img_width)

    # 사전 학습된 Keras 모델을 로드
    loaded_model = tf.keras.models.load_model(model_path)

    # 로드된 모델을 사용하여 전처리된 이미지에 대한 예측을 수행
    predicted_class, confidence = predict_image(loaded_model, img_array, class_names)

    # 원본 이미지를 다시 로드 (시각화에서 사용할 PIL 이미지 객체로 로드)
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))

    # 예측된 결과를 원본 이미지와 함께 시각적으로 표시
    display_prediction(img, predicted_class, confidence)
