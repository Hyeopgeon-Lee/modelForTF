# 🌼 꽃 이름 맞추기 앱 - TensorFlow를 활용한 학습 모델 생성

> **Google TensorFlow로 학습 모델을 생성하고 꽃 이름을 맞추는 인공지능 앱**  
> TensorFlow를 활용하여 꽃 데이터를 학습시키고, 학습 모델을 기반으로 꽃 이름을 예측하는 프로젝트입니다.  
> 공유되는 코드는 한국폴리텍대학 서울강서캠퍼스 빅데이터과 수업에서 사용된 코드입니다.

---

### 📚 **작성자**
- **한국폴리텍대학 서울강서캠퍼스 빅데이터과**  
- **이협건 교수**  
- ✉️ [hglee67@kopo.ac.kr](mailto:hglee67@kopo.ac.kr)  
- 🔗 [빅데이터학과 입학 상담 오픈채팅방](https://open.kakao.com/o/gEd0JIad)

---

## 🚀 주요 실습 내용

1. **학습 모델 만들기 기초**  
   - 꽃 데이터를 사용하여 딥러닝 모델 생성.
2. **학습 모델 과적합 해결하기**  
   - Dropout과 Early Stopping을 활용한 과적합 방지.
3. **꽃 이름 맞추기**  
   - 학습된 모델을 사용하여 꽃 이미지를 입력받아 꽃 이름 예측.

---

## 🛠️ 주요 기술 스택

- **TensorFlow**: 2.17.x  
- **Python**: 3.10.x  

---

## 📩 문의 및 입학 상담

- 📧 **이메일**: [hglee67@kopo.ac.kr](mailto:hglee67@kopo.ac.kr)  
- 💬 **입학 상담 오픈채팅방**: [바로가기](https://open.kakao.com/o/gEd0JIad)

---

## 💡 **우리 학과 소개**
- 한국폴리텍대학 서울강서캠퍼스 빅데이터과는 **클라우드 컴퓨팅, 인공지능, 빅데이터 기술**을 활용하여 소프트웨어 개발자를 양성하는 학과입니다.  
- 학과에 대한 더 자세한 정보는 [학과 홈페이지](https://www.kopo.ac.kr/kangseo/content.do?menu=1547)를 참고하세요.

---

## 📦 **설치 및 실행 방법**

### 1. 레포지토리 클론
- 아래 명령어를 실행하여 레포지토리를 클론합니다.

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. 가상환경 설정 및 의존성 설치
- Python 가상환경을 설정한 뒤, 필요한 패키지를 설치합니다.

```bash
코드 복사
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 학습 데이터 준비
- 꽃 데이터셋을 다운로드하고 프로젝트 디렉토리에 추가합니다.
- 데이터셋 경로는 코드에서 명시적으로 설정해야 합니다.
   
### 4. 학습 모델 생성
- 아래 명령어를 실행하여 모델을 학습시킵니다.

```bash
python flower_model1.py
python flower_model2.py
```

### 5. 꽃 이름 예측
- 학습된 모델을 사용하여 꽃 이름을 예측합니다.

```bash
python predict_flower.py
```

