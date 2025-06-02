# Logisim Neural Network Implementation

MNIST 손글씨 숫자 인식을 위한 다층 퍼셉트론(MLP) 신경망의 완전한 하드웨어 구현 프로젝트입니다.

## 프로젝트 개요

본 프로젝트는 **Logisim Evolution**을 사용하여 MNIST 데이터셋의 손글씨 숫자(0-9)를 인식하는 신경망을 하드웨어로 구현합니다. 16×16 이진화된 이미지를 입력으로 받아 실시간으로 숫자를 분류할 수 있는 완전한 디지털 회로입니다.

## 신경망 구조

- **입력층**: 16×16 = 256개 픽셀
- **은닉층**: 128개 뉴런 (ReLU 활성화)
- **출력층**: 10개 뉴런 (0-9 분류)
- **총 파라미터**: 약 34,000개 가중치

## 요구사항

- [Logisim Evolution](https://github.com/logisim-evolution/logisim-evolution) (최신 버전 권장)
- Java 8 이상


## 주요 구성 요소

### 하드웨어 모듈
- **입력 처리 모듈**: 16×16 픽셀 데이터 입력 및 전처리
- **가중치 메모리**: 훈련된 신경망 가중치 저장
- **연산 유닛**: 행렬 곱셈 및 활성화 함수 계산
- **출력 분류기**: 최종 분류 결과 출력

### 활성화 함수
- **ReLU**: 은닉층에서 사용

## 학습 자료

- [신경망 기초 이론](docs/neural_network_basics.md)
- [Logisim 사용법](docs/logisim_guide.md)
- [하드웨어 설계 문서](docs/hardware_design.md)


# 사용 방법

## 1. 환경 설정

### 필수 소프트웨어
- **Logisim Evolution 3.8.0 이상** ([다운로드](https://github.com/logisim-evolution/logisim-evolution/releases))
- **Python 3.8 이상** (가중치 생성용)

### Python 의존성 설치

## 2. 가중치 파일 생성

### 2.1 Python 스크립트 실행


### 2.2 생성되는 파일 확인
- `weights/layer1_weights_shifted.txt` - 입력→은닉층 가중치
- `weights/layer2_weights_shifted.txt` - 은닉→출력층 가중치  
- `weights/layer1_bias_shifted.txt` - 은닉층 바이어스
- `weights/layer2_bias_shifted.txt` - 출력층 바이어스

## 3. Logisim 회로 실행

### 3.1 회로 파일 열기
1. **Logisim Evolution** 실행
2. **File → Open** 선택
3. `MLP_circit.circ` 파일 열기

### 3.2 가중치 ROM 로딩
1. **HiddenLayer** 서브회로 더블클릭하여 진입
2. **가중치 ROM** 컴포넌트 우클릭 → **Load Image** 선택
3. `weights/layer1_weights_shifted.txt` 파일 로드
4. **바이어스 ROM** 컴포넌트에 `weights/layer1_bias_shifted.txt` 로드
5. **OutputLayer** 서브회로에서도 동일하게 layer2 파일들 로드

### 3.3 시뮬레이션 설정
1. **Simulate → Tick Frequency** 선택
2. **4.1 KHz** 이상으로 설정
3. **Simulate → Ticks Enabled** 체크 (Ctrl+K)

## 4. 테스트 실행

### 4.1 기본 테스트
1. **시뮬레이션 시작**: Ctrl+K 또는 Simulate → Ticks Enabled
2. **Clock** 신호가 동작하는지 확인
3. **RGB Video**에 조이스틱을 활용하여 숫자 작성
4. **start** input을 1로 변경하면 연산 시작
5. 30-40초 후 **7-Segment Display**에서 결과 확인 (0-9 숫자)

### 4.2 단계별 디버깅
1. **시뮬레이션 속도 조정**: Simulate → Tick Frequency → **1 Hz**
2. **프로브 도구** 사용하여 신호 확인
3. **InputLayer** 출력 확인: 256개 픽셀 값이 올바른지
4. **HiddenLayer** 출력 확인: 128개 뉴런 출력 값
5. **OutputLayer** 출력 확인: 10개 클래스 점수

## 개발자

- **Lloyd Kwak** - [GitHub](https://github.com/lloydkwak)

