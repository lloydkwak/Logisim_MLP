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

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 개발자

- **Lloyd Kwak** - [GitHub](https://github.com/lloydkwak)

