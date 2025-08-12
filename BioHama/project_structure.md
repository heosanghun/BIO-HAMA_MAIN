# BioHama 프로젝트 구조

## 📁 전체 구조

```
BioHama/
├── src/
│   └── biohama/                    # 메인 패키지
│       ├── __init__.py            # 패키지 초기화
│       ├── biohama_system.py      # 메인 시스템 클래스
│       ├── cli.py                 # 명령줄 인터페이스
│       ├── version.py             # 버전 정보
│       ├── core/                  # 핵심 구성 요소
│       │   ├── __init__.py
│       │   ├── base/              # 기본 인터페이스
│       │   │   ├── __init__.py
│       │   │   ├── module_interface.py
│       │   │   ├── router_interface.py
│       │   │   └── state_interface.py
│       │   ├── meta_router.py     # 메타 라우터
│       │   ├── cognitive_state.py # 인지 상태 관리
│       │   ├── working_memory.py  # 작업 메모리
│       │   ├── decision_engine.py # 의사결정 엔진
│       │   └── attention_control.py # 주의 제어
│       ├── communication/         # 통신 시스템
│       │   ├── __init__.py
│       │   ├── message_passing.py # 메시지 전달
│       │   ├── attention_graph.py # 주의 그래프
│       │   ├── hebbian_learning.py # 헤비안 학습
│       │   └── temporal_credit.py # 시간적 신용
│       ├── learning/              # 학습 시스템
│       │   ├── __init__.py
│       │   ├── bio_agrpo.py       # Bio-A-GRPO 알고리즘
│       │   ├── meta_learning.py   # 메타 학습
│       │   ├── neurotransmitter.py # 신경전달물질 시스템
│       │   ├── policy_optimizer.py # 정책 최적화
│       │   └── reward_system.py   # 보상 시스템
│       └── utils/                 # 유틸리티
│           ├── __init__.py
│           ├── config.py          # 설정 관리
│           ├── logging.py         # 로깅
│           ├── visualization.py   # 시각화
│           ├── profiling.py       # 성능 프로파일링
│           ├── memory_management.py # 메모리 관리
│           └── device_utils.py    # 디바이스 유틸리티
├── configs/                       # 설정 파일
│   └── base_config.yaml          # 기본 설정
├── examples/                      # 사용 예제
│   └── basic_usage.py            # 기본 사용법
├── tests/                        # 테스트 코드
│   └── test_biohama_system.py    # 시스템 테스트
├── docs/                         # 문서
├── README.md                     # 프로젝트 소개
├── requirements.txt              # 의존성 목록
├── setup.py                     # 설치 스크립트
└── project_structure.md         # 이 파일
```

## 🔧 핵심 구성 요소

### 1. 메인 시스템 (`biohama_system.py`)
- **역할**: 모든 BioHama 구성 요소들을 통합하는 메인 클래스
- **주요 기능**:
  - 시스템 초기화 및 통합
  - 입력 처리 및 라우팅
  - 훈련 관리
  - 체크포인트 저장/로드
  - 시스템 통계 수집

### 2. 메타 라우터 (`core/meta_router.py`)
- **역할**: 계층적 의사결정과 라우팅을 담당
- **주요 기능**:
  - 입력 분석 및 모듈 선택
  - 멀티헤드 어텐션 기반 라우팅
  - 탐색 vs 활용 전략
  - 라우팅 신뢰도 계산

### 3. 인지 상태 관리 (`core/cognitive_state.py`)
- **역할**: 시스템의 인지 상태를 추적하고 관리
- **주요 기능**:
  - 작업 메모리 상태 관리
  - 주의 상태 추적
  - 감정 상태 모델링
  - 메타인지 상태 관리

### 4. 작업 메모리 (`core/working_memory.py`)
- **역할**: 시스템의 작업 메모리를 관리
- **주요 기능**:
  - 메모리 항목 저장/검색
  - 우선순위 기반 메모리 관리
  - 메모리 통합 및 정리
  - 유사도 기반 검색

### 5. Bio-A-GRPO (`learning/bio_agrpo.py`)
- **역할**: 바이오-인스파이어드 적응형 정책 최적화
- **주요 기능**:
  - 신경전달물질 시스템 모방
  - 강화학습 기반 학습
  - 경험 리플레이
  - 정책 및 가치 네트워크 최적화

## 📋 파일별 상세 설명

### 기본 인터페이스 (`core/base/`)
- **`module_interface.py`**: 모든 모듈이 구현해야 하는 기본 인터페이스
- **`router_interface.py`**: 라우터의 기본 인터페이스
- **`state_interface.py`**: 상태 관리의 기본 인터페이스

### 통신 시스템 (`communication/`)
- **`message_passing.py`**: 모듈 간 메시지 전달 시스템
- **`attention_graph.py`**: 주의 기반 그래프 구조
- **`hebbian_learning.py`**: 헤비안 학습 메커니즘
- **`temporal_credit.py`**: 시간적 신용 할당

### 학습 시스템 (`learning/`)
- **`meta_learning.py`**: 메타 학습 알고리즘
- **`neurotransmitter.py`**: 신경전달물질 시스템
- **`policy_optimizer.py`**: 정책 최적화 도구
- **`reward_system.py`**: 보상 시스템

### 유틸리티 (`utils/`)
- **`config.py`**: 설정 파일 로드 및 관리
- **`logging.py`**: 로깅 시스템
- **`visualization.py`**: 시각화 도구
- **`profiling.py`**: 성능 프로파일링
- **`memory_management.py`**: 메모리 관리
- **`device_utils.py`**: 디바이스 관련 유틸리티

## 🚀 사용 방법

### 1. 설치
```bash
pip install -e .
```

### 2. 기본 사용법
```python
from biohama import BioHamaSystem

# 설정 생성
config = {...}  # 설정 딕셔너리

# 시스템 초기화
biohama = BioHamaSystem(config)
biohama.start()

# 입력 처리
result = biohama.process_input({'text': '안녕하세요'})

# 시스템 중지
biohama.stop()
```

### 3. CLI 사용법
```bash
# 데모 실행
biohama demo

# 시스템 초기화
biohama init --config configs/base_config.yaml

# 입력 처리
biohama process --config config.yaml --text "안녕하세요"

# 훈련
biohama train --config config.yaml --data training_data.json
```

## 🧪 테스트

### 테스트 실행
```bash
# 전체 테스트 실행
python -m pytest tests/

# 특정 테스트 실행
python tests/test_biohama_system.py
```

### 테스트 커버리지
```bash
python -m pytest tests/ --cov=biohama --cov-report=html
```

## 📊 성능 모니터링

### 시스템 통계 확인
```python
stats = biohama.get_system_statistics()
print(f"시스템 상태: {stats['system_status']}")
print(f"인지 상태: {stats['cognitive_state']}")
print(f"작업 메모리: {stats['working_memory']}")
```

### 체크포인트 관리
```python
# 체크포인트 저장
biohama.save_checkpoint('checkpoint.pkl')

# 체크포인트 로드
biohama.load_checkpoint('checkpoint.pkl')
```

## 🔧 설정

### 기본 설정 파일 (`configs/base_config.yaml`)
- 시스템 기본 설정
- 각 모듈별 파라미터
- 성능 모니터링 설정
- 메모리 관리 설정

### 설정 커스터마이징
```python
# 설정 로드
from biohama.utils.config import get_config
config = get_config()

# 설정 수정
config['meta_router']['input_dim'] = 256
config['device'] = 'cuda'
```

## 📈 확장성

### 새로운 모듈 추가
1. `core/base/module_interface.py` 구현
2. 메인 시스템에 등록
3. 메타 라우터에 라우팅 규칙 추가

### 새로운 학습 알고리즘 추가
1. `learning/` 디렉토리에 새 알고리즘 구현
2. BioHamaSystem에 통합
3. 설정 파일에 파라미터 추가

## 🐛 디버깅

### 로깅 설정
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 성능 프로파일링
```python
from biohama.utils.profiling import profile_performance
profile_performance(biohama)
```

## 📚 문서

- **README.md**: 프로젝트 개요 및 사용법
- **API 문서**: 각 클래스와 메서드의 상세 설명
- **아키텍처 가이드**: 시스템 설계 원칙
- **훈련 가이드**: 학습 알고리즘 사용법
- **배포 가이드**: 프로덕션 환경 배포 방법

---

이 구조는 BioHama 시스템의 모듈화된 설계를 반영하며, 각 구성 요소가 명확한 역할을 가지고 있습니다. 새로운 기능 추가나 수정이 필요할 때는 해당 모듈을 찾아서 작업하면 됩니다.
