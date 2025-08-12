# BioHama: 바이오-인스파이어드 하이브리드 적응형 메타 아키텍처

BioHama는 뇌과학적 기반의 인공지능 시스템으로, 생물학적 신경망의 적응성과 학습 메커니즘을 모방하여 지능적인 의사결정과 문제 해결을 수행합니다.

## 🌟 주요 특징

- **뇌과학적 기반**: 생물학적 신경망의 구조와 기능을 모방
- **적응형 학습**: Bio-A-GRPO 알고리즘을 통한 지속적 학습
- **메타 라우팅**: 계층적 의사결정과 지능적 모듈 선택
- **인지 상태 관리**: 작업 메모리, 주의, 감정 상태 추적
- **신경전달물질 시스템**: 도파민, 세로토닌, 노르에피네프린 모방
- **모듈화 설계**: 확장 가능하고 유연한 아키텍처

## 🏗️ 아키텍처 개요

```
BioHama System
├── Core Components
│   ├── Meta Router (메타 라우터)
│   ├── Cognitive State (인지 상태 관리)
│   ├── Working Memory (작업 메모리)
│   ├── Decision Engine (의사결정 엔진)
│   └── Attention Control (주의 제어)
├── Communication
│   ├── Message Passing (메시지 전달)
│   ├── Attention Graph (주의 그래프)
│   ├── Hebbian Learning (헤비안 학습)
│   └── Temporal Credit (시간적 신용)
└── Learning
    ├── Bio-A-GRPO (바이오 적응형 정책 최적화)
    ├── Meta Learning (메타 학습)
    ├── Neurotransmitter System (신경전달물질 시스템)
    └── Reward System (보상 시스템)
```

## 🚀 설치 방법

### 요구사항

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/biohama.git
cd biohama

# 의존성 설치
pip install -r requirements.txt

# 개발 모드 설치
pip install -e .
```

## 📖 사용법

### 기본 사용법

```python
from biohama import BioHamaSystem

# 설정 생성
config = {
    'device': 'cpu',
    'meta_router': {
        'input_dim': 128,
        'hidden_dim': 256,
        'num_layers': 3
    },
    # ... 기타 설정
}

# 시스템 초기화
biohama = BioHamaSystem(config)
biohama.start()

# 입력 처리
result = biohama.process_input({
    'text': '안녕하세요, BioHama입니다.',
    'task_type': 'greeting'
})

print(f"선택된 모듈: {result['selected_module']}")
print(f"신뢰도: {result['routing_confidence']}")

# 시스템 중지
biohama.stop()
```

### 훈련 예제

```python
# 훈련 데이터 준비
training_data = [
    {
        'input': {'text': '훈련 텍스트', 'task_type': 'text_processing'},
        'context': {'training_step': 0},
        'feedback': {'reward': 0.8, 'success': True}
    }
    # ... 더 많은 훈련 데이터
]

# 훈련 실행
training_result = biohama.train(training_data)
print(f"훈련 완료: {training_result['training_samples']} 개 샘플")
```

### 체크포인트 저장/로드

```python
# 체크포인트 저장
biohama.save_checkpoint('biohama_checkpoint.pkl')

# 체크포인트 로드
biohama.load_checkpoint('biohama_checkpoint.pkl')
```

## 🔧 구성 요소 상세

### 1. 메타 라우터 (Meta Router)

계층적 의사결정과 라우팅을 담당하는 핵심 구성 요소입니다.

```python
from biohama import MetaRouter

router = MetaRouter({
    'input_dim': 128,
    'hidden_dim': 256,
    'num_heads': 8,
    'routing_temperature': 1.0,
    'exploration_rate': 0.1
})
```

### 2. 인지 상태 관리 (Cognitive State)

시스템의 인지 상태를 추적하고 관리합니다.

```python
from biohama import CognitiveState

cognitive_state = CognitiveState({
    'working_memory_dim': 256,
    'attention_dim': 128,
    'emotion_dim': 64
})

# 상태 업데이트
cognitive_state.update_working_memory(torch.randn(256), priority=0.8)
cognitive_state.update_emotion_state(valence=0.3, arousal=0.7, dominance=0.5)
```

### 3. Bio-A-GRPO

바이오-인스파이어드 적응형 정책 최적화 알고리즘입니다.

```python
from biohama import BioAGRPO

bio_agrpo = BioAGRPO({
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'dopamine_decay': 0.95,
    'serotonin_modulation': 0.1
})
```

## 📊 성능 모니터링

시스템의 성능과 상태를 모니터링할 수 있습니다.

```python
# 시스템 통계 확인
stats = biohama.get_system_statistics()

print(f"시스템 상태: {stats['system_status']}")
print(f"인지 상태: {stats['cognitive_state']}")
print(f"작업 메모리: {stats['working_memory']}")
print(f"메타 라우터: {stats['meta_router']}")
```

## 🧪 예제 실행

프로젝트에 포함된 예제를 실행해보세요:

```bash
cd examples
python basic_usage.py
```

## 📚 문서

- [API 문서](docs/api_reference.md)
- [아키텍처 가이드](docs/architecture.md)
- [훈련 가이드](docs/training_guide.md)
- [배포 가이드](docs/deployment.md)

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 연락처

- 프로젝트 링크: [https://github.com/your-username/biohama](https://github.com/your-username/biohama)
- 이슈 리포트: [https://github.com/your-username/biohama/issues](https://github.com/your-username/biohama/issues)

## 🙏 감사의 말

이 프로젝트는 다음과 같은 연구와 기술에 영감을 받았습니다:

- 뇌과학 및 인지과학 연구
- 강화학습 및 메타학습
- 신경망 아키텍처
- 생물학적 신경망 모델링

---

**BioHama** - 뇌과학적 기반의 차세대 인공지능 시스템
