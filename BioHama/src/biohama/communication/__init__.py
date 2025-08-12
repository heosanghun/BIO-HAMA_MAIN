"""
BioHama 통신 시스템 모듈

이 모듈은 BioHama 시스템의 모듈 간 통신을 담당합니다:
- 메시지 전달: 모듈 간 정보 교환
- 주의 그래프: 동적 주의 메커니즘
- 헤비안 학습: 연결 강화 학습
- 시간적 신용 할당: 장기 의존성 학습
"""

from .message_passing import MessagePassing
from .attention_graph import AttentionGraph
from .hebbian_learning import HebbianLearning
from .temporal_credit import TemporalCredit

__all__ = [
    "MessagePassing",
    "AttentionGraph",
    "HebbianLearning", 
    "TemporalCredit"
]

