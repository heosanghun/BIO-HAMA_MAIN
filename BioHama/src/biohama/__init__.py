"""
BioHama: 바이오-인스파이어드 하이브리드 적응형 메타 아키텍처

뇌과학적 기반의 인공지능 시스템으로, 생물학적 신경망의
적응성과 학습 메커니즘을 모방하여 지능적인 의사결정과
문제 해결을 수행합니다.
"""

from .core.meta_router import MetaRouter
from .core.cognitive_state import CognitiveState
from .core.working_memory import WorkingMemory
from .core.decision_engine import DecisionEngine
from .core.attention_control import AttentionControl
from .communication.message_passing import MessagePassing
from .learning.bio_agrpo import BioAGRPO
from .biohama_system import BioHamaSystem
from .version import __version__

__all__ = [
    'MetaRouter',
    'CognitiveState', 
    'WorkingMemory',
    'DecisionEngine',
    'AttentionControl',
    'MessagePassing',
    'BioAGRPO',
    'BioHamaSystem',
    '__version__'
]

__version__ = __version__
