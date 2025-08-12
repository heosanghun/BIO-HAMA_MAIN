"""
신경전달물질 시스템 모듈

BioHama 시스템의 뇌과학적 보상 메커니즘을 구현합니다.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import numpy as np

from ..core.base.module_interface import ModuleInterface


class NeurotransmitterSystem(ModuleInterface):
    """
    신경전달물질 시스템
    
    뇌의 신경전달물질 시스템을 모방하여 보상과 학습을 조절합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        신경전달물질 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 신경전달물질 설정
        self.dopamine_decay = config.get('dopamine_decay', 0.95)
        self.serotonin_modulation = config.get('serotonin_modulation', 0.1)
        self.norepinephrine_boost = config.get('norepinephrine_boost', 0.2)
        
        # 신경전달물질 수준
        self.dopamine_level = 0.5
        self.serotonin_level = 0.5
        self.norepinephrine_level = 0.5
        
        # 신경망 구성 요소들
        self._build_networks()
        
    def _build_networks(self):
        """신경망 구성 요소들을 구축합니다."""
        
        # 신경전달물질 예측 네트워크
        self.neurotransmitter_predictor = nn.Sequential(
            nn.Linear(self.config.get('state_dim', 128), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 도파민, 세로토닌, 노르에피네프린
        )
        
    def forward(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        신경전달물질 시스템 순전파
        
        Args:
            inputs: 입력 데이터
            context: 컨텍스트 정보
            
        Returns:
            출력 데이터
        """
        state = inputs.get('state')
        reward = inputs.get('reward', 0.0)
        
        if state is not None:
            # 상태를 텐서로 변환
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            
            # 신경전달물질 수준 예측
            neurotransmitter_pred = self.neurotransmitter_predictor(state)
            dopamine_pred, serotonin_pred, norepinephrine_pred = torch.split(neurotransmitter_pred, 1, dim=-1)
            
            # 신경전달물질 수준 업데이트
            self._update_neurotransmitter_levels(reward, dopamine_pred.item(), 
                                               serotonin_pred.item(), norepinephrine_pred.item())
        
        return {
            'dopamine': self.dopamine_level,
            'serotonin': self.serotonin_level,
            'norepinephrine': self.norepinephrine_level,
            'reward_modulation': self._compute_reward_modulation()
        }
    
    def _update_neurotransmitter_levels(self, reward: float, dopamine_pred: float, 
                                      serotonin_pred: float, norepinephrine_pred: float):
        """신경전달물질 수준을 업데이트합니다."""
        
        # 도파민: 보상 기반 학습
        if reward > 0:
            self.dopamine_level = min(1.0, self.dopamine_level + reward * 0.1)
        else:
            self.dopamine_level *= self.dopamine_decay
        
        # 세로토닌: 안정성과 만족감
        self.serotonin_level = (self.serotonin_level + serotonin_pred) / 2
        
        # 노르에피네프린: 각성과 주의
        self.norepinephrine_level = (self.norepinephrine_level + norepinephrine_pred) / 2
        
        # 수준 정규화
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)
        self.serotonin_level = np.clip(self.serotonin_level, 0.0, 1.0)
        self.norepinephrine_level = np.clip(self.norepinephrine_level, 0.0, 1.0)
    
    def _compute_reward_modulation(self) -> float:
        """신경전달물질 기반 보상 조절을 계산합니다."""
        
        # 도파민: 긍정적 보상 증폭
        dopamine_modulation = 1.0 + self.dopamine_level * 0.5
        
        # 세로토닌: 안정성 조절
        serotonin_modulation = 1.0 + (self.serotonin_level - 0.5) * self.serotonin_modulation
        
        # 노르에피네프린: 각성 증폭
        norepinephrine_modulation = 1.0 + self.norepinephrine_level * self.norepinephrine_boost
        
        # 종합 조절
        total_modulation = dopamine_modulation * serotonin_modulation * norepinephrine_modulation
        
        return total_modulation
    
    def get_neurotransmitter_state(self) -> Dict[str, float]:
        """
        신경전달물질 상태 반환
        
        Returns:
            신경전달물질 상태 딕셔너리
        """
        return {
            'dopamine': self.dopamine_level,
            'serotonin': self.serotonin_level,
            'norepinephrine': self.norepinephrine_level,
            'reward_modulation': self._compute_reward_modulation()
        }
    
    def reset(self) -> None:
        """신경전달물질 시스템 초기화"""
        super().reset()
        self.dopamine_level = 0.5
        self.serotonin_level = 0.5
        self.norepinephrine_level = 0.5

